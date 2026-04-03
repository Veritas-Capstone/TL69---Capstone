#!/usr/bin/env python3
"""
Threshold calibration for bias detection.
Step 1: Cache model outputs (logits) on BASIL
Step 2: Grid search over logit offsets + temperature scaling
Step 3: Report best thresholds for accuracy, macro-F1, and balanced
"""

import json
import time
from types import SimpleNamespace
import os
import sys
import numpy as np
from collections import Counter
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from basil_loader import load_basil

LABELS = ["Left", "Center", "Right"]
CACHE_FILE = "basil_logits_cache.json"

# config settings
config = SimpleNamespace(
    basil_dir="../../datasets/other_data/BASIL",
    bias_model="../../models/demo_models/bias_detector",
    pol_model="../../models/demo_models/politicalness_filter",
    device=None,
    use_fp16=True,
    cache_file=CACHE_FILE,
    cache_only=False,
    search_only=False, 
    fine=False, 
    label_key="source_label", 
    output_json="calibration_results.json",
)



# Step 1: Cache model outputs
def cache_model_outputs(basil_dir, bias_model_path, pol_model_path,
                        device=None, use_fp16=True):
    from pipeline import BiasDetector

    detector = BiasDetector(
        bias_path=bias_model_path,
        pol_path=pol_model_path,
        device=device,
        use_fp16=use_fp16,
    )

    # Load BASIL with both label sources
    articles_source = load_basil(basil_dir, label_source="source")
    articles_ann = load_basil(basil_dir, label_source="annotation")

    # Build annotation label lookup
    ann_labels = {}
    for a in articles_ann:
        key = f"{a['event_id']}_{a['file_suffix']}"
        ann_labels[key] = a["true_label"]

    cache = []
    total_time = 0

    for i, article in enumerate(articles_source):
        # Get raw model outputs - we need to access internals
        import torch

        body = article["body"]
        headline = article["headline"]
        text = f"{headline}\n\n{body}" if headline else body

        # politicalness score
        pol_inputs = detector.pol_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(detector.device)

        with torch.no_grad():
            if detector.use_fp16:
                with torch.cuda.amp.autocast():
                    pol_out = detector.pol_model(**pol_inputs)
            else:
                pol_out = detector.pol_model(**pol_inputs)

        pol_logits = pol_out.logits.cpu().float().numpy()[0]
        pol_probs = torch.softmax(pol_out.logits.float(), dim=-1).cpu().numpy()[0]

        # bias classification
        bias_inputs = detector.bias_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(detector.device)

        with torch.no_grad():
            if detector.use_fp16:
                with torch.cuda.amp.autocast():
                    bias_out = detector.bias_model(**bias_inputs)
            else:
                bias_out = detector.bias_model(**bias_inputs)

        bias_logits = bias_out.logits.cpu().float().numpy()[0]
        bias_probs = torch.softmax(bias_out.logits.float(), dim=-1).cpu().numpy()[0]

        logits_ordered = [float(bias_logits[j]) for j in range(min(3, len(bias_logits)))]
        probs_ordered = [float(bias_probs[j]) for j in range(min(3, len(bias_probs)))]

        id2label = detector.bias_model.config.id2label
        label_order = [id2label.get(j, f"idx_{j}") for j in range(len(id2label))]

        key = f"{article['event_id']}_{article['file_suffix']}"
        ann_label = ann_labels.get(key, "")

        # pipeline's original prediction
        result = detector.analyze(body=body, headline=headline)
        total_time += result["elapsed"]

        cache.append({
            "index": i,
            "source": article["source"],
            "source_label": article["true_label"], 
            "annotation_label": ann_label,
            "event_id": article["event_id"],
            "file_suffix": article["file_suffix"],
            "headline": headline[:100],
            "bias_logits": logits_ordered, 
            "bias_probs": probs_ordered, 
            "pol_logits": pol_logits.tolist(),
            "pol_probs": pol_probs.tolist(),
            "political_confidence": float(result["political_confidence"]),
            "original_prediction": result["prediction"] or "Center",
            "label_order": label_order,
        })

        # verify cached logits reproduce the original prediction
        cached_pred = LABELS[int(np.argmax(logits_ordered))]
        orig_pred = result["prediction"] or "Center"
        if cached_pred != orig_pred and result["prediction"] is not None:
            # mismatch could indicate label mapping error
            if i < 5:  # only warn for first few
                print(f" Prediction mismatch at {i}: "
                      f"cached={cached_pred} vs pipeline={orig_pred} "
                      f"logits={logits_ordered}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(articles_source)}] cached..")

    # save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n  Cached {len(cache)} articles to {CACHE_FILE}")
    print(f"  Inference time: {total_time:.1f}s ({total_time/len(cache):.2f}s/article)")

    # verify label mapping
    print(f"\n  Model label order: {cache[0]['label_order']}")
    print(f"  Logits order: [Left, Center, Right]")
    print(f"  Sample logits: {cache[0]['bias_logits']}")
    print(f"  Sample probs:  {cache[0]['bias_probs']}")

    return cache


# grid search over calibration parameters
def softmax(x):
    x = np.array(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def compute_metrics_fast(predictions, ground_truths):
    n = len(predictions)
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    accuracy = correct / n if n > 0 else 0

    per_class_f1 = []
    per_class = {}
    for label in LABELS:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class_f1.append(f1)
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1,
                            "support": sum(1 for g in ground_truths if g == label)}

    macro_f1 = np.mean(per_class_f1)
    return accuracy, macro_f1, per_class


def predict_with_calibration(cache, bias_offsets, temperature, pol_threshold=None,
                             label_key="source_label"):
    predictions = []
    ground_truths = []

    for item in cache:
        gt = item[label_key]
        if not gt:
            continue

        # apply calibration
        logits = np.array(item["bias_logits"])
        calibrated = (logits + np.array(bias_offsets)) / temperature
        probs = softmax(calibrated)

        # politicalness gate
        if pol_threshold is not None and item["political_confidence"] < pol_threshold:
            pred = "Center"
        else:
            pred_idx = np.argmax(probs)
            pred = LABELS[pred_idx]

        predictions.append(pred)
        ground_truths.append(gt)

    return predictions, ground_truths


def grid_search(cache, label_key="source_label", fine=False):
    print(f"  GRID SEARCH (labels={label_key})")

    if fine:
        bias_L_range = np.arange(-4.0, 1.0, 0.1)
        bias_C_range = np.arange(-1.0, 6.0, 0.1)
        bias_R_range = [0.0] 
        temp_range = np.arange(0.5, 3.0, 0.1)
        pol_range = [None] 
    else:
        # coarse search
        bias_L_range = np.arange(-5.0, 2.0, 0.5)
        bias_C_range = np.arange(-2.0, 8.0, 0.5)
        bias_R_range = [0.0] 
        temp_range = np.arange(0.5, 4.0, 0.25)
        pol_range = [None, 0.3, 0.5, 0.7, 0.8, 0.9]

    total_combos = (len(bias_L_range) * len(bias_C_range) * len(bias_R_range) *
                    len(temp_range) * len(pol_range))

    print(f"  Search space: {total_combos:,} combinations")
    print(f"    bias_L: [{bias_L_range[0]:.1f}, {bias_L_range[-1]:.1f}] "
          f"({len(bias_L_range)} steps)")
    print(f"    bias_C: [{bias_C_range[0]:.1f}, {bias_C_range[-1]:.1f}] "
          f"({len(bias_C_range)} steps)")
    print(f"    temp:   [{temp_range[0]:.2f}, {temp_range[-1]:.2f}] "
          f"({len(temp_range)} steps)")
    print(f"    pol:    {pol_range}")

    best_acc = {"accuracy": 0, "params": None}
    best_f1 = {"macro_f1": 0, "params": None}
    best_balanced = {"score": 0, "params": None} 

    results = []
    t_start = time.time()
    checked = 0

    for bias_L, bias_C, bias_R, temp, pol_thresh in product(
        bias_L_range, bias_C_range, bias_R_range, temp_range, pol_range
    ):
        offsets = [float(bias_L), float(bias_C), float(bias_R)]

        preds, gts = predict_with_calibration(
            cache, offsets, float(temp), pol_thresh, label_key
        )

        acc, macro_f1, per_class = compute_metrics_fast(preds, gts)

        params = {
            "bias_L": round(float(bias_L), 2),
            "bias_C": round(float(bias_C), 2),
            "bias_R": round(float(bias_R), 2),
            "temperature": round(float(temp), 2),
            "pol_threshold": pol_thresh,
        }

        # track best results
        if acc > best_acc["accuracy"]:
            best_acc = {"accuracy": acc, "macro_f1": macro_f1,
                        "per_class": per_class, "params": params}

        if macro_f1 > best_f1["macro_f1"]:
            best_f1 = {"accuracy": acc, "macro_f1": macro_f1,
                        "per_class": per_class, "params": params}

        balanced = 0.5 * acc + 0.5 * macro_f1
        if balanced > best_balanced["score"]:
            best_balanced = {"score": balanced, "accuracy": acc,
                             "macro_f1": macro_f1, "per_class": per_class,
                             "params": params}

        # store all results above a threshold for analysis
        if acc > 0.45 or macro_f1 > 0.35:
            results.append({
                "accuracy": acc, "macro_f1": macro_f1,
                "per_class": per_class, "params": params,
            })

        checked += 1
        if checked % 50000 == 0:
            elapsed = time.time() - t_start
            rate = checked / elapsed
            remaining = (total_combos - checked) / rate
            print(f"  [{checked:,}/{total_combos:,}] "
                  f"best_acc={best_acc['accuracy']:.3f} "
                  f"best_f1={best_f1['macro_f1']:.3f} "
                  f"(~{remaining:.0f}s remaining)")

    # report results
    print(f"  BEST BY ACCURACY")
    _print_result(best_acc)

    print(f"  BEST BY MACRO-F1")
    _print_result(best_f1)

    print(f"  BEST BALANCED (0.5*acc + 0.5*f1)")
    _print_result(best_balanced)

    # show top 10 configurations 
    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    print(f"  TOP 10 BY MACRO-F1")
    print(f"  {'Acc':>6} {'MF1':>6} {'F1(L)':>6} {'F1(C)':>6} {'F1(R)':>6}  "
          f"{'bL':>5} {'bC':>5} {'T':>5} {'pol':>5}")
    for r in results[:10]:
        p = r["params"]
        pc = r["per_class"]
        pol_str = f"{p['pol_threshold']:.1f}" if p["pol_threshold"] else " None"
        print(f"  {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} "
              f"{pc['Left']['f1']:>6.3f} {pc['Center']['f1']:>6.3f} "
              f"{pc['Right']['f1']:>6.3f}  "
              f"{p['bias_L']:>5.1f} {p['bias_C']:>5.1f} "
              f"{p['temperature']:>5.2f} {pol_str:>5}")

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    print(f"  TOP 10 BY ACCURACY")
    print(f"  {'Acc':>6} {'MF1':>6} {'F1(L)':>6} {'F1(C)':>6} {'F1(R)':>6}  "
          f"{'bL':>5} {'bC':>5} {'T':>5} {'pol':>5}")
    for r in results[:10]:
        p = r["params"]
        pc = r["per_class"]
        pol_str = f"{p['pol_threshold']:.1f}" if p["pol_threshold"] else " None"
        print(f"  {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} "
              f"{pc['Left']['f1']:>6.3f} {pc['Center']['f1']:>6.3f} "
              f"{pc['Right']['f1']:>6.3f}  "
              f"{p['bias_L']:>5.1f} {p['bias_C']:>5.1f} "
              f"{p['temperature']:>5.02f} {pol_str:>5}")

    return {
        "best_accuracy": best_acc,
        "best_macro_f1": best_f1,
        "best_balanced": best_balanced,
        "top_results": results[:50],
    }


def _print_result(result):
    p = result.get("params", {})
    print(f"  Accuracy:  {result.get('accuracy', 0):.1%}")
    print(f"  Macro F1:  {result.get('macro_f1', 0):.3f}")
    if "per_class" in result:
        for label in LABELS:
            pc = result["per_class"].get(label, {})
            print(f"    {label:<8} P={pc.get('precision',0):.3f} "
                  f"R={pc.get('recall',0):.3f} F1={pc.get('f1',0):.3f} "
                  f"(n={pc.get('support',0)})")
    print(f"  Parameters:")
    print(f"    bias_L={p.get('bias_L', 0):.2f}  "
          f"bias_C={p.get('bias_C', 0):.2f}  "
          f"bias_R={p.get('bias_R', 0):.2f}")
    print(f"    temperature={p.get('temperature', 1.0):.2f}  "
          f"pol_threshold={p.get('pol_threshold')}")


# detailed analysis of a specific configuration
def analyze_configuration(cache, bias_offsets, temperature, pol_threshold=None,
                          label_key="source_label"):
    preds, gts = predict_with_calibration(
        cache, bias_offsets, temperature, pol_threshold, label_key
    )
    acc, macro_f1, per_class = compute_metrics_fast(preds, gts)

    print(f"  DETAILED ANALYSIS")
    print(f"  bias=[{bias_offsets[0]:.2f}, {bias_offsets[1]:.2f}, {bias_offsets[2]:.2f}]  "
          f"T={temperature:.2f}  pol={pol_threshold}")
    print(f"  Accuracy: {acc:.1%}  Macro-F1: {macro_f1:.3f}")

    for label in LABELS:
        pc = per_class[label]
        print(f"  {label:<8} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")

    # confusion matrix
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for p, g in zip(preds, gts):
        confusion[g][p] += 1

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>12} {'Left':>8} {'Center':>8} {'Right':>8}")
    for true in LABELS:
        row = "".join(f"{confusion[true][p]:>8}" for p in LABELS)
        print(f"  {true:>12}{row}")

    # prediction distribution
    pred_dist = Counter(preds)
    print(f"\n  Prediction distribution:")
    for label in LABELS:
        print(f"    {label}: {pred_dist.get(label, 0)}")

    # Per-event analysis
    events = {}
    for item, pred, gt in zip(cache, preds, gts):
        eid = item.get("event_id", "")
        if eid:
            events.setdefault(eid, []).append((pred, gt, item["source"]))

    triplet_correct = sum(
        1 for articles in events.values()
        if len(articles) == 3 and all(p == g for p, g, _ in articles)
    )
    print(f"\n  Triplets all correct: {triplet_correct}/{len(events)}")

    return acc, macro_f1, per_class


def main():
    cfg = config

    CACHE_FILE_PATH = cfg.cache_file

    # step 1: Cache
    if not cfg.search_only:
        print("  Step 1: Caching model outputs on BASIL..")
        cache = cache_model_outputs(
            cfg.basil_dir, cfg.bias_model, cfg.pol_model,
            device=cfg.device, use_fp16=cfg.use_fp16,
        )
        if cfg.cache_only:
            return

    if not os.path.isfile(CACHE_FILE_PATH):
        print(f"Cache file not found: {CACHE_FILE_PATH}")
        sys.exit(1)

    with open(CACHE_FILE_PATH) as f:
        cache = json.load(f)
    print(f"\n  Loaded {len(cache)} cached articles from {CACHE_FILE_PATH}")

    sample_logits = cache[0]["bias_logits"]
    all_zero = all(abs(l) < 1e-6 for l in sample_logits)
    if all_zero:
        print(f"\nSample logits are all zero: {sample_logits}")
        sys.exit(1)
    else:
        print(f"  Sample logits: {sample_logits} ✓")
        print(f"  Logit ranges: L=[{min(c['bias_logits'][0] for c in cache):.2f}, "
              f"{max(c['bias_logits'][0] for c in cache):.2f}]  "
              f"C=[{min(c['bias_logits'][1] for c in cache):.2f}, "
              f"{max(c['bias_logits'][1] for c in cache):.2f}]  "
              f"R=[{min(c['bias_logits'][2] for c in cache):.2f}, "
              f"{max(c['bias_logits'][2] for c in cache):.2f}]")

    orig_preds = [c["original_prediction"] for c in cache]
    orig_gts = [c["source_label"] for c in cache]
    orig_acc = sum(p == g for p, g in zip(orig_preds, orig_gts)) / len(orig_preds)
    print(f"  Original pipeline accuracy: {orig_acc:.1%}")

    if cfg.analyze:
        bias_L, bias_C, bias_R, temp, pol = cfg.analyze
        pol_thresh = None if pol < 0 else pol
        analyze_configuration(
            cache, [bias_L, bias_C, bias_R], temp, pol_thresh, cfg.label_key
        )
        return

    # step 2: grid search
    print(f"  Step 2: Grid search over calibration parameters..")

    results = grid_search(cache, label_key=cfg.label_key, fine=cfg.fine)

    # run against annotation labels for comparison
    if cfg.label_key == "source_label":
        print(f"  Cross-check: applying best source-label params to annotation labels")

        for name, best in [("best_accuracy", results["best_accuracy"]),
                           ("best_macro_f1", results["best_macro_f1"]),
                           ("best_balanced", results["best_balanced"])]:
            p = best["params"]
            offsets = [p["bias_L"], p["bias_C"], p["bias_R"]]
            print(f"\n {name} params -> annotation labels")
            analyze_configuration(
                cache, offsets, p["temperature"],
                p["pol_threshold"], "annotation_label"
            )

    # save results
    if cfg.output_json:
        output = {
            "label_key": cfg.label_key,
            "fine": cfg.fine,
            "n_articles": len(cache),
            "original_accuracy": orig_acc,
            "best_accuracy": {
                "accuracy": results["best_accuracy"]["accuracy"],
                "macro_f1": results["best_accuracy"]["macro_f1"],
                "params": results["best_accuracy"]["params"],
            },
            "best_macro_f1": {
                "accuracy": results["best_macro_f1"]["accuracy"],
                "macro_f1": results["best_macro_f1"]["macro_f1"],
                "params": results["best_macro_f1"]["params"],
            },
            "best_balanced": {
                "accuracy": results["best_balanced"].get("accuracy", 0),
                "macro_f1": results["best_balanced"].get("macro_f1", 0),
                "params": results["best_balanced"]["params"],
            },
        }
        with open(cfg.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {cfg.output_json}")


if __name__ == "__main__":
    main()