#!/usr/bin/env python3
"""
Threshold Calibration for Bias Detection Pipeline
===================================================
Step 1: Run inference once on BASIL, cache raw logits/probs
Step 2: Exhaustive grid search over calibration parameters
Step 3: Report best thresholds for each metric (accuracy, macro-F1)

Calibration strategies:
  A) Logit bias offsets:  logits' = logits + [bias_L, bias_C, bias_R]
  B) Temperature scaling:  probs = softmax(logits / T)
  C) Combined:            probs = softmax((logits + bias) / T)
  D) Per-class confidence thresholds with Center default

Usage:
  # Step 1: Cache model outputs (requires GPU)
  python calibrate_thresholds.py --basil-dir BASIL --cache-only

  # Step 2: Grid search (CPU-only, fast)
  python calibrate_thresholds.py --basil-dir BASIL --search-only

  # Both steps
  python calibrate_thresholds.py --basil-dir BASIL

  # Fine-grained search around a known good point
  python calibrate_thresholds.py --basil-dir BASIL --search-only --fine
"""

import json
import time
import argparse
import os
import sys
import numpy as np
from collections import Counter
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basil_loader import load_basil

LABELS = ["Left", "Center", "Right"]
CACHE_FILE = "basil_logits_cache.json"


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Cache model outputs
# ═══════════════════════════════════════════════════════════════════════

def cache_model_outputs(basil_dir, bias_model_path, pol_model_path,
                        device=None, use_fp16=True):
    """
    Run inference on all BASIL articles and cache logits + metadata.
    This is the expensive step (requires GPU). Only needs to run once.
    """
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

        # ── Politicalness score ──
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

        # ── Bias classification ──
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

        # pipeline_v2 uses LABELS[pred_idx] directly, so:
        #   index 0 = Left, index 1 = Center, index 2 = Right
        # Just take logits in order — no name matching needed.
        logits_ordered = [float(bias_logits[j]) for j in range(min(3, len(bias_logits)))]
        probs_ordered = [float(bias_probs[j]) for j in range(min(3, len(bias_probs)))]

        # Get model's id2label for reference
        id2label = detector.bias_model.config.id2label
        label_order = [id2label.get(j, f"idx_{j}") for j in range(len(id2label))]

        # Annotation label
        key = f"{article['event_id']}_{article['file_suffix']}"
        ann_label = ann_labels.get(key, "")

        # Get the pipeline's original prediction
        result = detector.analyze(body=body, headline=headline)
        total_time += result["elapsed"]

        cache.append({
            "index": i,
            "source": article["source"],
            "source_label": article["true_label"],  # source-based
            "annotation_label": ann_label,            # expert annotation
            "event_id": article["event_id"],
            "file_suffix": article["file_suffix"],
            "headline": headline[:100],
            "bias_logits": logits_ordered,      # [Left, Center, Right]
            "bias_probs": probs_ordered,         # [Left, Center, Right]
            "pol_logits": pol_logits.tolist(),
            "pol_probs": pol_probs.tolist(),
            "political_confidence": float(result["political_confidence"]),
            "original_prediction": result["prediction"] or "Center",
            "label_order": label_order,
        })

        # Verify cached logits reproduce the original prediction
        cached_pred = LABELS[int(np.argmax(logits_ordered))]
        orig_pred = result["prediction"] or "Center"
        if cached_pred != orig_pred and result["prediction"] is not None:
            # Mismatch could indicate label mapping error
            if i < 5:  # Only warn for first few
                print(f"  ⚠ Prediction mismatch at {i}: "
                      f"cached={cached_pred} vs pipeline={orig_pred} "
                      f"logits={logits_ordered}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(articles_source)}] cached...")

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n  Cached {len(cache)} articles to {CACHE_FILE}")
    print(f"  Inference time: {total_time:.1f}s ({total_time/len(cache):.2f}s/article)")

    # Verify label mapping
    print(f"\n  Model label order: {cache[0]['label_order']}")
    print(f"  Logits order: [Left, Center, Right]")
    print(f"  Sample logits: {cache[0]['bias_logits']}")
    print(f"  Sample probs:  {cache[0]['bias_probs']}")

    return cache


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Grid search over calibration parameters
# ═══════════════════════════════════════════════════════════════════════

def softmax(x):
    """Numerically stable softmax."""
    x = np.array(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def compute_metrics_fast(predictions, ground_truths):
    """Fast metrics computation for grid search."""
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
    """
    Apply calibration to cached logits and return predictions + ground truths.

    Args:
        cache:         Cached model outputs
        bias_offsets:  [offset_L, offset_C, offset_R] added to logits
        temperature:   Temperature for softmax (>1 = softer, <1 = sharper)
        pol_threshold: If set, articles below this political confidence → Center
        label_key:     Which ground truth to use
    """
    predictions = []
    ground_truths = []

    for item in cache:
        gt = item[label_key]
        if not gt:
            continue

        # Apply calibration
        logits = np.array(item["bias_logits"])
        calibrated = (logits + np.array(bias_offsets)) / temperature
        probs = softmax(calibrated)

        # Politicalness gate
        if pol_threshold is not None and item["political_confidence"] < pol_threshold:
            pred = "Center"
        else:
            pred_idx = np.argmax(probs)
            pred = LABELS[pred_idx]

        predictions.append(pred)
        ground_truths.append(gt)

    return predictions, ground_truths


def grid_search(cache, label_key="source_label", fine=False):
    """
    Exhaustive grid search over calibration parameters.
    """
    print(f"\n{'═' * 70}")
    print(f"  GRID SEARCH (labels={label_key})")
    print(f"{'═' * 70}")

    # ── Define search ranges ──
    if fine:
        # Fine-grained (use after coarse to zoom in)
        bias_L_range = np.arange(-4.0, 1.0, 0.1)
        bias_C_range = np.arange(-1.0, 6.0, 0.1)
        bias_R_range = [0.0]  # Fix Right as reference
        temp_range = np.arange(0.5, 3.0, 0.1)
        pol_range = [None]  # Skip pol threshold in fine search
    else:
        # Coarse search
        bias_L_range = np.arange(-5.0, 2.0, 0.5)
        bias_C_range = np.arange(-2.0, 8.0, 0.5)
        bias_R_range = [0.0]  # Fix Right as reference
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
    best_balanced = {"score": 0, "params": None}  # 0.5*acc + 0.5*f1

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

        # Track best results
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

        # Store all results above a threshold for analysis
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

    elapsed = time.time() - t_start
    print(f"\n  Search completed in {elapsed:.1f}s "
          f"({checked/elapsed:.0f} combos/sec)")

    # ── Report best results ──
    print(f"\n{'─' * 70}")
    print(f"  BEST BY ACCURACY")
    print(f"{'─' * 70}")
    _print_result(best_acc)

    print(f"\n{'─' * 70}")
    print(f"  BEST BY MACRO-F1")
    print(f"{'─' * 70}")
    _print_result(best_f1)

    print(f"\n{'─' * 70}")
    print(f"  BEST BALANCED (0.5*acc + 0.5*f1)")
    print(f"{'─' * 70}")
    _print_result(best_balanced)

    # ── Show top 10 configurations ──
    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    print(f"\n{'─' * 70}")
    print(f"  TOP 10 BY MACRO-F1")
    print(f"{'─' * 70}")
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
    print(f"\n{'─' * 70}")
    print(f"  TOP 10 BY ACCURACY")
    print(f"{'─' * 70}")
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
    """Print a single calibration result."""
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


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Detailed analysis of a specific configuration
# ═══════════════════════════════════════════════════════════════════════

def analyze_configuration(cache, bias_offsets, temperature, pol_threshold=None,
                          label_key="source_label"):
    """
    Detailed per-article analysis for a specific calibration config.
    """
    preds, gts = predict_with_calibration(
        cache, bias_offsets, temperature, pol_threshold, label_key
    )
    acc, macro_f1, per_class = compute_metrics_fast(preds, gts)

    print(f"\n{'═' * 70}")
    print(f"  DETAILED ANALYSIS")
    print(f"  bias=[{bias_offsets[0]:.2f}, {bias_offsets[1]:.2f}, {bias_offsets[2]:.2f}]  "
          f"T={temperature:.2f}  pol={pol_threshold}")
    print(f"{'═' * 70}")
    print(f"  Accuracy: {acc:.1%}  Macro-F1: {macro_f1:.3f}")

    for label in LABELS:
        pc = per_class[label]
        print(f"  {label:<8} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")

    # Confusion matrix
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for p, g in zip(preds, gts):
        confusion[g][p] += 1

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>12} {'Left':>8} {'Center':>8} {'Right':>8}")
    for true in LABELS:
        row = "".join(f"{confusion[true][p]:>8}" for p in LABELS)
        print(f"  {true:>12}{row}")

    # Prediction distribution
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


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate bias detection thresholds on BASIL"
    )
    parser.add_argument("--basil-dir", default="BASIL")
    parser.add_argument("--bias-model", default="models/bias_detector")
    parser.add_argument("--pol-model", default="models/politicalness_filter")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--cache-file", default=CACHE_FILE)
    parser.add_argument("--cache-only", action="store_true",
                        help="Only run inference and cache, skip search")
    parser.add_argument("--search-only", action="store_true",
                        help="Only run grid search on existing cache")
    parser.add_argument("--fine", action="store_true",
                        help="Fine-grained search (0.1 steps)")
    parser.add_argument("--label-key", default="source_label",
                        choices=["source_label", "annotation_label"],
                        help="Which ground truth to optimize against")
    parser.add_argument("--analyze", nargs=5, type=float, metavar=(
                        "BIAS_L", "BIAS_C", "BIAS_R", "TEMP", "POL"),
                        help="Analyze a specific config: bias_L bias_C bias_R temp pol_thresh (-1 for None)")
    parser.add_argument("--output-json", default="calibration_results.json")
    args = parser.parse_args()

    CACHE_FILE_PATH = args.cache_file

    # ── Step 1: Cache ──
    if not args.search_only:
        print("=" * 70)
        print("  Step 1: Caching model outputs on BASIL...")
        print("=" * 70)
        cache = cache_model_outputs(
            args.basil_dir, args.bias_model, args.pol_model,
            device=args.device, use_fp16=not args.no_fp16,
        )
        if args.cache_only:
            return

    # ── Load cache ──
    if not os.path.isfile(CACHE_FILE_PATH):
        print(f"ERROR: Cache file not found: {CACHE_FILE_PATH}")
        print(f"  Run with --cache-only first to generate it.")
        sys.exit(1)

    with open(CACHE_FILE_PATH) as f:
        cache = json.load(f)
    print(f"\n  Loaded {len(cache)} cached articles from {CACHE_FILE_PATH}")

    # Sanity check: verify logits have actual signal
    sample_logits = cache[0]["bias_logits"]
    all_zero = all(abs(l) < 1e-6 for l in sample_logits)
    if all_zero:
        print(f"\n  ⚠ WARNING: Sample logits are all zero: {sample_logits}")
        print(f"  This likely means the label mapping failed during caching.")
        print(f"  Model label order was: {cache[0].get('label_order', 'unknown')}")
        print(f"  Re-run with --cache-only to regenerate the cache.")
        sys.exit(1)
    else:
        print(f"  Sample logits: {sample_logits} ✓")
        print(f"  Logit ranges: L=[{min(c['bias_logits'][0] for c in cache):.2f}, "
              f"{max(c['bias_logits'][0] for c in cache):.2f}]  "
              f"C=[{min(c['bias_logits'][1] for c in cache):.2f}, "
              f"{max(c['bias_logits'][1] for c in cache):.2f}]  "
              f"R=[{min(c['bias_logits'][2] for c in cache):.2f}, "
              f"{max(c['bias_logits'][2] for c in cache):.2f}]")

    # Quick sanity check: verify original results match
    orig_preds = [c["original_prediction"] for c in cache]
    orig_gts = [c["source_label"] for c in cache]
    orig_acc = sum(p == g for p, g in zip(orig_preds, orig_gts)) / len(orig_preds)
    print(f"  Original pipeline accuracy (sanity check): {orig_acc:.1%}")

    # ── Analyze specific config ──
    if args.analyze:
        bias_L, bias_C, bias_R, temp, pol = args.analyze
        pol_thresh = None if pol < 0 else pol
        analyze_configuration(
            cache, [bias_L, bias_C, bias_R], temp, pol_thresh, args.label_key
        )
        return

    # ── Step 2: Grid search ──
    print(f"\n{'=' * 70}")
    print(f"  Step 2: Grid search over calibration parameters...")
    print(f"{'=' * 70}")

    results = grid_search(cache, label_key=args.label_key, fine=args.fine)

    # Also run against annotation labels for comparison
    if args.label_key == "source_label":
        print(f"\n\n{'=' * 70}")
        print(f"  Cross-check: applying best source-label params to annotation labels")
        print(f"{'=' * 70}")

        for name, best in [("best_accuracy", results["best_accuracy"]),
                           ("best_macro_f1", results["best_macro_f1"]),
                           ("best_balanced", results["best_balanced"])]:
            p = best["params"]
            offsets = [p["bias_L"], p["bias_C"], p["bias_R"]]
            print(f"\n  --- {name} params → annotation labels ---")
            analyze_configuration(
                cache, offsets, p["temperature"],
                p["pol_threshold"], "annotation_label"
            )

    # ── Save results ──
    if args.output_json:
        output = {
            "label_key": args.label_key,
            "fine": args.fine,
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
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {args.output_json}")


if __name__ == "__main__":
    main()