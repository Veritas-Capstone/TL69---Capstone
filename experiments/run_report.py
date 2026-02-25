#!/usr/bin/env python3
"""
Bias Detection — Evaluation Report Generator
==============================================
Runs the core evaluation suite and saves results to a report file.

Tests:
  1. Demo on hand-picked examples (sanity check)
  2. BASIL dataset evaluation (source-based labels)
  3. BASIL dataset evaluation (annotation-based labels)
  4. Within-event analysis (can the model distinguish bias across outlets?)
  5. Calibration summary (if cached results exist)

Usage:
  python run_report.py --basil-dir BASIL
  python run_report.py --basil-dir BASIL --output results/report.txt
  python run_report.py --basil-dir BASIL --skip-events
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from io import StringIO
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basil_loader import load_basil
from pipeline import BiasDetector, compute_metrics, LABELS


# ═══════════════════════════════════════════════════════════════════════
# Report Writer — tees output to console + file
# ═══════════════════════════════════════════════════════════════════════

class ReportWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.buffer = StringIO()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, file=self.buffer, **kwargs)

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(self.buffer.getvalue())
        print(f"\n  Report saved to {self.filepath}")


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Demo Examples
# ═══════════════════════════════════════════════════════════════════════

def run_demo(detector, report):
    examples = [
        ("Left",
         "Progressive Democrats championed Medicare for All and free "
         "college tuition, arguing that universal healthcare is a fundamental "
         "right that the richest nation on Earth can and should guarantee."),
        ("Center",
         "The Congressional Budget Office released its annual economic outlook "
         "on Wednesday, projecting GDP growth of 2.1% for the coming fiscal year "
         "and noting that the federal deficit is expected to widen modestly."),
        ("Right",
         "Conservative lawmakers defended free market principles and "
         "called for reduced government regulation, arguing that individual "
         "liberty and personal responsibility are the cornerstones of "
         "American prosperity."),
        ("Non-political",
         "The weather forecast predicts rain this weekend with temperatures "
         "dropping into the low 50s across the eastern seaboard."),
    ]

    report.print(f"\n{'=' * 70}")
    report.print(f"  TEST 1: DEMO EXAMPLES")
    report.print(f"{'=' * 70}")

    correct = 0
    total_political = 0

    for expected, text in examples:
        result = detector.analyze(text, explain_top_k=1)
        pred = result["prediction"] or "(non-political)"
        pol = "political" if result["is_political"] else "non-political"

        if expected == "Non-political":
            match = not result["is_political"]
        else:
            match = pred == expected
            total_political += 1
        if match:
            correct += 1

        icon = "PASS" if match else "FAIL"
        report.print(f"\n  [{expected}] {icon}")
        report.print(f"    Text: {text[:80]}...")
        report.print(f"    Politicalness: {pol} (conf={result['political_confidence']:.2f})")
        if result["probs"]:
            report.print(f"    Prediction: {pred}  |  "
                         f"L={result['probs']['Left']:.3f}  "
                         f"C={result['probs']['Center']:.3f}  "
                         f"R={result['probs']['Right']:.3f}")
        if result["sentences"]:
            for sr in result["sentences"]:
                if sr.get("top_tokens"):
                    tstr = ", ".join(
                        f"{t['clean_token']}({t['score']:.2f})"
                        for t in sr["top_tokens"][:5]
                    )
                    report.print(f"    Key tokens: {tstr}")

    report.print(f"\n  Demo: {correct}/{total_political + 1} correct")
    return correct, total_political + 1


# ═══════════════════════════════════════════════════════════════════════
# Test 2 & 3: BASIL Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_basil_eval(detector, basil_dir, label_source, report):
    articles = load_basil(basil_dir, label_source=label_source)
    if not articles:
        report.print(f"  ERROR: No articles loaded for label_source={label_source}")
        return None

    report.print(f"\n{'=' * 70}")
    report.print(f"  BASIL EVALUATION (labels={label_source}, n={len(articles)})")
    report.print(f"{'=' * 70}")

    predictions = []
    ground_truths = []
    non_political = 0
    total_time = 0

    for article in articles:
        result = detector.analyze(body=article["body"], headline=article["headline"])
        total_time += result["elapsed"]

        pred = result["prediction"] or "Center"
        if not result["is_political"]:
            non_political += 1

        predictions.append(pred)
        ground_truths.append(article["true_label"])

    metrics = compute_metrics(predictions, ground_truths)

    report.print(f"\n  Accuracy:      {metrics['accuracy']:.1%} "
                 f"({metrics['correct']}/{metrics['total']})")
    report.print(f"  Macro F1:      {metrics['macro_f1']:.3f}")
    report.print(f"  Non-political: {non_political}/{len(articles)}")
    report.print(f"  Avg time:      {total_time/len(articles):.2f}s/article")

    report.print(f"\n  {'Label':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    report.print(f"  {'-' * 42}")
    for label in LABELS:
        m = metrics["per_class"][label]
        report.print(f"  {label:<10} {m['precision']:>8.3f} {m['recall']:>8.3f} "
                     f"{m['f1']:>8.3f} {m['support']:>8}")

    report.print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    report.print(f"  {'':>12} {'Left':>8} {'Center':>8} {'Right':>8}")
    for tl in LABELS:
        row = "".join(f" {metrics['confusion'][tl][pl]:>8}" for pl in LABELS)
        report.print(f"  {tl:>12}{row}")

    # Top confusions
    errors = [(g, p) for p, g in zip(predictions, ground_truths) if p != g]
    if errors:
        confusion_pairs = Counter(errors)
        report.print(f"\n  Top confusions:")
        for (true, pred), count in confusion_pairs.most_common(5):
            report.print(f"    {true:>8} -> {pred:<8}  ({count}x)")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Within-Event Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_within_event_analysis(detector, basil_dir, report):
    """For each event (same story, 3 outlets): can the model tell them apart?"""
    articles = load_basil(basil_dir, label_source="source")

    report.print(f"\n{'=' * 70}")
    report.print(f"  WITHIN-EVENT ANALYSIS")
    report.print(f"{'=' * 70}")

    events = defaultdict(list)
    for a in articles:
        events[a["event_id"]].append(a)

    triplet_correct = 0
    triplet_partial = 0
    triplet_wrong = 0
    separation = 0
    total = 0

    for eid, event_articles in events.items():
        if len(event_articles) != 3:
            continue
        total += 1

        results = []
        for a in event_articles:
            r = detector.analyze(body=a["body"], headline=a["headline"])
            results.append((r["prediction"] or "Center", a["true_label"]))

        correct_count = sum(1 for p, t in results if p == t)
        if correct_count == 3:
            triplet_correct += 1
        elif correct_count > 0:
            triplet_partial += 1
        else:
            triplet_wrong += 1

        if len(set(p for p, _ in results)) > 1:
            separation += 1

    report.print(f"  Triplets (same story, 3 outlets): {total}")
    if total > 0:
        report.print(f"  All 3 correct:     {triplet_correct:4d} ({triplet_correct/total:.1%})")
        report.print(f"  Partially correct: {triplet_partial:4d} ({triplet_partial/total:.1%})")
        report.print(f"  All 3 wrong:       {triplet_wrong:4d} ({triplet_wrong/total:.1%})")
        report.print(f"  Model sees difference (>1 unique pred): "
                     f"{separation}/{total} ({separation/total:.1%})")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Calibration Summary
# ═══════════════════════════════════════════════════════════════════════

def run_calibration_summary(report, results_file="calibration_results.json"):
    report.print(f"\n{'=' * 70}")
    report.print(f"  CALIBRATION RESULTS")
    report.print(f"{'=' * 70}")

    if not os.path.isfile(results_file):
        report.print(f"  No calibration results found.")
        report.print(f"  Run: python threshold.py --basil-dir BASIL")
        return

    with open(results_file) as f:
        cal = json.load(f)

    report.print(f"  Original (uncalibrated) accuracy: "
                 f"{cal.get('original_accuracy', 0):.1%}")

    for name in ["best_accuracy", "best_macro_f1", "best_balanced"]:
        if name in cal:
            r = cal[name]
            report.print(f"\n  {name}:")
            report.print(f"    Accuracy: {r.get('accuracy', 0):.1%}")
            report.print(f"    Macro F1: {r.get('macro_f1', 0):.3f}")
            if "params" in r:
                p = r["params"]
                report.print(f"    Params: bias_L={p.get('bias_L', 0):.2f}  "
                             f"bias_C={p.get('bias_C', 0):.2f}  "
                             f"T={p.get('temperature', 1):.2f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation suite and generate report"
    )
    parser.add_argument("--basil-dir", default="BASIL")
    parser.add_argument("--bias-model", default="models/bias_detector")
    parser.add_argument("--pol-model", default="models/politicalness_filter")
    parser.add_argument("--output", default="results/report.txt")
    parser.add_argument("--output-json", default="results/metrics.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-events", action="store_true",
                        help="Skip within-event analysis (saves time)")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    report = ReportWriter(args.output)

    report.print("=" * 70)
    report.print("  BIAS DETECTION PIPELINE — EVALUATION REPORT")
    report.print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.print("=" * 70)

    # Load models
    report.print(f"\n  Loading models...")
    detector = BiasDetector(
        bias_path=args.bias_model,
        pol_path=args.pol_model,
        device=args.device,
    )

    all_results = {"timestamp": datetime.now().isoformat(), "model": args.bias_model}

    # ── Test 1: Demo ──
    demo_correct, demo_total = run_demo(detector, report)
    all_results["demo"] = {"correct": demo_correct, "total": demo_total}

    # ── Test 2: BASIL source labels ──
    m_source = run_basil_eval(detector, args.basil_dir, "source", report)
    if m_source:
        all_results["basil_source"] = {
            "accuracy": m_source["accuracy"],
            "macro_f1": m_source["macro_f1"],
            "per_class": {
                l: {k: m_source["per_class"][l][k]
                    for k in ("precision", "recall", "f1")}
                for l in LABELS
            },
        }

    # ── Test 3: BASIL annotation labels ──
    m_ann = run_basil_eval(detector, args.basil_dir, "annotation", report)
    if m_ann:
        all_results["basil_annotation"] = {
            "accuracy": m_ann["accuracy"],
            "macro_f1": m_ann["macro_f1"],
        }

    # ── Test 4: Within-event ──
    if not args.skip_events:
        run_within_event_analysis(detector, args.basil_dir, report)

    # ── Test 5: Calibration ──
    if not args.skip_calibration:
        run_calibration_summary(report)

    # ── Summary table ──
    report.print(f"\n{'=' * 70}")
    report.print(f"  SUMMARY")
    report.print(f"{'=' * 70}")

    if m_source and m_ann:
        report.print(f"\n  {'Benchmark':<30} {'Accuracy':>10} {'Macro F1':>10}")
        report.print(f"  {'-' * 50}")
        report.print(f"  {'BASIL (source labels)':<30} "
                     f"{m_source['accuracy']:>10.1%} "
                     f"{m_source['macro_f1']:>10.3f}")
        report.print(f"  {'BASIL (annotation labels)':<30} "
                     f"{m_ann['accuracy']:>10.1%} "
                     f"{m_ann['macro_f1']:>10.3f}")

    report.print(f"\n  Note: BASIL is an out-of-distribution benchmark.")
    report.print(f"  The model paper reports OOD accuracy of 38-67%.")
    report.print(f"  Fine-tuning with LoRA (finetune_basil.py) is the")
    report.print(f"  recommended next step to improve these numbers.")

    # Save
    report.save()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  JSON metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
