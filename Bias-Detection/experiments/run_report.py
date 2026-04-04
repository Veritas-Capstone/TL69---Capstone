"""
Evaluation report generator - runs the pipeline on BASIL and saves results.

Tests:
  1. Demo on hand-picked examples
  2. BASIL evaluation (source-based labels)
  3. BASIL evaluation (annotation-based labels)
  4. Within-event analysis (can the model distinguish bias across outlets?)
"""

import sys
import os
import json
import time
import numpy as np
from io import StringIO
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basil_loader import load_basil
from pipeline import BiasDetector, compute_metrics, LABELS

# config
BASIL_DIR = "./datasets/other_data/BASIL"
BIAS_MODEL = "./models/demo_models/bias_detector"
POL_MODEL = "./models/demo_models/politicalness_filter"
OUTPUT_FILE = "results/report.txt"
OUTPUT_JSON = "results/metrics.json"


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
        print(f"\nReport saved to {self.filepath}")


# test 1: demo examples
def run_demo(detector, report):
    examples = [
        ("Left", "Progressive Democrats championed Medicare for All and free "
         "college tuition, arguing that universal healthcare is a fundamental right."),
        ("Center", "The Congressional Budget Office released its annual economic outlook "
         "on Wednesday, projecting GDP growth of 2.1% for the coming fiscal year."),
        ("Right", "Conservative lawmakers defended free market principles and called for "
         "reduced government regulation, arguing that individual liberty is paramount."),
        ("Non-political", "The weather forecast predicts rain this weekend with "
         "temperatures dropping into the low 50s."),
    ]

    report.print("\n  Demo Examples")
    correct = 0
    total = 0

    for expected, text in examples:
        result = detector.analyze(text, explain_top_k=1)
        pred = result["prediction"] or "(non-political)"

        if expected == "Non-political":
            match = not result["is_political"]
        else:
            match = pred == expected
            total += 1
        if match:
            correct += 1

        icon = "PASS" if match else "FAIL"
        report.print(f"\n  [{expected}] {icon}")
        report.print(f"    Text: {text[:80]}...")
        report.print(f"    Politicalness: {'yes' if result['is_political'] else 'no'} ({result['political_confidence']:.2f})")
        if result["probs"]:
            report.print(f"    Prediction: {pred}  L={result['probs']['Left']:.3f}  C={result['probs']['Center']:.3f}  R={result['probs']['Right']:.3f}")

    total += 1  # count non-political test
    report.print(f"\n  Demo: {correct}/{total} correct")
    return correct, total


# test 2/3: BASIL evaluation
def run_basil_eval(detector, label_source, report):
    articles = load_basil(BASIL_DIR, label_source=label_source)
    if not articles:
        report.print(f"  No articles loaded for label_source={label_source}")
        return None

    report.print(f"\n  BASIL Evaluation (labels={label_source}, n={len(articles)})")

    predictions, ground_truths = [], []
    non_political = 0

    for article in articles:
        result = detector.analyze(body=article["body"], headline=article["headline"])
        pred = result["prediction"] or "Center"
        if not result["is_political"]:
            non_political += 1
        predictions.append(pred)
        ground_truths.append(article["true_label"])

    metrics = compute_metrics(predictions, ground_truths)

    report.print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    report.print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    report.print(f"  Non-political: {non_political}/{len(articles)}")

    for label in LABELS:
        m = metrics["per_class"][label]
        report.print(f"  {label:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    # confusion matrix
    report.print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    report.print(f"  {'':>12} {'Left':>8} {'Center':>8} {'Right':>8}")
    for tl in LABELS:
        row = "".join(f" {metrics['confusion'][tl][pl]:>8}" for pl in LABELS)
        report.print(f"  {tl:>12}{row}")

    return metrics


# test 4: within-event analysis
def run_within_event_analysis(detector, report):
    articles = load_basil(BASIL_DIR, label_source="source")

    report.print(f"\n  Within-Event Analysis")

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

    report.print(f"  Triplets: {total}")
    if total > 0:
        report.print(f"  All 3 correct:     {triplet_correct} ({triplet_correct/total:.1%})")
        report.print(f"  Partially correct: {triplet_partial} ({triplet_partial/total:.1%})")
        report.print(f"  All 3 wrong:       {triplet_wrong} ({triplet_wrong/total:.1%})")
        report.print(f"  Sees difference:   {separation}/{total} ({separation/total:.1%})")


if __name__ == "__main__":
    report = ReportWriter(OUTPUT_FILE)

    report.print(f"Bias Detection Pipeline — Evaluation Report")
    report.print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    print("Loading models...")
    detector = BiasDetector(bias_path=BIAS_MODEL, pol_path=POL_MODEL)

    all_results = {"timestamp": datetime.now().isoformat(), "model": BIAS_MODEL}

    # test 1: demo
    demo_correct, demo_total = run_demo(detector, report)
    all_results["demo"] = {"correct": demo_correct, "total": demo_total}

    # test 2: BASIL source labels
    m_source = run_basil_eval(detector, "source", report)
    if m_source:
        all_results["basil_source"] = {
            "accuracy": m_source["accuracy"], "macro_f1": m_source["macro_f1"],
        }

    # test 3: BASIL annotation labels
    m_ann = run_basil_eval(detector, "annotation", report)
    if m_ann:
        all_results["basil_annotation"] = {
            "accuracy": m_ann["accuracy"], "macro_f1": m_ann["macro_f1"],
        }

    # test 4: within-event
    run_within_event_analysis(detector, report)

    # summary
    if m_source and m_ann:
        report.print(f"\n  Summary:")
        report.print(f"  {'BASIL (source)':<30} Acc={m_source['accuracy']:.1%}  F1={m_source['macro_f1']:.3f}")
        report.print(f"  {'BASIL (annotation)':<30} Acc={m_ann['accuracy']:.1%}  F1={m_ann['macro_f1']:.3f}")

    report.save()

    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON metrics saved to {OUTPUT_JSON}")