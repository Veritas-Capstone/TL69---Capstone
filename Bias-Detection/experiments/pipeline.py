"""
Bias Detection Pipeline — Two-Pass Architecture
=================================================
Pass 1: Document-level prediction (full article → single label)
         This is the REAL prediction. The model was trained on full
         articles, so feeding it full articles gives the best accuracy.

Pass 2: Sentence-level Gradient×Input explainability (optional)
         Only used to highlight which words/sentences drive the prediction.
         ~30x faster than Integrated Gradients (1 fwd+bwd vs 30 fwd passes).

Models:
  - Politicalness filter: Political DEBATE (NLI-based DeBERTa)
    Paper: arxiv.org/abs/2409.02078
  - Bias classifier: matous-volf/political-leaning-deberta-large (3-class)
    Paper: arxiv.org/abs/2507.13913

Usage:
  python pipeline.py --allsides data1.json data2.json
  python pipeline.py --allsides data1.json --explain-top-k 3
  python pipeline.py --demo
"""

import json
import time
import argparse
import torch
import numpy as np
import warnings
from pathlib import Path
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

LABELS = ["Left", "Center", "Right"]

ALLSIDES_MAP = {
    "Left": "Left",
    "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right",
    "Right": "Right",
}

# Filtered out of explainability token lists (not from predictions)
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "can", "that", "this", "these", "those", "it", "its", "not",
    "no", "so", "if", "then", "than", "too", "very", "just",
    "about", "up", "out", "also", "as", "from", "into", "he",
    "she", "they", "we", "you", "his", "her", "their", "our",
    "my", "me", "him", "them", "us", "who", "which", "what",
    "when", "where", "how", "all", "each", "every", "both",
    "more", "most", "other", "some", "such", "only", "own",
    "same", "any", "there", "here", "said", "says", "new",
})

PUNCTUATION = frozenset({
    ".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]",
    "-", "–", "—", "▁.", "▁,", "▁!", "▁?", "▁;", "▁:", "▁-",
})

# Multiple hypotheses for politicalness — take the max.
# The single "about politics" hypothesis misses policy, economics,
# regulation, criminal justice, etc.  These catch a wider net.
POLITICALNESS_HYPOTHESES = [
    "This text is about politics or government policy.",
    "This text discusses a politically controversial or partisan topic.",
    "This text is about legislation, regulation, or political debate.",
    "This text discusses government officials or political parties.",
]


# ═══════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════

class BiasDetector:
    """
    Wraps both models with GPU/FP16 support and provides
    document-level prediction + sentence-level explainability.
    """

    def __init__(self, bias_path="models/bias_detector",
                 pol_path="models/politicalness_filter",
                 device=None, use_fp16=True,
                 pol_threshold=0.5, center_bias=0.0):

        # ── Device selection ──
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pol_threshold = pol_threshold   # ← store them
        self.center_bias = center_bias

        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        dtype_str = "FP16" if self.use_fp16 else "FP32"
        print(f"Device: {self.device}  |  Precision: {dtype_str}")

        # ── Load bias model ──
        print(f"Loading bias model from {bias_path} ...")
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(bias_path)
        self.bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
        self.bias_model.eval().to(self.device)
        if self.use_fp16:
            self.bias_model.half()

        # ── Load politicalness model ──
        print(f"Loading politicalness model from {pol_path} ...")
        self.pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
        self.pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
        self.pol_model.eval().to(self.device)
        if self.use_fp16:
            self.pol_model.half()

        # Cache the entailment index
        label2id = self.pol_model.config.label2id
        self.entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))

        # Find embedding layer (for Gradient×Input)
        self._embedding_layer = self._find_embedding_layer(self.bias_model)
        if self._embedding_layer is None:
            print("⚠  Could not locate embedding layer — explainability disabled")

        print("✓ Models loaded\n")

    # ─── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _find_embedding_layer(model):
        for attr in ("deberta", "bert", "roberta", "distilbert"):
            backbone = getattr(model, attr, None)
            if backbone is not None:
                return backbone.embeddings.word_embeddings
        # Generic fallback
        for name, module in model.named_modules():
            if "word_embedding" in name.lower() and hasattr(module, "weight"):
                return module
        return None

    # ═══════════════════════════════════════════════════════════════════
    # PASS 1 — Document-Level Prediction
    # ═══════════════════════════════════════════════════════════════════

    def check_politicalness(self, text, threshold=0.5):
        """
        NLI-based politicalness check using multiple hypotheses.

        Runs the text against each hypothesis in POLITICALNESS_HYPOTHESES
        and takes the MAX entailment score.  This catches articles about
        policy, regulation, economics, criminal justice, etc. that the
        single "about politics" hypothesis misses.

        Returns (is_political: bool, confidence: float).
        """
        best_conf = 0.0

        # Batch all hypotheses in one forward pass
        pairs = [(text, h) for h in POLITICALNESS_HYPOTHESES]
        inputs = self.pol_tokenizer(
            pairs,
            return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.pol_model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1)  # (num_hypotheses, num_labels)
        best_conf = float(probs[:, self.entail_idx].max())

        return best_conf > threshold, best_conf

    def predict_bias_document(self, text):
        """
        Document-level bias prediction.
        Feeds the full article (truncated to 512 tokens) into the model.
        This is the PRIMARY prediction — not sentence-level.
        """
        inputs = self.bias_tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.bias_model(**inputs).logits
        
        if self.center_bias != 0.0:
            logits[0, 1] += self.center_bias  # index 1 = Center

        probs = torch.softmax(logits.float(), dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

        return {
            "predicted_label": LABELS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {l: float(probs[i]) for i, l in enumerate(LABELS)},
        }

    # ═══════════════════════════════════════════════════════════════════
    # PASS 2 — Gradient × Input Explainability
    # ═══════════════════════════════════════════════════════════════════

    def gradient_x_input(self, text, target_class=None):
        """
        Compute token-level attributions via Gradient × Input.

        Cost: 1 forward + 1 backward pass (vs. IG's ~30 forward passes).
        Quality: Comparable to IG for transformers in practice.

        Returns list of {token, clean_token, score, position} sorted desc.
        """
        if self._embedding_layer is None:
            return []

        # We need gradients, so temporarily enable them
        # and run in FP32 for numerical stability
        was_fp16 = next(self.bias_model.parameters()).dtype == torch.float16
        if was_fp16:
            self.bias_model.float()

        inputs = self.bias_tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tokens = self.bias_tokenizer.convert_ids_to_tokens(input_ids[0])

        # Get embeddings with grad tracking
        embeddings = self._embedding_layer(input_ids)
        embeddings = embeddings.detach().requires_grad_(True)

        # Forward pass with embeddings
        outputs = self.bias_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        # Backward pass
        logits[0, target_class].backward()

        # Attribution = gradient ⊙ input, summed over embedding dim
        attr = (embeddings.grad * embeddings).sum(dim=-1).abs().squeeze(0)

        # Normalize to [0, 1]
        if attr.max() > 0:
            attr = attr / attr.max()
        scores = attr.detach().cpu().numpy()

        # Restore FP16 if we switched
        if was_fp16:
            self.bias_model.half()

        return self._build_token_list(tokens, scores)

    def explain_sentences(self, text, doc_prediction, top_k=3):
        """
        Run Gradient×Input on the top-k most 'interesting' sentences.

        Strategy:
          1. Split text into sentences (lightweight, no spaCy)
          2. Batch-predict bias on all sentences (fast)
          3. Pick top-k sentences furthest from Center
          4. Run Gradient×Input only on those
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Batch bias prediction on sentences (fast, no grad)
        sent_inputs = self.bias_tokenizer(
            sentences, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            sent_logits = self.bias_model(**sent_inputs).logits
        sent_probs = torch.softmax(sent_logits.float(), dim=-1)

        # Score each sentence by "bias strength" (1 - center_prob)
        sentence_data = []
        for i, sent in enumerate(sentences):
            probs = sent_probs[i]
            pred_idx = torch.argmax(probs).item()
            center_prob = float(probs[1])  # Center is index 1
            bias_strength = 1.0 - center_prob

            sentence_data.append({
                "text": sent,
                "predicted_label": LABELS[pred_idx],
                "confidence": float(probs[pred_idx]),
                "all_probs": {l: float(probs[j]) for j, l in enumerate(LABELS)},
                "bias_strength": bias_strength,
                "top_tokens": [],
            })

        # Sort by bias strength, pick top-k for full explainability
        ranked = sorted(
            enumerate(sentence_data),
            key=lambda x: x[1]["bias_strength"],
            reverse=True,
        )

        for idx, _ in ranked[:top_k]:
            sd = sentence_data[idx]
            sd["top_tokens"] = self.gradient_x_input(sd["text"])[:5]
            sd["explained"] = True

        return sentence_data

    # ═══════════════════════════════════════════════════════════════════
    # Combined Pipeline
    # ═══════════════════════════════════════════════════════════════════

    def analyze(self, body, headline=None, explain_top_k=0):
        """
        Full two-pass analysis pipeline.

        Args:
            body: Article body text
            headline: Optional headline (prepended for extra context)
            explain_top_k: Number of sentences to explain (0 = skip Pass 2)

        Returns:
            dict with prediction, probabilities, and optional explanations
        """
        start = time.time()

        # Combine headline + body for maximum context
        full_text = body
        if headline:
            full_text = headline + ". " + body

        # ── Pass 1A: Politicalness check ──
        is_political, pol_conf = self.check_politicalness(full_text , threshold=self.pol_threshold)

        if not is_political:
            return {
                "is_political": False,
                "political_confidence": pol_conf,
                "prediction": None,
                "probs": None,
                "sentences": [],
                "elapsed": time.time() - start,
            }

        # ── Pass 1B: Document-level bias prediction ──
        doc_result = self.predict_bias_document(full_text)

        # ── Pass 2 (optional): Sentence-level explainability ──
        sentences = []
        if explain_top_k > 0:
            sentences = self.explain_sentences(body, doc_result, top_k=explain_top_k)

        return {
            "is_political": True,
            "political_confidence": pol_conf,
            "prediction": doc_result["predicted_label"],
            "confidence": doc_result["confidence"],
            "probs": doc_result["all_probs"],
            "sentences": sentences,
            "elapsed": time.time() - start,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _split_sentences(text):
        """
        Lightweight sentence splitter — no spaCy dependency.
        Splits on .!? followed by space+uppercase or newline.
        Merges short fragments (< 8 words) into the previous sentence.
        """
        import re
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        # Also split on double newlines
        expanded = []
        for chunk in raw:
            parts = chunk.split("\n\n")
            expanded.extend(p.strip() for p in parts if p.strip())

        # Merge short fragments
        merged = []
        for s in expanded:
            if merged and len(s.split()) < 8:
                merged[-1] += " " + s
            else:
                merged.append(s)

        # Filter out very short sentences
        return [s for s in merged if len(s.split()) >= 5]

    @staticmethod
    def _build_token_list(tokens, scores):
        """Build filtered, sorted list of (token, score) pairs."""
        result = []
        for i, (token, score) in enumerate(zip(tokens, scores)):
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
                continue
            clean = token.replace("##", "").replace("▁", "").strip()
            if not clean or clean in PUNCTUATION:
                continue
            if all(c in '.,!?;:\'"()[]<>-–—/' for c in clean):
                continue
            if clean.lower() in STOP_WORDS:
                continue
            result.append({
                "token": token.replace("##", ""),
                "clean_token": clean,
                "score": float(score),
                "position": i,
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return result


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_allsides_data(filepaths):
    """
    Load and flatten AllSides JSON files into individual article entries.
    Handles multiple files, deduplicates by original_url.
    """
    articles = []
    seen_urls = set()

    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = json.load(f)

        for story in data:
            main_headline = story.get("main_headline", "")
            for side in story.get("sides", []):
                bias_detail = side.get("bias_detail")
                body = (side.get("body") or "").strip()
                url = side.get("original_url", "")

                # Skip if no body, no label, or duplicate
                if not body or not bias_detail or bias_detail not in ALLSIDES_MAP:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                articles.append({
                    "story": main_headline,
                    "source": side.get("source", "Unknown"),
                    "headline": side.get("headline", ""),
                    "body": body,
                    "bias_detail": bias_detail,
                    "true_label": ALLSIDES_MAP[bias_detail],
                    "url": url,
                })

    return articles


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(predictions, ground_truths):
    """Accuracy, per-class P/R/F1, macro F1, confusion matrix."""
    n = len(predictions)
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    accuracy = correct / n if n > 0 else 0

    class_metrics = {}
    for label in LABELS:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        class_metrics[label] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": sum(1 for g in ground_truths if g == label),
        }

    macro_f1 = np.mean([m["f1"] for m in class_metrics.values()])

    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for p, g in zip(predictions, ground_truths):
        confusion[g][p] += 1

    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "correct": correct, "total": n,
        "per_class": class_metrics, "confusion": confusion,
    }


def print_metrics(metrics, title="RESULTS"):
    w = 70
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}  ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro F1:  {metrics['macro_f1']:.3f}")

    print(f"\n  {'Label':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'─' * 42}")
    for label in LABELS:
        m = metrics["per_class"][label]
        print(f"  {label:<10} {m['precision']:>8.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>8}")

    print(f"\n  Confusion Matrix (rows = true, cols = predicted):")
    print(f"  {'':>10}", end="")
    for l in LABELS:
        print(f" {l:>8}", end="")
    print()
    for tl in LABELS:
        print(f"  {tl:>10}", end="")
        for pl in LABELS:
            print(f" {metrics['confusion'][tl][pl]:>8}", end="")
        print()
    print(f"{'═' * w}")


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_evaluation(articles, detector, explain_top_k=0):
    """Run the two-pass pipeline on all articles and report metrics."""
    predictions = []
    ground_truths = []
    errors = []
    total_time = 0

    for i, article in enumerate(articles):
        result = detector.analyze(
            body=article["body"],
            headline=article["headline"],
            explain_top_k=explain_top_k,
        )

        total_time += result["elapsed"]
        true = article["true_label"]

        # Fallback if article deemed non-political
        pred = result["prediction"] or "Center"

        predictions.append(pred)
        ground_truths.append(true)

        match = "✓" if pred == true else "✗"

        # ── Per-article output ──
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(articles)}] {match}  {article['source']}")
        print(f"  Headline: {article['headline'][:65]}")
        print(f"  True: {true:>8}  →  Predicted: {pred:>8}  "
              f"(pol={result['political_confidence']:.2f}, {result['elapsed']:.2f}s)")

        if result["probs"]:
            probs_str = "  ".join(f"{l}: {p:.3f}" for l, p in result["probs"].items())
            print(f"  Probs: {probs_str}")

        # Show explainability tokens if available
        if result["sentences"]:
            all_tokens = []
            for sr in result["sentences"]:
                all_tokens.extend(sr.get("top_tokens", []))
            if all_tokens:
                all_tokens.sort(key=lambda x: x["score"], reverse=True)
                seen = set()
                unique = []
                for t in all_tokens:
                    if t["clean_token"].lower() not in seen:
                        seen.add(t["clean_token"].lower())
                        unique.append(t)
                    if len(unique) >= 5:
                        break
                tstr = ", ".join(f"{t['clean_token']}({t['score']:.2f})" for t in unique)
                print(f"  Key tokens: {tstr}")

        if pred != true:
            errors.append({**article, **result})

    # ── Aggregate metrics ──
    metrics = compute_metrics(predictions, ground_truths)
    print_metrics(metrics, "DOCUMENT-LEVEL PREDICTION RESULTS")
    print(f"\n  Total time: {total_time:.1f}s  "
          f"({total_time / len(articles):.2f}s/article)")

    # ── Error analysis ──
    if errors:
        print(f"\n{'═' * 70}")
        print(f"  ERROR ANALYSIS — {len(errors)} misclassifications")
        print(f"{'═' * 70}")

        # Error pattern: what gets confused with what?
        confusion_pairs = Counter()
        for e in errors:
            pred = e.get("prediction") or "Center"
            confusion_pairs[(e["true_label"], pred)] += 1

        print("\n  Most common confusions:")
        for (true, pred), count in confusion_pairs.most_common(5):
            print(f"    {true:>8} → {pred:<8}  ({count}x)")

        # Show a few examples
        print("\n  Sample errors:")
        for e in errors[:5]:
            pred = e.get("prediction") or "Center"
            print(f"\n    Source: {e['source']} ({e['bias_detail']})")
            print(f"    True: {e['true_label']}  →  Predicted: {pred}")
            if e.get("probs"):
                print(f"    Probs: {e['probs']}")
            print(f"    Body: {e['body'][:100]}...")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Quick Demo (no AllSides data needed)
# ═══════════════════════════════════════════════════════════════════════

def run_demo(detector):
    """Quick sanity check on hand-picked examples."""
    examples = [
        ("Left-leaning", "Progressive Democrats championed Medicare for All and free "
         "college tuition, arguing that universal healthcare is a fundamental right "
         "that the richest nation on Earth can and should guarantee to every citizen."),
        ("Center", "The Congressional Budget Office released its annual economic outlook "
         "on Wednesday, projecting GDP growth of 2.1% for the coming fiscal year and "
         "noting that the federal deficit is expected to widen modestly."),
        ("Right-leaning", "Conservative lawmakers defended free market principles and "
         "called for reduced government regulation, arguing that individual liberty and "
         "personal responsibility are the cornerstones of American prosperity."),
        ("Non-political", "The weather forecast predicts rain this weekend with "
         "temperatures dropping into the low 50s across the eastern seaboard."),
    ]

    print("\n" + "═" * 70)
    print("  QUICK DEMO — Hand-picked examples")
    print("═" * 70)

    for label, text in examples:
        result = detector.analyze(text, explain_top_k=1)
        pred = result["prediction"] or "(non-political)"
        pol = "political" if result["is_political"] else "non-political"
        print(f"\n  [{label}]")
        print(f"  Text: {text[:80]}...")
        print(f"  → {pol} (conf={result['political_confidence']:.2f})")
        if result["probs"]:
            print(f"  → Prediction: {pred}  |  "
                  f"L={result['probs']['Left']:.3f}  "
                  f"C={result['probs']['Center']:.3f}  "
                  f"R={result['probs']['Right']:.3f}")

        # Show explainability
        if result["sentences"]:
            for sr in result["sentences"]:
                if sr.get("top_tokens"):
                    tstr = ", ".join(
                        f"{t['clean_token']}({t['score']:.2f})"
                        for t in sr["top_tokens"][:5]
                    )
                    print(f"  → Key tokens: {tstr}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-pass bias detection: document-level prediction + Gradient×Input explainability"
    )
    parser.add_argument(
        "--allsides", nargs="+", default=[],
        help="Path(s) to AllSides JSON file(s)",
    )
    parser.add_argument(
        "--explain-top-k", type=int, default=0,
        help="Number of sentences to explain per article (0 = prediction only, faster)",
    )
    parser.add_argument(
        "--bias-model", default="models/bias_detector",
        help="Path to bias detection model",
    )
    parser.add_argument(
        "--pol-model", default="models/politicalness_filter",
        help="Path to politicalness filter model",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU even if GPU is available",
    )
    parser.add_argument(
    "--pol-threshold", type=float, default=0.5,
    help="Politicalness filter threshold (0.0 = disabled, 0.3 = relaxed)",
    )
    parser.add_argument(
        "--center-bias", type=float, default=0.0,
        help="Logit offset for Center class (negative = less Center, e.g. -1.0)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run quick demo on hand-picked examples",
    )
    args = parser.parse_args()

    device = "cpu" if args.no_gpu else None

    # Load models
    detector = BiasDetector(
    bias_path=args.bias_model,
    pol_path=args.pol_model,
    device=device,
    pol_threshold=args.pol_threshold,
    center_bias=args.center_bias,
    )

    # Demo mode
    if args.demo or not args.allsides:
        run_demo(detector)
        if not args.allsides:
            print("\n  Tip: pass --allsides <file.json> to run full evaluation")
            return

    # Load AllSides data
    articles = load_allsides_data(args.allsides)
    print(f"Loaded {len(articles)} articles from {len(args.allsides)} file(s)")

    dist = Counter(a["true_label"] for a in articles)
    print(f"Distribution: {dict(dist)}")

    # Separate by bias_detail for finer-grained analysis
    detail_dist = Counter(a["bias_detail"] for a in articles)
    print(f"Detail distribution: {dict(detail_dist)}")

    # Run evaluation
    print(f"\n{'#' * 70}")
    print(f"# EVALUATION — Document-Level Prediction"
          + (f" + Gradient×Input (top-{args.explain_top_k})" if args.explain_top_k else ""))
    print(f"{'#' * 70}")

    metrics = run_evaluation(articles, detector, explain_top_k=args.explain_top_k)

    # ── Lean vs. Strong bias breakdown ──
    print(f"\n{'═' * 70}")
    print(f"  BREAKDOWN: Strong vs Lean bias articles")
    print(f"{'═' * 70}")

    for subset_name, filter_fn in [
        ("Strong (Left/Right only)", lambda a: a["bias_detail"] in ("Left", "Right")),
        ("Lean (Lean Left/Lean Right only)", lambda a: a["bias_detail"] in ("Lean Left", "Lean Right")),
        ("Center only", lambda a: a["bias_detail"] == "Center"),
    ]:
        subset = [a for a in articles if filter_fn(a)]
        if not subset:
            continue

        # Re-run predictions (cheap — already fast)
        preds = []
        truths = []
        for a in subset:
            r = detector.analyze(a["body"], headline=a["headline"])
            preds.append(r["prediction"] or "Center")
            truths.append(a["true_label"])

        sub_metrics = compute_metrics(preds, truths)
        print(f"\n  {subset_name}: {sub_metrics['accuracy']:.1%} accuracy "
              f"({sub_metrics['correct']}/{sub_metrics['total']}), "
              f"Macro F1={sub_metrics['macro_f1']:.3f}")

    print(f"\n{'═' * 70}")
    print("  ✓ EVALUATION COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()