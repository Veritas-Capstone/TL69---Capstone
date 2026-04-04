"""
Pass 1: Document-level prediction (full article -> single label).
        The model was trained on full articles so this gives best accuracy.
Pass 2: Sentence-level Gradient x Input explainability (optional).
        Only used to highlight which words/sentences drive the prediction.
"""

import json
import time
import re
import torch
import numpy as np
import warnings
from pathlib import Path
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

LABELS = ["Left", "Center", "Right"]

ALLSIDES_MAP = {
    "Left": "Left", "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right", "Right": "Right",
}

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
    "-", "\u2013", "\u2014", "\u2581.", "\u2581,", "\u2581!", "\u2581?", "\u2581;", "\u2581:", "\u2581-",
})

# multiple hypotheses for politicalness — take the max
POLITICALNESS_HYPOTHESES = [
    "This text is about politics or government policy.",
    "This text discusses a politically controversial or partisan topic.",
    "This text is about legislation, regulation, or political debate.",
    "This text discusses government officials or political parties.",
]


class BiasDetector:

    def __init__(self, bias_path="models/bias_detector",
                 pol_path="models/politicalness_filter",
                 device=None, use_fp16=True, pol_threshold=0.5):

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pol_threshold = pol_threshold
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        print(f"Device: {self.device}  |  FP16: {self.use_fp16}")

        # load bias model
        print(f"Loading bias model from {bias_path} ...")
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(bias_path)
        self.bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
        self.bias_model.eval().to(self.device)
        if self.use_fp16:
            self.bias_model.half()

        # load politicalness model
        print(f"Loading politicalness model from {pol_path} ...")
        self.pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
        self.pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
        self.pol_model.eval().to(self.device)
        if self.use_fp16:
            self.pol_model.half()

        label2id = self.pol_model.config.label2id
        self.entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))

        # embedding layer for Gradient x Input
        self._embedding_layer = self.bias_model.deberta.embeddings.word_embeddings
        print("Models loaded\n")

    # pass 1a: politicalness check
    def check_politicalness(self, text):
        pairs = [(text, h) for h in POLITICALNESS_HYPOTHESES]
        inputs = self.pol_tokenizer(
            pairs, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.pol_model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1)
        best_conf = float(probs[:, self.entail_idx].max())
        return best_conf > self.pol_threshold, best_conf

    # pass 1b: document-level bias prediction
    def predict_bias_document(self, text):
        inputs = self.bias_tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.bias_model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

        return {
            "predicted_label": LABELS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {l: float(probs[i]) for i, l in enumerate(LABELS)},
        }

    # pass 2: gradient x input explainability
    def gradient_x_input(self, text, target_class=None):
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

        embeddings = self._embedding_layer(input_ids)
        embeddings = embeddings.detach().requires_grad_(True)

        outputs = self.bias_model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits

        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        logits[0, target_class].backward()

        attr = (embeddings.grad * embeddings).sum(dim=-1).abs().squeeze(0)
        if attr.max() > 0:
            attr = attr / attr.max()
        scores = attr.detach().cpu().numpy()

        if was_fp16:
            self.bias_model.half()

        return self._build_token_list(tokens, scores)

    def explain_sentences(self, text, doc_prediction, top_k=3):
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # batch bias prediction on all sentences (fast, no grad)
        sent_inputs = self.bias_tokenizer(
            sentences, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            sent_logits = self.bias_model(**sent_inputs).logits
        sent_probs = torch.softmax(sent_logits.float(), dim=-1)

        # score sentences by bias strength (1 - center_prob)
        sentence_data = []
        for i, sent in enumerate(sentences):
            probs = sent_probs[i]
            pred_idx = torch.argmax(probs).item()
            center_prob = float(probs[1])

            sentence_data.append({
                "text": sent,
                "predicted_label": LABELS[pred_idx],
                "confidence": float(probs[pred_idx]),
                "all_probs": {l: float(probs[j]) for j, l in enumerate(LABELS)},
                "bias_strength": 1.0 - center_prob,
                "top_tokens": [],
            })

        # run gradient x input on the top-k most biased sentences
        ranked = sorted(enumerate(sentence_data), key=lambda x: x[1]["bias_strength"], reverse=True)
        for idx, _ in ranked[:top_k]:
            sd = sentence_data[idx]
            sd["top_tokens"] = self.gradient_x_input(sd["text"])[:5]
            sd["explained"] = True

        return sentence_data

    # combined pipeline
    def analyze(self, body, headline=None, explain_top_k=0):
        start = time.time()
        full_text = body
        if headline:
            full_text = headline + ". " + body

        is_political, pol_conf = self.check_politicalness(full_text)
        if not is_political:
            return {
                "is_political": False, "political_confidence": pol_conf,
                "prediction": None, "probs": None,
                "sentences": [], "elapsed": time.time() - start,
            }

        doc_result = self.predict_bias_document(full_text)

        sentences = []
        if explain_top_k > 0:
            sentences = self.explain_sentences(body, doc_result, top_k=explain_top_k)

        return {
            "is_political": True, "political_confidence": pol_conf,
            "prediction": doc_result["predicted_label"],
            "confidence": doc_result["confidence"],
            "probs": doc_result["all_probs"],
            "sentences": sentences,
            "elapsed": time.time() - start,
        }

    @staticmethod
    def _split_sentences(text):
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        expanded = []
        for chunk in raw:
            parts = chunk.split("\n\n")
            expanded.extend(p.strip() for p in parts if p.strip())
        merged = []
        for s in expanded:
            if merged and len(s.split()) < 8:
                merged[-1] += " " + s
            else:
                merged.append(s)
        return [s for s in merged if len(s.split()) >= 5]

    @staticmethod
    def _build_token_list(tokens, scores):
        result = []
        for i, (token, score) in enumerate(zip(tokens, scores)):
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
                continue
            clean = token.replace("##", "").replace("\u2581", "").strip()
            if not clean or clean in PUNCTUATION:
                continue
            if all(c in '.,!?;:\'"()[]<>-\u2013\u2014/' for c in clean):
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


# data loading
def load_allsides_data(filepaths):
    articles = []
    seen_urls = set()
    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = json.load(f)
        for story in data:
            for side in story.get("sides", []):
                bias_detail = side.get("bias_detail")
                body = (side.get("body") or "").strip()
                url = side.get("original_url", "")
                if not body or not bias_detail or bias_detail not in ALLSIDES_MAP:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append({
                    "source": side.get("source", "Unknown"),
                    "headline": side.get("headline", ""),
                    "body": body,
                    "bias_detail": bias_detail,
                    "true_label": ALLSIDES_MAP[bias_detail],
                    "url": url,
                })
    return articles


# metrics
def compute_metrics(predictions, ground_truths):
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


def print_metrics(metrics, title="Results"):
    print(f"\n{title}")
    print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    for label in LABELS:
        m = metrics["per_class"][label]
        print(f"  {label:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")


# evaluation
def run_evaluation(articles, detector, explain_top_k=0):
    predictions, ground_truths = [], []
    total_time = 0

    for i, article in enumerate(articles):
        result = detector.analyze(
            body=article["body"], headline=article["headline"],
            explain_top_k=explain_top_k,
        )
        total_time += result["elapsed"]
        pred = result["prediction"] or "Center"
        predictions.append(pred)
        ground_truths.append(article["true_label"])

        match = "yes" if pred == article["true_label"] else "no"
        print(f"  [{i+1}/{len(articles)}] {match} True: {article['true_label']:>7}  Pred: {pred:>7}  ({article['source']})")

    metrics = compute_metrics(predictions, ground_truths)
    print_metrics(metrics, "Document-Level Prediction Results")
    print(f"  Time: {total_time:.1f}s ({total_time / len(articles):.2f}s/article)")
    return metrics


# demo
def run_demo(detector):
    examples = [
        ("Left-leaning", "Progressive Democrats championed Medicare for All and free "
         "college tuition, arguing that universal healthcare is a fundamental right."),
        ("Center", "The Congressional Budget Office released its annual economic outlook "
         "on Wednesday, projecting GDP growth of 2.1% for the coming fiscal year."),
        ("Right-leaning", "Conservative lawmakers defended free market principles and "
         "called for reduced government regulation, arguing that individual liberty "
         "and personal responsibility are paramount."),
        ("Non-political", "The weather forecast predicts rain this weekend with "
         "temperatures dropping into the low 50s."),
    ]

    print("\n--- Demo ---")
    for label, text in examples:
        result = detector.analyze(text, explain_top_k=1)
        pred = result["prediction"] or "(non-political)"
        print(f"\n  [{label}] -> {pred} (pol_conf={result['political_confidence']:.2f})")
        if result["probs"]:
            print(f"  L={result['probs']['Left']:.3f}  C={result['probs']['Center']:.3f}  R={result['probs']['Right']:.3f}")


if __name__ == "__main__":
    allsides_files = ["allsides_data.json"]

    detector = BiasDetector()
    run_demo(detector)

    # load and evaluate on allsides data
    existing = [f for f in allsides_files if Path(f).exists()]
    if existing:
        articles = load_allsides_data(existing)
        print(f"\nLoaded {len(articles)} articles")
        print(f"Distribution: {dict(Counter(a['true_label'] for a in articles))}")
        run_evaluation(articles, detector, explain_top_k=0)
    else:
        print("\nNo allsides data found. Pass --allsides <file.json> or run allsides_scrapper.py first.")