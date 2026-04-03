"""
Hybrid pipeline: predict bias on the full article, then use sentence-level
IG only for explainability.

The Volf model was trained on full articles, not individual sentences.
Splitting into sentences and aggregating maybe hurts accuracy? So we wanted to test
"""

import json
import time
import torch
import numpy as np
import spacy
from pathlib import Path
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nlp = spacy.load("en_core_web_sm")

LABELS = ['Left', 'Center', 'Right']
ALLSIDES_MAP = {
    'Left': 'Left', 'Lean Left': 'Left',
    'Center': 'Center',
    'Lean Right': 'Right', 'Right': 'Right',
}

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall',
    'can', 'that', 'this', 'these', 'those', 'it', 'its', 'not',
    'no', 'so', 'if', 'then', 'than', 'too', 'very', 'just',
    'about', 'up', 'out', 'also', 'as', 'from', 'into', 'he',
    'she', 'they', 'we', 'you', 'his', 'her', 'their', 'our',
    'my', 'me', 'him', 'them', 'us', 'who', 'which', 'what',
    'when', 'where', 'how', 'all', 'each', 'every', 'both',
    'more', 'most', 'other', 'some', 'such', 'only', 'own',
    'same', 'any', 'there', 'here', 'said', 'says',
}


def load_models(bias_path="../models/demo_models/bias_detector", pol_path="../models/demo_models/politicalness_filter"):
    print("Loading models..")
    bias_model = AutoModelForSequenceClassification.from_pretrained(bias_path)
    bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
    bias_model.eval()

    pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
    pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
    pol_model.eval()

    print("Models loaded")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


def check_political(text, model, tokenizer, threshold=0.5):
    hypothesis = "This text is about politics."
    inputs = tokenizer(text, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    conf = float(probs[entail_idx])
    return conf > threshold, conf


def predict_bias_full_article(text, model, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred_class = torch.argmax(probs).item()
    return {
        'predicted_label': LABELS[pred_class],
        'confidence': float(probs[pred_class]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
    }


def split_and_merge(text, min_tokens=20, max_tokens=150):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    merged = []
    buffer = ""
    for sent in sentences:
        n = len(sent.split())
        if n < min_tokens:
            buffer += " " + sent
            if len(buffer.split()) >= max_tokens:
                merged.append(buffer.strip())
                buffer = ""
        else:
            if buffer.strip():
                merged.append(buffer.strip())
                buffer = ""
            merged.append(sent)
    if buffer.strip():
        if merged:
            merged[-1] += " " + buffer.strip()
        else:
            merged.append(buffer.strip())
    return merged


def get_sentence_bias_scores(sentences, model, tokenizer):
    if not sentences:
        return []
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    results = []
    for i, sent in enumerate(sentences):
        p = probs[i]
        pred_class = torch.argmax(p).item()
        center_prob = float(p[LABELS.index('Center')])
        results.append({
            'text': sent,
            'predicted_label': LABELS[pred_class],
            'confidence': float(p[pred_class]),
            'all_probs': {l: float(p[j]) for j, l in enumerate(LABELS)},
            'bias_strength': 1.0 - center_prob,
        })
    return results


# IG explainability on a single sentence
def ig_attribution(text, model, tokenizer, target_class=None, n_steps=30):
    from captum.attr import LayerIntegratedGradients

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # get DeBERTa embedding layer
    embedding_layer = model.deberta.embeddings.word_embeddings

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    predicted_class = torch.argmax(probs).item()
    if target_class is None:
        target_class = predicted_class

    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, pad_token_id)

    lig = LayerIntegratedGradients(forward_func, embedding_layer)
    attributions = lig.attribute(
        inputs=input_ids, baselines=baseline_ids,
        target=target_class, additional_forward_args=(attention_mask,),
        n_steps=n_steps,
    )

    attr_scores = attributions.sum(dim=-1).abs()[0]
    if attr_scores.max() > 0:
        normalized = (attr_scores / attr_scores.max()).detach().numpy()
    else:
        normalized = attr_scores.detach().numpy()

    return {
        'top_tokens': _build_filtered_token_list(tokens, normalized)[:5],
    }


def _build_filtered_token_list(tokens, scores):
    result = []
    for i, (token, score) in enumerate(zip(tokens, scores)):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue
        clean = token.replace('##', '').replace('▁', '').strip()
        if not clean or all(c in '.,!?;:\'"()[]<>-–—' for c in clean):
            continue
        if clean.lower() in STOP_WORDS:
            continue
        result.append({'token': clean, 'score': float(score), 'position': i})
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


# main hybrid pipeline
def analyze_article(body, headline, bias_model, bias_tokenizer,
                    pol_model, pol_tokenizer, explain_top_k=2):
    start = time.time()

    # step 1: politicalness on full text
    is_political, pol_conf = check_political(body, pol_model, pol_tokenizer, threshold=0.3)
    if not is_political and headline:
        is_political, pol_conf = check_political(headline, pol_model, pol_tokenizer, threshold=0.3)

    if not is_political:
        return {
            'prediction': 'Center', 'confidence': 0.0,
            'all_probs': {'Left': 0, 'Center': 1, 'Right': 0},
            'is_political': False, 'political_confidence': pol_conf,
            'sentence_explanations': [],
            'elapsed_seconds': time.time() - start,
        }

    # step 2: full-article bias prediction
    bias_result = predict_bias_full_article(body, bias_model, bias_tokenizer)

    # step 3: sentence-level IG on the most biased sentences (for explainability only)
    sentence_explanations = []
    if explain_top_k > 0:
        sentences = split_and_merge(body)
        if sentences:
            sent_scores = get_sentence_bias_scores(sentences, bias_model, bias_tokenizer)
            sent_scores.sort(key=lambda x: x['bias_strength'], reverse=True)
            for sr in sent_scores[:explain_top_k]:
                ig_result = ig_attribution(sr['text'], bias_model, bias_tokenizer)
                sentence_explanations.append({
                    'text': sr['text'][:150],
                    'predicted_label': sr['predicted_label'],
                    'confidence': sr['confidence'],
                    'top_tokens': ig_result['top_tokens'],
                })

    return {
        'prediction': bias_result['predicted_label'],
        'confidence': bias_result['confidence'],
        'all_probs': bias_result['all_probs'],
        'is_political': True, 'political_confidence': pol_conf,
        'sentence_explanations': sentence_explanations,
        'elapsed_seconds': time.time() - start,
    }


# evaluation on allsides data
def load_allsides_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    articles = []
    for story in data:
        for side in story.get('sides', []):
            bias_detail = side.get('bias_detail')
            body = side.get('body', '').strip()
            if not body or not bias_detail or bias_detail not in ALLSIDES_MAP:
                continue
            articles.append({
                'source': side.get('source', 'Unknown'),
                'headline': side.get('headline', ''),
                'body': body,
                'bias_detail': bias_detail,
                'true_label': ALLSIDES_MAP[bias_detail],
            })
    return articles


def compute_metrics(predictions, ground_truths):
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    class_metrics = {}
    for label in LABELS:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        class_metrics[label] = {'precision': prec, 'recall': rec, 'f1': f1,
                                'support': sum(1 for g in ground_truths if g == label)}

    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    confusion = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for p, g in zip(predictions, ground_truths):
        confusion[g][p] += 1

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'per_class': class_metrics, 'confusion': confusion}


def run_comparison(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer):
    preds_full, preds_sent, truths = [], [], []

    for i, article in enumerate(articles):
        body = article['body']
        truths.append(article['true_label'])

        # full-article prediction
        result = analyze_article(body, article['headline'],
                                 bias_model, bias_tokenizer, pol_model, pol_tokenizer)
        preds_full.append(result['prediction'])

        # sentence-aggregated prediction (for comparison)
        sentences = split_and_merge(body)
        sent_scores = get_sentence_bias_scores(sentences, bias_model, bias_tokenizer)
        if sent_scores:
            weighted = torch.zeros(3)
            for sr in sent_scores:
                probs_t = torch.tensor([sr['all_probs'][l] for l in LABELS])
                w = 0.3 if sr['confidence'] < 0.45 else 1.0
                weighted += probs_t * w
            sent_pred = LABELS[torch.argmax(weighted).item()]
        else:
            sent_pred = 'Center'
        preds_sent.append(sent_pred)

        match_f = "✓" if preds_full[-1] == truths[-1] else "✗"
        match_s = "✓" if preds_sent[-1] == truths[-1] else "✗"
        print(f"  [{i+1}/{len(articles)}] True: {truths[-1]:>7}  Full: {preds_full[-1]:>7} {match_f}  Sent: {preds_sent[-1]:>7} {match_s}  ({article['source']})")

    metrics_full = compute_metrics(preds_full, truths)
    metrics_sent = compute_metrics(preds_sent, truths)

    print(f"\nFull-article: Acc={metrics_full['accuracy']:.1%}  F1={metrics_full['macro_f1']:.3f}")
    print(f"Sentence-agg: Acc={metrics_sent['accuracy']:.1%}  F1={metrics_sent['macro_f1']:.3f}")

    return metrics_full, metrics_sent


if __name__ == "__main__":
    allsides_path = "../datasets/other_data/allsides_data.json"

    articles = load_allsides_data(allsides_path)
    print(f"Loaded {len(articles)} articles")
    print(f"Distribution: {dict(Counter(a['true_label'] for a in articles))}")

    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()
    run_comparison(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer)