"""
Test on how Volfs model performs on newly scrapped data it has never seen before.
Full pipeline evaluation.
politicalness filter -> sentence split -> bias detection
-> weighted aggregation -> comparison to AllSides ground truth labels.
"""

import json
import time
import torch
import numpy as np
import spacy
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients

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
    'same', 'any', 'been', 'there', 'here',
}

PUNCTUATION = {
    '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']',
    '-', '-', '—', '▁.', '▁,', '▁!', '▁?', '▁;', '▁:', '▁-'
}


def load_models():
    print("Loading models...")
    bias_model = AutoModelForSequenceClassification.from_pretrained("../models/demo_models/bias_detector")
    bias_tokenizer = AutoTokenizer.from_pretrained("../models/demo_models/bias_detector")
    pol_model = AutoModelForSequenceClassification.from_pretrained("../models/demo_models/politicalness_filter")
    pol_tokenizer = AutoTokenizer.from_pretrained("../models/demo_models/politicalness_filter")
    bias_model.eval()
    pol_model.eval()
    print("Models loaded\n")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


# politicalness filter (NLI)
def check_politicalness(text, model, tokenizer):
    hypothesis = "This text is about politics."
    inputs = tokenizer(text, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    political_prob = float(probs[entail_idx])
    return {'is_political': political_prob > 0.5, 'confidence': political_prob}


# sentence processing
def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def merge_short_sentences(sentences, min_tokens=20, max_tokens=150):
    merged = []
    buffer = ""
    for sent in sentences:
        if len(sent.split()) < min_tokens:
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


# bias detection (with optional IG explainability)
def predict_bias(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    return {
        'predicted_label': LABELS[pred],
        'confidence': float(probs[pred]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
    }


def predict_bias_with_ig(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(probs).item()

    # IG attribution
    top_tokens = []
    embedding_layer = _get_embedding_layer(model)
    if embedding_layer is not None:
        lig = LayerIntegratedGradients(
            lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits,
            embedding_layer
        )
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)
        attributions = lig.attribute(
            inputs=input_ids, baselines=baseline_ids, target=pred,
            additional_forward_args=(attention_mask,), n_steps=30,
        )
        attr = attributions.sum(dim=-1).squeeze(0).abs()
        if attr.max() > 0:
            normalized = (attr / attr.max()).detach().numpy()
        else:
            normalized = attr.detach().numpy()
        top_tokens = _build_filtered_token_list(tokens, normalized)

    return {
        'predicted_label': LABELS[pred],
        'confidence': float(probs[pred]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
        'top_tokens': top_tokens[:5],
    }


def _get_embedding_layer(model):
    for attr in ['deberta', 'bert', 'roberta', 'distilbert']:
        if hasattr(model, attr):
            return getattr(model, attr).embeddings.word_embeddings
    return None


def _build_filtered_token_list(tokens, scores):
    result = []
    for i, (token, score) in enumerate(zip(tokens, scores)):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue
        clean = token.replace('##', '').replace('▁', '').strip()
        if not clean or clean in PUNCTUATION:
            continue
        if all(c in '.,!?;:\'"()[]<>-–—' for c in clean):
            continue
        if clean.lower() in STOP_WORDS:
            continue
        result.append({'token': clean, 'score': float(score), 'position': i})
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


# aggregation
def compute_sentence_weight(sentence, probs_dict, low_conf_threshold=0.45):
    weight = 1.0
    if max(probs_dict.values()) < low_conf_threshold:
        weight *= 0.3
    if sentence.strip().startswith('"') or sentence.strip().startswith('\u201c'):
        weight *= 0.5
    if len(sentence.split()) < 10:
        weight *= 0.5
    return weight


def aggregate_sentence_predictions(sentence_results):
    if not sentence_results:
        return None, {}
    weighted_probs = []
    for sr in sentence_results:
        probs = torch.tensor([sr['all_probs'][l] for l in LABELS])
        weighted_probs.append(probs * sr['weight'])
    total = torch.stack(weighted_probs).sum(dim=0)
    if total.sum() > 0:
        normalized = total / total.sum()
    else:
        normalized = total
    idx = torch.argmax(total).item()
    return LABELS[idx], {l: float(normalized[i]) for i, l in enumerate(LABELS)}


# three pipeline modes: sequential, parallel, batched
def analyze_article_sequential(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer, with_ig=False):
    """Original sequential: filter each sentence, then predict bias."""
    start = time.time()
    sentences = split_into_sentences(body)
    sentences = merge_short_sentences(sentences)

    sentence_results = []
    filtered = 0
    for sent in sentences:
        pol = check_politicalness(sent, pol_model, pol_tokenizer)
        if not pol['is_political']:
            filtered += 1
            continue
        bias = predict_bias_with_ig(sent, bias_model, bias_tokenizer) if with_ig else predict_bias(sent, bias_model, bias_tokenizer)
        weight = compute_sentence_weight(sent, bias['all_probs'])
        sentence_results.append({
            'text': sent[:100], 'predicted_label': bias['predicted_label'],
            'confidence': bias['confidence'], 'all_probs': bias['all_probs'],
            'weight': weight, 'top_tokens': bias.get('top_tokens', []),
        })

    overall_label, overall_probs = aggregate_sentence_predictions(sentence_results)
    return {
        'overall_prediction': overall_label, 'overall_probs': overall_probs,
        'sentences': sentence_results, 'total_sentences': len(sentences),
        'filtered_sentences': filtered, 'analyzed_sentences': len(sentence_results),
        'elapsed_seconds': time.time() - start, 'mode': 'sequential',
    }


def analyze_article_parallel(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer, with_ig=False):
    """Run politicalness and bias detection in parallel threads per sentence."""
    start = time.time()
    sentences = split_into_sentences(body)
    sentences = merge_short_sentences(sentences)

    sentence_results = []
    filtered = 0
    for sent in sentences:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pol_future = executor.submit(check_politicalness, sent, pol_model, pol_tokenizer)
            bias_fn = predict_bias_with_ig if with_ig else predict_bias
            bias_future = executor.submit(bias_fn, sent, bias_model, bias_tokenizer)
            pol = pol_future.result()
            bias = bias_future.result()

        if not pol['is_political']:
            filtered += 1
            continue
        weight = compute_sentence_weight(sent, bias['all_probs'])
        sentence_results.append({
            'text': sent[:100], 'predicted_label': bias['predicted_label'],
            'confidence': bias['confidence'], 'all_probs': bias['all_probs'],
            'weight': weight, 'top_tokens': bias.get('top_tokens', []),
        })

    overall_label, overall_probs = aggregate_sentence_predictions(sentence_results)
    return {
        'overall_prediction': overall_label, 'overall_probs': overall_probs,
        'sentences': sentence_results, 'total_sentences': len(sentences),
        'filtered_sentences': filtered, 'analyzed_sentences': len(sentence_results),
        'elapsed_seconds': time.time() - start, 'mode': 'parallel',
    }


def analyze_article_batched(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer, with_ig=False):
    """Batch all sentences through politicalness first, then batch through bias model."""
    start = time.time()
    sentences = split_into_sentences(body)
    sentences = merge_short_sentences(sentences)
    if not sentences:
        return {'overall_prediction': None, 'overall_probs': {}, 'sentences': [],
                'total_sentences': 0, 'filtered_sentences': 0, 'analyzed_sentences': 0,
                'elapsed_seconds': 0, 'mode': 'batched'}

    # batch politicalness
    hypothesis = "This text is about politics."
    pol_inputs = pol_tokenizer([(s, hypothesis) for s in sentences],
                               return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        pol_probs = torch.nn.functional.softmax(pol_model(**pol_inputs).logits, dim=-1)
    label2id = pol_model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    political_mask = pol_probs[:, entail_idx] > 0.5

    political_sentences = [s for i, s in enumerate(sentences) if political_mask[i]]
    filtered = len(sentences) - len(political_sentences)

    # batch bias detection
    sentence_results = []
    if political_sentences:
        if with_ig:
            # IG can't batch, fall back to sequential
            for sent in political_sentences:
                bias = predict_bias_with_ig(sent, bias_model, bias_tokenizer)
                weight = compute_sentence_weight(sent, bias['all_probs'])
                sentence_results.append({
                    'text': sent[:100], 'predicted_label': bias['predicted_label'],
                    'confidence': bias['confidence'], 'all_probs': bias['all_probs'],
                    'weight': weight, 'top_tokens': bias.get('top_tokens', []),
                })
        else:
            bias_inputs = bias_tokenizer(political_sentences, return_tensors="pt",
                                          truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                bias_probs = torch.nn.functional.softmax(bias_model(**bias_inputs).logits, dim=-1)
            for i, sent in enumerate(political_sentences):
                p = bias_probs[i]
                pred = torch.argmax(p).item()
                all_probs = {l: float(p[j]) for j, l in enumerate(LABELS)}
                weight = compute_sentence_weight(sent, all_probs)
                sentence_results.append({
                    'text': sent[:100], 'predicted_label': LABELS[pred],
                    'confidence': float(p[pred]), 'all_probs': all_probs,
                    'weight': weight, 'top_tokens': [],
                })

    overall_label, overall_probs = aggregate_sentence_predictions(sentence_results)
    return {
        'overall_prediction': overall_label, 'overall_probs': overall_probs,
        'sentences': sentence_results, 'total_sentences': len(sentences),
        'filtered_sentences': filtered, 'analyzed_sentences': len(sentence_results),
        'elapsed_seconds': time.time() - start, 'mode': 'batched',
    }


# metrics
def compute_metrics(predictions, ground_truths):
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)

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

    confusion = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for p, g in zip(predictions, ground_truths):
        confusion[g][p] += 1

    return {
        'accuracy': correct / total if total > 0 else 0,
        'macro_f1': np.mean([m['f1'] for m in class_metrics.values()]),
        'correct': correct, 'total': total,
        'per_class': class_metrics, 'confusion': confusion,
    }


def print_metrics(metrics, title="Results"):
    print(f"\n{title}")
    print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    for label in LABELS:
        m = metrics['per_class'][label]
        print(f"  {label:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")


# allsides data loading
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


# run evaluation
def run_evaluation(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                   mode='sequential', with_ig=True):
    pipeline_fn = {
        'sequential': analyze_article_sequential,
        'parallel': analyze_article_parallel,
        'batched': analyze_article_batched,
    }[mode]

    predictions, ground_truths = [], []
    total_time = 0

    for i, article in enumerate(articles):
        result = pipeline_fn(article['body'], bias_model, bias_tokenizer,
                             pol_model, pol_tokenizer, with_ig=with_ig)
        total_time += result['elapsed_seconds']
        pred = result['overall_prediction'] or 'Center'
        predictions.append(pred)
        ground_truths.append(article['true_label'])

        match = "✓" if pred == article['true_label'] else "✗"
        print(f"  [{i+1}/{len(articles)}] {match} True: {article['true_label']:>7}  Pred: {pred:>7}  ({article['source']})")

    return predictions, ground_truths, total_time


if __name__ == "__main__":
    data_path = Path("../datasets/other_data/allsides_data.json")

    articles = load_allsides_data(data_path)
    print(f"Loaded {len(articles)} articles")
    print(f"Distribution: {dict(Counter(a['true_label'] for a in articles))}")

    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    # run with IG (sequential)
    print("\nSequential + IG")
    preds, truths, elapsed = run_evaluation(articles, bias_model, bias_tokenizer,
                                             pol_model, pol_tokenizer, mode='sequential', with_ig=True)
    metrics_ig = compute_metrics(preds, truths)
    print_metrics(metrics_ig, "Sequential + IG")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(articles):.2f}s/article)")

    # run batched without IG (speed comparison)
    print("\n--- Batched (no IG) ---")
    preds_b, truths_b, elapsed_b = run_evaluation(articles, bias_model, bias_tokenizer,
                                                    pol_model, pol_tokenizer, mode='batched', with_ig=False)
    metrics_b = compute_metrics(preds_b, truths_b)
    print_metrics(metrics_b, "Batched")
    print(f"  Time: {elapsed_b:.1f}s ({elapsed_b/len(articles):.2f}s/article)")

    if elapsed_b > 0:
        print(f"\n  Speedup: {elapsed/elapsed_b:.1f}x faster with batching")