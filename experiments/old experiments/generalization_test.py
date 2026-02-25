"""
Full Pipeline Evaluation on AllSides Data
==========================================
Tests: politicalness filter → sentence split → per-sentence bias detection
       with IG explainability → weighted aggregation → comparison to AllSides labels

Also includes:
- Parallel processing mode (politicalness + bias in parallel)
- Per-article and aggregate metrics
- Detailed explainability output
"""

import json
import time
import torch
import numpy as np
import spacy
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("⚠️  captum not installed — using occlusion fallback")

nlp = spacy.load("en_core_web_sm")


# ===========================================================================
# Constants
# ===========================================================================

LABELS = ['Left', 'Center', 'Right']

# Map AllSides bias_detail strings to our 3-class labels
ALLSIDES_MAP = {
    'Left': 'Left',
    'Lean Left': 'Left',
    'Center': 'Center',
    'Lean Right': 'Right',
    'Right': 'Right',
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
    '-', '–', '—', '▁.', '▁,', '▁!', '▁?', '▁;', '▁:', '▁-'
}


# ===========================================================================
# Model Loading
# ===========================================================================

def load_models():
    print("Loading models...")
    bias_model = AutoModelForSequenceClassification.from_pretrained("models/bias_detector")
    bias_tokenizer = AutoTokenizer.from_pretrained("models/bias_detector")

    pol_model = AutoModelForSequenceClassification.from_pretrained("models/politicalness_filter")
    pol_tokenizer = AutoTokenizer.from_pretrained("models/politicalness_filter")

    bias_model.eval()
    pol_model.eval()

    print("✓ Models loaded\n")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


# ===========================================================================
# Politicalness Filter (NLI — corrected)
# ===========================================================================

def check_politicalness(text, model, tokenizer):
    hypothesis = "This text is about politics."
    inputs = tokenizer(text, hypothesis, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    political_prob = float(probs[entail_idx])

    return {
        'is_political': political_prob > 0.5,
        'confidence': political_prob
    }


# ===========================================================================
# Sentence Splitting
# ===========================================================================

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def merge_short_sentences(sentences, min_tokens=20, max_tokens=150):
    merged = []
    buffer = ""

    for sent in sentences:
        num_tokens = len(sent.split())
        if num_tokens < min_tokens:
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
        # Attach leftover to last sentence if possible
        if merged:
            merged[-1] += " " + buffer.strip()
        else:
            merged.append(buffer.strip())

    return merged


# ===========================================================================
# Bias Detection
# ===========================================================================

def predict_bias(text, model, tokenizer):
    """Simple bias prediction without explainability (fast path)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    predicted_class = torch.argmax(probs).item()
    return {
        'predicted_label': LABELS[predicted_class],
        'confidence': float(probs[predicted_class]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
    }


def predict_bias_with_ig(text, model, tokenizer):
    """Bias prediction with Integrated Gradients explainability."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    predicted_class = torch.argmax(probs).item()

    # IG attribution
    top_tokens = []
    if HAS_CAPTUM:
        embedding_layer = _get_embedding_layer(model)
        if embedding_layer is not None:
            lig = LayerIntegratedGradients(
                lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits,
                embedding_layer
            )
            baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)

            attributions = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                target=predicted_class,
                additional_forward_args=(attention_mask,),
                n_steps=30,
                return_convergence_delta=False
            )

            token_attr = attributions.sum(dim=-1).squeeze(0).abs()
            if token_attr.max() > 0:
                normalized = (token_attr / token_attr.max()).detach().numpy()
            else:
                normalized = token_attr.detach().numpy()

            top_tokens = _build_filtered_token_list(tokens, normalized)

    return {
        'predicted_label': LABELS[predicted_class],
        'confidence': float(probs[predicted_class]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
        'top_tokens': top_tokens[:5],
    }


def _get_embedding_layer(model):
    for attr in ['deberta', 'bert', 'roberta', 'distilbert']:
        if hasattr(model, attr):
            return getattr(model, attr).embeddings.word_embeddings
    return None


def _build_filtered_token_list(tokens, scores):
    """Build token list, filtering special tokens, punctuation, AND stop words."""
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
        result.append({
            'token': clean,
            'score': float(score),
            'position': i
        })
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


# ===========================================================================
# Aggregation
# ===========================================================================

def compute_sentence_weight(sentence, probs_dict, low_conf_threshold=0.45):
    """Compute per-sentence weight for aggregation."""
    peak = max(probs_dict.values())
    weight = 1.0

    if peak < low_conf_threshold:
        weight *= 0.3

    stripped = sentence.strip()
    if stripped.startswith('"') or stripped.startswith('\u201c'):
        weight *= 0.5

    # Downweight very short sentences (less context = less reliable)
    if len(sentence.split()) < 10:
        weight *= 0.5

    return weight


def aggregate_sentence_predictions(sentence_results):
    """Weighted aggregation of sentence-level predictions."""
    if not sentence_results:
        return None, {}

    weighted_probs = []
    for sr in sentence_results:
        probs = torch.tensor([sr['all_probs'][l] for l in LABELS])
        weighted_probs.append(probs * sr['weight'])

    total = torch.stack(weighted_probs).sum(dim=0)

    # Normalize to get a proper distribution
    if total.sum() > 0:
        normalized = total / total.sum()
    else:
        normalized = total

    overall_idx = torch.argmax(total).item()
    return LABELS[overall_idx], {l: float(normalized[i]) for i, l in enumerate(LABELS)}


# ===========================================================================
# Sequential Pipeline (Original)
# ===========================================================================

def analyze_article_sequential(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                                with_ig=False):
    """Original sequential pipeline: filter first, then bias."""
    start = time.time()

    sentences = split_into_sentences(body)
    sentences = merge_short_sentences(sentences)

    sentence_results = []
    filtered_count = 0

    for sent in sentences:
        # Step 1: politicalness filter
        pol = check_politicalness(sent, pol_model, pol_tokenizer)

        if not pol['is_political']:
            filtered_count += 1
            continue

        # Step 2: bias detection
        if with_ig:
            bias = predict_bias_with_ig(sent, bias_model, bias_tokenizer)
        else:
            bias = predict_bias(sent, bias_model, bias_tokenizer)

        weight = compute_sentence_weight(sent, bias['all_probs'])

        sentence_results.append({
            'text': sent[:100],
            'political_conf': pol['confidence'],
            'predicted_label': bias['predicted_label'],
            'confidence': bias['confidence'],
            'all_probs': bias['all_probs'],
            'weight': weight,
            'top_tokens': bias.get('top_tokens', []),
        })

    overall_label, overall_probs = aggregate_sentence_predictions(sentence_results)
    elapsed = time.time() - start

    return {
        'overall_prediction': overall_label,
        'overall_probs': overall_probs,
        'sentences': sentence_results,
        'total_sentences': len(sentences),
        'filtered_sentences': filtered_count,
        'analyzed_sentences': len(sentence_results),
        'elapsed_seconds': elapsed,
        'mode': 'sequential',
    }


# ===========================================================================
# Parallel Pipeline (New — filter + bias simultaneously)
# ===========================================================================

def analyze_article_parallel(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                              with_ig=False):
    """
    Parallel pipeline: run politicalness AND bias detection on each sentence
    simultaneously, then discard non-political results during aggregation.

    This is faster because we don't wait for the politicalness check before
    starting bias detection. The trade-off is wasted compute on non-political
    sentences, but for news articles most sentences are political anyway.
    """
    start = time.time()

    sentences = split_into_sentences(body)
    sentences = merge_short_sentences(sentences)

    sentence_results = []
    filtered_count = 0

    # Run both models on all sentences
    # (For true GPU parallelism you'd batch these, but this shows the pattern)
    for sent in sentences:
        # Run both in parallel using threads (IO-bound with torch.no_grad)
        with ThreadPoolExecutor(max_workers=2) as executor:
            pol_future = executor.submit(check_politicalness, sent, pol_model, pol_tokenizer)

            if with_ig:
                bias_future = executor.submit(predict_bias_with_ig, sent, bias_model, bias_tokenizer)
            else:
                bias_future = executor.submit(predict_bias, sent, bias_model, bias_tokenizer)

            pol = pol_future.result()
            bias = bias_future.result()

        if not pol['is_political']:
            filtered_count += 1
            continue

        weight = compute_sentence_weight(sent, bias['all_probs'])

        sentence_results.append({
            'text': sent[:100],
            'political_conf': pol['confidence'],
            'predicted_label': bias['predicted_label'],
            'confidence': bias['confidence'],
            'all_probs': bias['all_probs'],
            'weight': weight,
            'top_tokens': bias.get('top_tokens', []),
        })

    overall_label, overall_probs = aggregate_sentence_predictions(sentence_results)
    elapsed = time.time() - start

    return {
        'overall_prediction': overall_label,
        'overall_probs': overall_probs,
        'sentences': sentence_results,
        'total_sentences': len(sentences),
        'filtered_sentences': filtered_count,
        'analyzed_sentences': len(sentence_results),
        'elapsed_seconds': elapsed,
        'mode': 'parallel',
    }


# ===========================================================================
# Batched Pipeline (Best for throughput)
# ===========================================================================

def analyze_article_batched(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                             with_ig=False):
    """
    Batched pipeline: batch all sentences through politicalness model first,
    then batch surviving sentences through bias model.
    Most efficient for GPU inference.
    """
    start = time.time()

    sentences = split_into_sentences(body)
    sentences = merge_short_sentences(sentences)

    if not sentences:
        return {
            'overall_prediction': None, 'overall_probs': {},
            'sentences': [], 'total_sentences': 0,
            'filtered_sentences': 0, 'analyzed_sentences': 0,
            'elapsed_seconds': 0, 'mode': 'batched',
        }

    # Batch politicalness check
    hypothesis = "This text is about politics."
    pol_inputs = pol_tokenizer(
        [(s, hypothesis) for s in sentences],
        return_tensors="pt", truncation=True, max_length=512, padding=True
    )

    with torch.no_grad():
        pol_outputs = pol_model(**pol_inputs)
        pol_probs = torch.nn.functional.softmax(pol_outputs.logits, dim=-1)

    label2id = pol_model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))

    # Filter political sentences
    political_mask = pol_probs[:, entail_idx] > 0.5
    political_sentences = []
    political_confs = []
    filtered_count = 0

    for i, sent in enumerate(sentences):
        if political_mask[i]:
            political_sentences.append(sent)
            political_confs.append(float(pol_probs[i, entail_idx]))
        else:
            filtered_count += 1

    # Batch bias detection on political sentences
    sentence_results = []
    if political_sentences:
        if with_ig:
            # IG doesn't batch well — fall back to sequential for explainability
            for sent, pol_conf in zip(political_sentences, political_confs):
                bias = predict_bias_with_ig(sent, bias_model, bias_tokenizer)
                weight = compute_sentence_weight(sent, bias['all_probs'])
                sentence_results.append({
                    'text': sent[:100],
                    'political_conf': pol_conf,
                    'predicted_label': bias['predicted_label'],
                    'confidence': bias['confidence'],
                    'all_probs': bias['all_probs'],
                    'weight': weight,
                    'top_tokens': bias.get('top_tokens', []),
                })
        else:
            bias_inputs = bias_tokenizer(
                political_sentences,
                return_tensors="pt", truncation=True, max_length=256, padding=True
            )
            with torch.no_grad():
                bias_outputs = bias_model(**bias_inputs)
                bias_probs = torch.nn.functional.softmax(bias_outputs.logits, dim=-1)

            for i, (sent, pol_conf) in enumerate(zip(political_sentences, political_confs)):
                probs = bias_probs[i]
                pred_class = torch.argmax(probs).item()
                all_probs = {l: float(probs[j]) for j, l in enumerate(LABELS)}
                weight = compute_sentence_weight(sent, all_probs)

                sentence_results.append({
                    'text': sent[:100],
                    'political_conf': pol_conf,
                    'predicted_label': LABELS[pred_class],
                    'confidence': float(probs[pred_class]),
                    'all_probs': all_probs,
                    'weight': weight,
                    'top_tokens': [],
                })

    overall_label, overall_probs = aggregate_sentence_predictions(sentence_results)
    elapsed = time.time() - start

    return {
        'overall_prediction': overall_label,
        'overall_probs': overall_probs,
        'sentences': sentence_results,
        'total_sentences': len(sentences),
        'filtered_sentences': filtered_count,
        'analyzed_sentences': len(sentence_results),
        'elapsed_seconds': elapsed,
        'mode': 'batched',
    }


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(predictions, ground_truths):
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix."""
    assert len(predictions) == len(ground_truths)

    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    # Per-class metrics
    class_metrics = {}
    for label in LABELS:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for g in ground_truths if g == label)
        }

    # Confusion matrix
    confusion = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for p, g in zip(predictions, ground_truths):
        confusion[g][p] += 1

    # Macro F1
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'correct': correct,
        'total': total,
        'per_class': class_metrics,
        'confusion': confusion,
    }


def print_metrics(metrics, title="METRICS"):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro F1:  {metrics['macro_f1']:.3f}")

    print(f"\n  {'Label':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-' * 42}")
    for label in LABELS:
        m = metrics['per_class'][label]
        print(f"  {label:<10} {m['precision']:>8.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>8}")

    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':>10}", end="")
    for l in LABELS:
        print(f" {l:>8}", end="")
    print()
    for true_label in LABELS:
        print(f"  {true_label:>10}", end="")
        for pred_label in LABELS:
            print(f" {metrics['confusion'][true_label][pred_label]:>8}", end="")
        print()
    print(f"{'=' * 70}")


# ===========================================================================
# Load AllSides Data
# ===========================================================================

def load_allsides_data(filepath):
    """Load and flatten AllSides JSON into individual article entries."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    articles = []
    for story in data:
        main_headline = story.get('main_headline', '')
        for side in story.get('sides', []):
            bias_detail = side.get('bias_detail')
            body = side.get('body', '').strip()

            if not body or not bias_detail or bias_detail not in ALLSIDES_MAP:
                continue

            articles.append({
                'story': main_headline,
                'source': side.get('source', 'Unknown'),
                'headline': side.get('headline', ''),
                'body': body,
                'bias_detail': bias_detail,
                'true_label': ALLSIDES_MAP[bias_detail],
                'original_url': side.get('original_url', ''),
            })

    return articles


# ===========================================================================
# Main Evaluation
# ===========================================================================

def run_evaluation(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                   mode='sequential', with_ig=True):
    """Run full evaluation on all articles."""

    # Select pipeline mode
    pipeline_fn = {
        'sequential': analyze_article_sequential,
        'parallel': analyze_article_parallel,
        'batched': analyze_article_batched,
    }[mode]

    predictions = []
    ground_truths = []
    detailed_results = []
    total_time = 0

    for i, article in enumerate(articles):
        result = pipeline_fn(
            article['body'],
            bias_model, bias_tokenizer,
            pol_model, pol_tokenizer,
            with_ig=with_ig,
        )

        total_time += result['elapsed_seconds']
        pred = result['overall_prediction']

        # Handle case where all sentences filtered (no prediction)
        if pred is None:
            pred = 'Center'  # Default fallback

        predictions.append(pred)
        ground_truths.append(article['true_label'])

        match = "✓" if pred == article['true_label'] else "✗"
        detailed_results.append({
            **article,
            **result,
            'match': match,
        })

        # Print per-article summary
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(articles)}] {match}  {article['source']}")
        print(f"  Headline: {article['headline'][:65]}...")
        print(f"  True: {article['true_label']:>8}  |  Predicted: {pred:>8}  "
              f"({result['analyzed_sentences']}/{result['total_sentences']} sents, "
              f"{result['elapsed_seconds']:.2f}s)")

        if result['overall_probs']:
            probs_str = "  ".join(f"{l}: {p:.3f}" for l, p in result['overall_probs'].items())
            print(f"  Probs: {probs_str}")

        # Show top explainability tokens if available
        if with_ig and result['sentences']:
            all_tokens = []
            for sr in result['sentences']:
                all_tokens.extend(sr.get('top_tokens', []))
            if all_tokens:
                all_tokens.sort(key=lambda x: x['score'], reverse=True)
                unique_seen = set()
                top_unique = []
                for t in all_tokens:
                    if t['token'].lower() not in unique_seen:
                        unique_seen.add(t['token'].lower())
                        top_unique.append(t)
                    if len(top_unique) >= 5:
                        break
                tokens_str = ", ".join(f"{t['token']}({t['score']:.3f})" for t in top_unique)
                print(f"  Key tokens: {tokens_str}")

    return predictions, ground_truths, detailed_results, total_time


def main():
    # Load data
    data_path = Path("allsides_data.json")
    if not data_path.exists():
        # Try uploads directory
        data_path = Path("/mnt/user-data/uploads/allsides_data.json")
    if not data_path.exists():
        print(f"ERROR: Cannot find allsides_data.json")
        return

    articles = load_allsides_data(data_path)
    print(f"Loaded {len(articles)} articles from AllSides")

    # Distribution
    dist = Counter(a['true_label'] for a in articles)
    print(f"Label distribution: {dict(dist)}")

    # Load models
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    # ===================================================================
    # Run 1: Sequential with IG (full explainability)
    # ===================================================================
    print("\n" + "#" * 70)
    print("# EVALUATION: SEQUENTIAL + INTEGRATED GRADIENTS")
    print("#" * 70)

    preds, truths, details, elapsed = run_evaluation(
        articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
        mode='sequential', with_ig=True
    )

    metrics_ig = compute_metrics(preds, truths)
    print_metrics(metrics_ig, "SEQUENTIAL + IG RESULTS")
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/len(articles):.2f}s/article)")

    # ===================================================================
    # Run 2: Batched without IG (speed comparison)
    # ===================================================================
    print("\n\n" + "#" * 70)
    print("# EVALUATION: BATCHED (NO IG — speed baseline)")
    print("#" * 70)

    preds_b, truths_b, details_b, elapsed_b = run_evaluation(
        articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
        mode='batched', with_ig=False
    )

    metrics_batched = compute_metrics(preds_b, truths_b)
    print_metrics(metrics_batched, "BATCHED RESULTS")
    print(f"\n  Total time: {elapsed_b:.1f}s ({elapsed_b/len(articles):.2f}s/article)")

    # ===================================================================
    # Error Analysis
    # ===================================================================
    print("\n\n" + "#" * 70)
    print("# ERROR ANALYSIS (from IG run)")
    print("#" * 70)

    errors = [d for d in details if d['match'] == '✗']
    print(f"\n  {len(errors)} misclassifications out of {len(details)}:")

    for e in errors:
        print(f"\n  Source: {e['source']} ({e['bias_detail']})")
        print(f"  True: {e['true_label']} → Predicted: {e['overall_prediction']}")
        print(f"  Body: {e['body'][:120]}...")
        if e['overall_probs']:
            print(f"  Probs: {e['overall_probs']}")
        print(f"  Sentences analyzed: {e['analyzed_sentences']}/{e['total_sentences']}"
              f" (filtered: {e['filtered_sentences']})")

    # ===================================================================
    # Speed Comparison Summary
    # ===================================================================
    print("\n\n" + "#" * 70)
    print("# SPEED COMPARISON")
    print("#" * 70)
    print(f"\n  Sequential + IG:  {elapsed:.1f}s total, {elapsed/len(articles):.2f}s/article")
    print(f"  Batched (no IG):  {elapsed_b:.1f}s total, {elapsed_b/len(articles):.2f}s/article")
    if elapsed_b > 0:
        print(f"  Speedup:          {elapsed/elapsed_b:.1f}x faster with batching")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Articles tested:     {len(articles)}")
    print(f"  With IG accuracy:    {metrics_ig['accuracy']:.1%} (Macro F1: {metrics_ig['macro_f1']:.3f})")
    print(f"  Batched accuracy:    {metrics_batched['accuracy']:.1%} (Macro F1: {metrics_batched['macro_f1']:.3f})")
    print(f"  Label distribution:  {dict(dist)}")
    print("=" * 70)


if __name__ == "__main__":
    main()