"""
Hybrid Pipeline: Full-Article Prediction + Sentence-Level Explainability
=========================================================================

Key insight: The Volf model was trained on FULL ARTICLES, not sentences.
Splitting into sentences and aggregating loses context and hurts accuracy.

Architecture:
  1. Politicalness check on full article text (not per-sentence)
  2. Bias prediction on full article text (single forward pass)
  3. Sentence-level IG ONLY for explainability (on top-k most biased sentences)
  4. Topic PMI as informational context

This separates two concerns:
  - PREDICTION: Use the model how it was trained (full article)
  - EXPLANATION: Use sentence-level IG to show users which parts matter
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


# ===================================================================
# MODEL LOADING
# ===================================================================

def load_models(bias_path="models/bias_detector", pol_path="models/politicalness_filter"):
    print("Loading models...")
    bias_model = AutoModelForSequenceClassification.from_pretrained(bias_path)
    bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
    bias_model.eval()

    pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
    pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
    pol_model.eval()

    print("✓ Models loaded")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


# ===================================================================
# POLITICALNESS CHECK (full text)
# ===================================================================

def check_political(text, model, tokenizer, threshold=0.5):
    """Check if text is political using NLI model."""
    hypothesis = "This text is about politics."
    inputs = tokenizer(text, hypothesis, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    conf = float(probs[entail_idx])
    return conf > threshold, conf


# ===================================================================
# BIAS PREDICTION (full article)
# ===================================================================

def predict_bias_full_article(text, model, tokenizer, max_length=512):
    """
    Predict bias on the full article text.
    
    The model was trained on full articles — this is how it should be used.
    Truncation to max_length tokens keeps the beginning of the article,
    which typically contains the lede and framing that signal bias.
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=max_length, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    pred_class = torch.argmax(probs).item()
    return {
        'predicted_label': LABELS[pred_class],
        'confidence': float(probs[pred_class]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
    }


# ===================================================================
# SENTENCE-LEVEL ANALYSIS (for explainability only)
# ===================================================================

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
    """Quick batch prediction to rank sentences by bias strength."""
    if not sentences:
        return []
    
    inputs = tokenizer(sentences, return_tensors="pt",
                       truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    results = []
    for i, sent in enumerate(sentences):
        p = probs[i]
        pred_class = torch.argmax(p).item()
        center_prob = float(p[LABELS.index('Center')])
        bias_strength = 1.0 - center_prob  # Higher = more biased
        results.append({
            'text': sent,
            'predicted_label': LABELS[pred_class],
            'confidence': float(p[pred_class]),
            'all_probs': {l: float(p[j]) for j, l in enumerate(LABELS)},
            'bias_strength': bias_strength,
        })
    
    return results


# ===================================================================
# INTEGRATED GRADIENTS (on selected sentences)
# ===================================================================

def ig_attribution(text, model, tokenizer, target_class=None, n_steps=30):
    """IG explainability on a single sentence."""
    from captum.attr import LayerIntegratedGradients
    
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    embedding_layer = _get_embedding_layer(model)
    if embedding_layer is None:
        return {'top_tokens': [], 'method': 'none'}
    
    model.zero_grad()
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
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
    )
    
    attr_scores = attributions.sum(dim=-1).abs()[0]
    if attr_scores.max() > 0:
        normalized = (attr_scores / attr_scores.max()).detach().numpy()
    else:
        normalized = attr_scores.detach().numpy()
    
    top_tokens = _build_filtered_token_list(tokens, normalized)
    
    return {
        'top_tokens': top_tokens[:5],
        'all_token_scores': top_tokens,
        'method': 'integrated_gradients',
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
        if not clean:
            continue
        if all(c in '.,!?;:\'"()[]<>-–—' for c in clean):
            continue
        if clean.lower() in STOP_WORDS:
            continue
        result.append({'token': clean, 'score': float(score), 'position': i})
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


# ===================================================================
# TOPIC BIAS (informational only)
# ===================================================================

class TopicBiasAnalyzer:
    def __init__(self, lookup_path="topic_pmi_lookup.json", model_path="topic_model"):
        print("Loading topic model...")
        with open(lookup_path, 'r') as f:
            self.lookup = json.load(f)
        self.pmi_matrix = self.lookup['pmi_matrix']
        self.descriptions = self.lookup['topic_descriptions']
        self.coverage = self.lookup['coverage']
        
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
        print(f"✓ Topic model loaded ({self.lookup['num_topics']} topics)")
    
    def analyze(self, text):
        truncated = ' '.join(text.split()[:300])
        topics, probs = self.topic_model.transform([truncated])
        topic_id = topics[0]
        
        if topic_id == -1:
            return {
                'topic_id': -1, 'topic_label': 'No clear topic',
                'pmi_scores': {'Left': 0, 'Center': 0, 'Right': 0},
                'coverage': {'Left': 0.33, 'Center': 0.33, 'Right': 0.33},
                'selection_bias': None, 'selection_bias_strength': 0,
            }
        
        tid = str(topic_id)
        pmi = self.pmi_matrix.get(tid, {'Left': 0, 'Center': 0, 'Right': 0})
        cov = self.coverage.get(tid, {'Left': 0.33, 'Center': 0.33, 'Right': 0.33})
        desc = self.descriptions.get(tid, {})
        
        max_pmi_label = max(pmi, key=pmi.get)
        max_pmi_value = pmi[max_pmi_label]
        
        return {
            'topic_id': topic_id,
            'topic_label': desc.get('label', f'Topic {topic_id}'),
            'topic_words': desc.get('words', []),
            'pmi_scores': pmi,
            'coverage': cov,
            'selection_bias': max_pmi_label if max_pmi_value > 0.3 else None,
            'selection_bias_strength': max_pmi_value if max_pmi_value > 0.3 else 0,
        }


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def analyze_article(body, headline, bias_model, bias_tokenizer,
                    pol_model, pol_tokenizer,
                    topic_analyzer=None, explain_top_k=2):
    """
    Hybrid pipeline:
      1. Full-article politicalness check
      2. Full-article bias prediction (how the model was trained)
      3. Sentence-level ranking → IG on top-k for explainability
      4. Topic PMI as context
    """
    start = time.time()
    
    # --- Step 1: Politicalness on full text ---
    is_political, pol_conf = check_political(body, pol_model, pol_tokenizer, threshold=0.3)
    
    # Also check headline as backup
    if not is_political and headline:
        is_political_hl, pol_conf_hl = check_political(headline, pol_model, pol_tokenizer, threshold=0.3)
        if is_political_hl:
            is_political = True
            pol_conf = pol_conf_hl
    
    if not is_political:
        return {
            'prediction': 'Center',
            'confidence': 0.0,
            'all_probs': {'Left': 0, 'Center': 1, 'Right': 0},
            'political_confidence': pol_conf,
            'is_political': False,
            'sentence_explanations': [],
            'topic_result': None,
            'elapsed_seconds': time.time() - start,
        }
    
    # --- Step 2: Full-article bias prediction ---
    bias_result = predict_bias_full_article(body, bias_model, bias_tokenizer)
    
    # --- Step 3: Sentence-level explainability ---
    sentence_explanations = []
    if explain_top_k > 0:
        sentences = split_and_merge(body)
        if sentences:
            # Rank sentences by bias strength (quick batch prediction)
            sent_scores = get_sentence_bias_scores(sentences, bias_model, bias_tokenizer)
            sent_scores.sort(key=lambda x: x['bias_strength'], reverse=True)
            
            # IG on the top-k most biased sentences
            for sr in sent_scores[:explain_top_k]:
                ig_result = ig_attribution(sr['text'], bias_model, bias_tokenizer)
                sentence_explanations.append({
                    'text': sr['text'][:150],
                    'predicted_label': sr['predicted_label'],
                    'confidence': sr['confidence'],
                    'bias_strength': sr['bias_strength'],
                    'top_tokens': ig_result['top_tokens'],
                })
    
    # --- Step 4: Topic PMI (informational) ---
    topic_result = None
    if topic_analyzer:
        topic_result = topic_analyzer.analyze(body)
    
    elapsed = time.time() - start
    
    return {
        'prediction': bias_result['predicted_label'],
        'confidence': bias_result['confidence'],
        'all_probs': bias_result['all_probs'],
        'political_confidence': pol_conf,
        'is_political': True,
        'sentence_explanations': sentence_explanations,
        'topic_result': topic_result,
        'elapsed_seconds': elapsed,
    }


# ===================================================================
# EVALUATION
# ===================================================================

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
    
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'correct': correct,
            'total': total, 'per_class': class_metrics, 'confusion': confusion}


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
    for tl in LABELS:
        print(f"  {tl:>10}", end="")
        for pl in LABELS:
            print(f" {metrics['confusion'][tl][pl]:>8}", end="")
        print()
    print(f"{'=' * 70}")


def run_comparison(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                   topic_analyzer=None, explain_top_k=2):
    """
    Run evaluation comparing:
      A) Full-article prediction (new)
      B) Sentence-aggregated prediction (old, for comparison)
    """
    preds_full = []
    preds_sent = []
    truths = []
    total_time = 0
    
    for i, article in enumerate(articles):
        body = article['body']
        headline = article['headline']
        true = article['true_label']
        truths.append(true)
        
        start = time.time()
        
        # --- A) Full-article prediction ---
        result = analyze_article(
            body, headline,
            bias_model, bias_tokenizer,
            pol_model, pol_tokenizer,
            topic_analyzer=topic_analyzer,
            explain_top_k=explain_top_k,
        )
        full_pred = result['prediction']
        preds_full.append(full_pred)
        
        # --- B) Sentence-aggregated prediction (for comparison) ---
        sentences = split_and_merge(body)
        sent_scores = get_sentence_bias_scores(sentences, bias_model, bias_tokenizer)
        if sent_scores:
            # Weighted aggregation
            weighted = torch.zeros(3)
            for sr in sent_scores:
                probs_t = torch.tensor([sr['all_probs'][l] for l in LABELS])
                w = 1.0
                if sr['confidence'] < 0.45:
                    w *= 0.3
                weighted += probs_t * w
            if weighted.sum() > 0:
                weighted = weighted / weighted.sum()
            sent_pred = LABELS[torch.argmax(weighted).item()]
        else:
            sent_pred = 'Center'
        preds_sent.append(sent_pred)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        match_f = "✓" if full_pred == true else "✗"
        match_s = "✓" if sent_pred == true else "✗"
        
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(articles)}]  {article['source']}")
        print(f"  True: {true:>8}  |  Full-article: {full_pred:>8} {match_f}"
              f"  |  Sentence-agg: {sent_pred:>8} {match_s}")
        
        # Show full-article probs
        if result['is_political']:
            p = result['all_probs']
            print(f"  Full probs: L={p['Left']:.3f}  C={p['Center']:.3f}  R={p['Right']:.3f}"
                  f"  (conf: {result['confidence']:.3f})")
        else:
            print(f"  Not political (conf: {result['political_confidence']:.3f})")
        
        # Show topic
        if result['topic_result'] and result['topic_result']['topic_id'] != -1:
            tr = result['topic_result']
            print(f"  Topic: {tr['topic_label']}")
            if tr['selection_bias']:
                print(f"  ⚠ Selection bias toward {tr['selection_bias']} "
                        f"(strength: {tr['selection_bias_strength']:.3f})")
        
        # Show top IG tokens
        if result['sentence_explanations']:
            all_tokens = []
            for se in result['sentence_explanations']:
                all_tokens.extend(se.get('top_tokens', []))
            if all_tokens:
                all_tokens.sort(key=lambda x: x['score'], reverse=True)
                seen = set()
                unique = []
                for t in all_tokens:
                    if t['token'].lower() not in seen:
                        seen.add(t['token'].lower())
                        unique.append(t)
                    if len(unique) >= 5:
                        break
                token_strs = [f"{t['token']}({t['score']:.3f})" for t in unique]
                print(f"  Key tokens: {', '.join(token_strs)}")
        
        print(f"  ({elapsed:.1f}s)")
    
    print(f"\n  Total time: {total_time:.1f}s ({total_time/len(articles):.1f}s/article)")
    
    # --- Metrics ---
    metrics_full = compute_metrics(preds_full, truths)
    print_metrics(metrics_full, "FULL-ARTICLE PREDICTION")
    
    metrics_sent = compute_metrics(preds_sent, truths)
    print_metrics(metrics_sent, "SENTENCE-AGGREGATED PREDICTION (comparison)")
    
    # --- Head-to-head ---
    print(f"\n{'=' * 70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 70}")
    full_only = sum(1 for f, s, t in zip(preds_full, preds_sent, truths) if f == t and s != t)
    sent_only = sum(1 for f, s, t in zip(preds_full, preds_sent, truths) if f != t and s == t)
    both_right = sum(1 for f, s, t in zip(preds_full, preds_sent, truths) if f == t and s == t)
    both_wrong = sum(1 for f, s, t in zip(preds_full, preds_sent, truths) if f != t and s != t)
    print(f"  Both correct:         {both_right}")
    print(f"  Full-article only:    {full_only}")
    print(f"  Sentence-agg only:    {sent_only}")
    print(f"  Both wrong:           {both_wrong}")
    print(f"{'=' * 70}")
    
    return metrics_full, metrics_sent


# ===================================================================
# MAIN
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--allsides', default='allsides_data.json')
    parser.add_argument('--explain-top-k', type=int, default=2,
                        help='Sentences to explain with IG (0=skip)')
    args = parser.parse_args()
    
    articles = load_allsides_data(args.allsides)
    print(f"Loaded {len(articles)} articles")
    print(f"Distribution: {dict(Counter(a['true_label'] for a in articles))}")
    
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()
    
    topic_analyzer = None
    if Path("topic_pmi_lookup.json").exists() and Path("topic_model").exists():
        topic_analyzer = TopicBiasAnalyzer()
    else:
        print("\n⚠ No topic model found. Skipping topic analysis.")
    
    print("\n" + "#" * 70)
    print("# HYBRID PIPELINE: FULL-ARTICLE vs SENTENCE-AGGREGATED")
    print("#" * 70)
    
    run_comparison(
        articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
        topic_analyzer=topic_analyzer,
        explain_top_k=args.explain_top_k,
    )


if __name__ == "__main__":
    main()