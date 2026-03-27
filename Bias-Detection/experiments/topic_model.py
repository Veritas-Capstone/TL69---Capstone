"""
Topic Selection Bias via PMI + Speed-Optimized Explainability
==============================================================

Part 1: OFFLINE — Build topic model + PMI matrix from training data
  - Run once on your article_bias_prediction dataset
  - Produces: topic_pmi_lookup.json (small file, loads instantly)

Part 2: INFERENCE — Fast pipeline with topic bias + language bias
  - Integrated Gradients for explainability (only on top-k most biased sentences)
  - Topic PMI lookup (embedding + dictionary lookup, ~50ms)
  - Combined bias score

Install: pip install bertopic sentence-transformers hdbscan umap-learn --break-system-packages
"""

import json
import time
import torch
import numpy as np
import spacy
from pathlib import Path
from collections import Counter, defaultdict
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

PUNCTUATION = {
    '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']',
    '-', '–', '—', '▁.', '▁,', '▁!', '▁?', '▁;', '▁:', '▁-'
}


# ===================================================================
# PART 1: OFFLINE — Topic Model + PMI Matrix
# ===================================================================

def build_topic_model(dataset_path="preprocessed/article_bias_prediction.parquet",
                      output_path="topic_pmi_lookup.json",
                      sample_size=10000,
                      n_topics=50):
    """
    Build BERTopic model on your training data and compute PMI matrix.
    
    Run this ONCE offline. Produces a JSON lookup file.
    
    Args:
        dataset_path: Path to your preprocessed parquet file
        output_path: Where to save the PMI lookup
        sample_size: How many articles to use (more = better topics, slower)
        n_topics: Target number of topics (BERTopic will auto-reduce)
    """
    import pandas as pd
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    
    print("=" * 70)
    print("  BUILDING TOPIC MODEL + PMI MATRIX")
    print("=" * 70)
    
    # --- Load data ---
    df = pd.read_parquet(dataset_path)
    print(f"\nLoaded {len(df)} articles")
    print(f"Label distribution: {dict(df['leaning'].value_counts())}")
    
    # Normalize labels
    label_map = {'left': 'Left', 'center': 'Center', 'right': 'Right'}
    df['label'] = df['leaning'].map(label_map)
    df = df.dropna(subset=['label', 'body'])
    
    # Sample for efficiency (BERTopic on 37k articles is slow)
    if len(df) > sample_size:
        # Stratified sample
        df_sample = df.groupby('label').apply(
            lambda x: x.sample(min(len(x), sample_size // 3), random_state=42)
        ).reset_index(drop=True)
    else:
        df_sample = df
    
    print(f"Using {len(df_sample)} articles for topic modeling")
    
    # Truncate bodies for embedding (first 300 words is enough for topic)
    texts = df_sample['body'].apply(lambda x: ' '.join(str(x).split()[:300])).tolist()
    labels = df_sample['label'].tolist()
    
    # --- Build BERTopic ---
    print("\nFitting BERTopic (this may take a few minutes)...")
    
    # Use a small, fast sentence-transformer for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics,  # Reduce to target
        min_topic_size=20,   # Minimum articles per topic
        verbose=True,
    )
    
    topics, probs = topic_model.fit_transform(texts)
    
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info) - 1} topics (excluding outlier topic -1)")
    
    # --- Compute PMI ---
    print("\nComputing PMI matrix...")
    
    total = len(topics)
    label_counts = Counter(labels)
    topic_counts = Counter(topics)
    
    # Joint counts: (topic, label) -> count
    joint_counts = Counter(zip(topics, labels))
    
    pmi_matrix = {}  # topic_id -> {label -> pmi_score}
    topic_descriptions = {}
    
    for topic_id in set(topics):
        if topic_id == -1:
            continue  # Skip outlier topic
        
        pmi_matrix[str(topic_id)] = {}
        
        # Get topic description (top words)
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            topic_descriptions[str(topic_id)] = {
                'words': [w for w, _ in topic_words[:10]],
                'label': ' | '.join([w for w, _ in topic_words[:5]]),
                'count': topic_counts[topic_id],
            }
        
        p_topic = topic_counts[topic_id] / total
        
        for label in LABELS:
            p_label = label_counts[label] / total
            p_joint = joint_counts.get((topic_id, label), 0) / total
            
            if p_joint > 0 and p_topic > 0 and p_label > 0:
                pmi = np.log2(p_joint / (p_topic * p_label))
            else:
                pmi = 0.0
            
            pmi_matrix[str(topic_id)][label] = round(pmi, 4)
    
    # --- Compute topic coverage ratios ---
    # For each topic, what % of articles come from each leaning?
    coverage = {}
    for topic_id in set(topics):
        if topic_id == -1:
            continue
        topic_articles = [(t, l) for t, l in zip(topics, labels) if t == topic_id]
        label_dist = Counter(l for _, l in topic_articles)
        total_topic = len(topic_articles)
        coverage[str(topic_id)] = {
            l: round(label_dist.get(l, 0) / total_topic, 3) for l in LABELS
        }
    
    # --- Find most biased topics ---
    print("\nMost Left-skewed topics:")
    sorted_left = sorted(pmi_matrix.items(), key=lambda x: x[1].get('Left', 0), reverse=True)
    for tid, scores in sorted_left[:5]:
        desc = topic_descriptions.get(tid, {}).get('label', '?')
        print(f"  Topic {tid}: PMI(Left)={scores['Left']:+.3f}  [{desc}]")
    
    print("\nMost Right-skewed topics:")
    sorted_right = sorted(pmi_matrix.items(), key=lambda x: x[1].get('Right', 0), reverse=True)
    for tid, scores in sorted_right[:5]:
        desc = topic_descriptions.get(tid, {}).get('label', '?')
        print(f"  Topic {tid}: PMI(Right)={scores['Right']:+.3f}  [{desc}]")
    
    print("\nMost Center/Neutral topics:")
    sorted_center = sorted(pmi_matrix.items(), key=lambda x: x[1].get('Center', 0), reverse=True)
    for tid, scores in sorted_center[:5]:
        desc = topic_descriptions.get(tid, {}).get('label', '?')
        print(f"  Topic {tid}: PMI(Center)={scores['Center']:+.3f}  [{desc}]")
    
    # --- Save ---
    lookup = {
        'pmi_matrix': pmi_matrix,
        'topic_descriptions': topic_descriptions,
        'coverage': coverage,
        'label_distribution': {l: c for l, c in label_counts.items()},
        'total_articles': total,
        'num_topics': len(pmi_matrix),
    }
    
    with open(output_path, 'w') as f:
        json.dump(lookup, f, indent=2)
    
    print(f"\n✓ Saved PMI lookup to {output_path}")
    print(f"  {len(pmi_matrix)} topics with PMI scores")
    
    # Save the BERTopic model for inference
    model_path = "topic_model"
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True,
                     save_embedding_model=embedding_model)
    print(f"✓ Saved BERTopic model to {model_path}/")
    
    return topic_model, lookup


# ===================================================================
# PART 2: INFERENCE — Topic Bias Lookup
# ===================================================================

class TopicBiasAnalyzer:
    """
    Lightweight inference-time topic analyzer.
    Loads the precomputed PMI lookup and BERTopic model.
    
    Runtime cost: ~50-100ms per article (one embedding + nearest topic lookup)
    """
    
    def __init__(self, lookup_path="topic_pmi_lookup.json", model_path="topic_model"):
        print("Loading topic model...")
        
        with open(lookup_path, 'r') as f:
            self.lookup = json.load(f)
        
        self.pmi_matrix = self.lookup['pmi_matrix']
        self.descriptions = self.lookup['topic_descriptions']
        self.coverage = self.lookup['coverage']
        
        # Load BERTopic model — must pass embedding model back in
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
        
        print(f"✓ Topic model loaded ({self.lookup['num_topics']} topics)")
    
    def analyze(self, text):
        """
        Get topic and PMI-based selection bias for a text.
        
        Returns:
            dict with topic_id, topic_label, pmi_scores, coverage,
            and a selection_bias indicator
        """
        # Truncate for efficiency
        truncated = ' '.join(text.split()[:300])
        
        # Predict topic
        topics, probs = self.topic_model.transform([truncated])
        topic_id = topics[0]
        
        if topic_id == -1:
            # Outlier — no strong topic match
            return {
                'topic_id': -1,
                'topic_label': 'No clear topic',
                'pmi_scores': {'Left': 0, 'Center': 0, 'Right': 0},
                'coverage': {'Left': 0.33, 'Center': 0.33, 'Right': 0.33},
                'selection_bias': None,
                'selection_bias_strength': 0,
            }
        
        tid = str(topic_id)
        pmi = self.pmi_matrix.get(tid, {'Left': 0, 'Center': 0, 'Right': 0})
        cov = self.coverage.get(tid, {'Left': 0.33, 'Center': 0.33, 'Right': 0.33})
        desc = self.descriptions.get(tid, {})
        
        # Determine selection bias direction
        max_pmi_label = max(pmi, key=pmi.get)
        max_pmi_value = pmi[max_pmi_label]
        
        # Only flag selection bias if PMI is meaningfully skewed
        if max_pmi_value > 0.3:
            selection_bias = max_pmi_label
            selection_bias_strength = max_pmi_value
        else:
            selection_bias = None
            selection_bias_strength = 0
        
        return {
            'topic_id': topic_id,
            'topic_label': desc.get('label', f'Topic {topic_id}'),
            'topic_words': desc.get('words', []),
            'pmi_scores': pmi,
            'coverage': cov,
            'selection_bias': selection_bias,
            'selection_bias_strength': selection_bias_strength,
        }


# ===================================================================
# PART 3: EXPLAINABILITY — Integrated Gradients (via captum)
# ===================================================================

def ig_attribution(text, model, tokenizer, target_class=None, n_steps=30):
    """
    Integrated Gradients explainability using captum.
    
    Cost: n_steps forward passes per sentence (default 30).
    Quality: Most faithful attribution method for transformers.
    
    Speed strategy: Only call this on the top-k most biased sentences,
    not every sentence. The pipeline handles this via explain_top_k.
    """
    from captum.attr import LayerIntegratedGradients
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Get embedding layer
    embedding_layer = _get_embedding_layer(model)
    if embedding_layer is None:
        return _predict_only(text, model, tokenizer)
    
    # Forward pass to get prediction
    model.zero_grad()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    predicted_class = torch.argmax(probs).item()
    if target_class is None:
        target_class = predicted_class
    
    # Define forward function — takes input_ids, NOT inputs_embeds.
    # Captum hooks into the embedding layer internally to intercept
    # and interpolate embeddings for the IG computation.
    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    # Baseline: pad token IDs (captum interpolates at the embedding level)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, pad_token_id)
    
    # Compute IG
    lig = LayerIntegratedGradients(forward_func, embedding_layer)
    
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
    )
    
    # Sum across embedding dimension, take absolute value
    attr_scores = attributions.sum(dim=-1).abs()[0]  # (seq_len,)
    
    # Normalize
    if attr_scores.max() > 0:
        normalized = (attr_scores / attr_scores.max()).detach().numpy()
    else:
        normalized = attr_scores.detach().numpy()
    
    top_tokens = _build_filtered_token_list(tokens, normalized)
    
    return {
        'predicted_label': LABELS[predicted_class],
        'confidence': float(probs[predicted_class]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
        'top_tokens': top_tokens[:5],
        'all_token_scores': top_tokens,
        'method': 'integrated_gradients',
    }


def _predict_only(text, model, tokenizer):
    """Fast prediction without explainability."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    return {
        'predicted_label': LABELS[pred],
        'confidence': float(probs[pred]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
        'top_tokens': [],
        'method': 'prediction_only',
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


# ===================================================================
# PART 4: COMBINED PIPELINE
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


def batch_check_politicalness(sentences, model, tokenizer, threshold=0.5):
    """Batch politicalness check — returns per-sentence results."""
    hypothesis = "This text is about politics."
    inputs = tokenizer(
        [(s, hypothesis) for s in sentences],
        return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    
    results = []
    for i, sent in enumerate(sentences):
        conf = float(probs[i, entail_idx])
        results.append({
            'sentence': sent,
            'is_political': conf > threshold,
            'confidence': conf,
        })
    return results


def smart_political_filter(sentences, model, tokenizer, headline=None,
                           primary_threshold=0.5, fallback_threshold=0.3,
                           min_sentences=1):
    """
    Smart filtering with cascading fallbacks:
      1. Try primary threshold (0.5)
      2. If too few pass, relax to fallback threshold (0.3)
      3. If still zero, take top-k by confidence (if best > 0.1)
      4. If all near-zero, check full text + headline as fallback
      5. If genuinely not political, return empty
    
    Returns:
        (political_sentences, political_confidences, filter_metadata)
    """
    if not sentences:
        return [], [], {'strategy': 'empty_input', 'filtered': 0, 'total': 0}
    
    # Step 1: Batch check all sentences
    results = batch_check_politicalness(sentences, model, tokenizer, threshold=primary_threshold)
    
    passed = [(r['sentence'], r['confidence']) for r in results if r['is_political']]
    
    if len(passed) >= min_sentences:
        return (
            [s for s, _ in passed],
            [c for _, c in passed],
            {
                'strategy': 'primary_threshold',
                'threshold_used': primary_threshold,
                'passed': len(passed),
                'total': len(sentences),
                'filtered': len(sentences) - len(passed),
            }
        )
    
    # Step 2: Relax threshold
    passed_relaxed = [(r['sentence'], r['confidence']) for r in results if r['confidence'] > fallback_threshold]
    
    if len(passed_relaxed) >= min_sentences:
        return (
            [s for s, _ in passed_relaxed],
            [c for _, c in passed_relaxed],
            {
                'strategy': 'relaxed_threshold',
                'threshold_used': fallback_threshold,
                'passed': len(passed_relaxed),
                'total': len(sentences),
                'filtered': len(sentences) - len(passed_relaxed),
            }
        )
    
    # Step 3: Force top-k by confidence
    sorted_results = sorted(results, key=lambda r: r['confidence'], reverse=True)
    top_k = sorted_results[:max(min_sentences, 2)]
    
    if top_k[0]['confidence'] > 0.1:
        return (
            [r['sentence'] for r in top_k],
            [r['confidence'] for r in top_k],
            {
                'strategy': 'top_k_forced',
                'threshold_used': None,
                'best_confidence': top_k[0]['confidence'],
                'passed': len(top_k),
                'total': len(sentences),
                'filtered': len(sentences) - len(top_k),
            }
        )
    
    # Step 4: Full text + headline fallback
    full_text = ' '.join(sentences)
    full_results = batch_check_politicalness([full_text], model, tokenizer, threshold=0.0)
    full_conf = full_results[0]['confidence']
    
    headline_conf = 0.0
    if headline:
        headline_results = batch_check_politicalness([headline], model, tokenizer, threshold=0.0)
        headline_conf = headline_results[0]['confidence']
    
    if full_conf > 0.2 or headline_conf > 0.4:
        return (
            sentences,
            [max(full_conf, 0.1)] * len(sentences),
            {
                'strategy': 'full_text_fallback',
                'full_text_confidence': full_conf,
                'headline_confidence': headline_conf,
                'passed': len(sentences),
                'total': len(sentences),
                'filtered': 0,
            }
        )
    
    # Step 5: Genuinely not political
    return (
        [],
        [],
        {
            'strategy': 'genuinely_not_political',
            'best_sentence_conf': sorted_results[0]['confidence'] if sorted_results else 0,
            'full_text_confidence': full_conf,
            'headline_confidence': headline_conf,
            'passed': 0,
            'total': len(sentences),
            'filtered': len(sentences),
        }
    )


def analyze_article_fast(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                         topic_analyzer=None, explain_top_k=2, headline=None):
    """
    Fast combined pipeline:
      1. Smart politicalness filter with fallbacks
      2. Batch bias prediction on political sentences
      3. Gradient × Input ONLY on the top-k most biased sentences
      4. Topic PMI lookup
      5. Combined score
    
    Args:
        explain_top_k: Only compute explainability for this many sentences
                       (the most confident biased ones). Set to 0 for no explainability.
        headline: Article headline (used as fallback signal for politicalness)
    """
    start = time.time()
    
    sentences = split_and_merge(body)
    if not sentences:
        return _empty_result(start)
    
    # --- Step 1: Smart politicalness filter with fallbacks ---
    political_sents, political_confs, filter_meta = smart_political_filter(
        sentences, pol_model, pol_tokenizer,
        headline=headline,
        primary_threshold=0.5,
        fallback_threshold=0.3,
        min_sentences=1,
    )
    
    filtered = filter_meta['filtered']
    
    if not political_sents:
        return _empty_result(start, len(sentences), filtered)
    
    # --- Step 2: Batch bias prediction ---
    bias_inputs = bias_tokenizer(
        political_sents,
        return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        bias_outputs = bias_model(**bias_inputs)
        bias_probs = torch.nn.functional.softmax(bias_outputs.logits, dim=-1)
    
    sentence_results = []
    for i, (sent, pol_conf) in enumerate(zip(political_sents, political_confs)):
        probs = bias_probs[i]
        pred_class = torch.argmax(probs).item()
        all_probs = {l: float(probs[j]) for j, l in enumerate(LABELS)}
        
        # Compute weight
        weight = 1.0
        peak = float(probs[pred_class])
        if peak < 0.45:
            weight *= 0.3
        # Downweight sentences that only passed via fallback
        if pol_conf < 0.5:
            weight *= pol_conf
        if sent.strip().startswith('"') or sent.strip().startswith('\u201c'):
            weight *= 0.5
        if len(sent.split()) < 10:
            weight *= 0.5
        
        sentence_results.append({
            'text': sent[:100],
            'full_text': sent,
            'political_conf': pol_conf,
            'predicted_label': LABELS[pred_class],
            'confidence': peak,
            'all_probs': all_probs,
            'weight': weight,
            'top_tokens': [],
        })
    
    # --- Step 3: IG on top-k most biased sentences ---
    if explain_top_k > 0:
        # Sort by bias strength (distance from center)
        scored = []
        for j, sr in enumerate(sentence_results):
            center_prob = sr['all_probs']['Center']
            bias_strength = 1.0 - center_prob  # Higher = more biased
            scored.append((bias_strength, j))
        scored.sort(reverse=True)
        
        for _, idx in scored[:explain_top_k]:
            sr = sentence_results[idx]
            attr = ig_attribution(
                sr['full_text'], bias_model, bias_tokenizer
            )
            sr['top_tokens'] = attr['top_tokens']
            sr['method'] = 'integrated_gradients'
    
    # --- Step 4: Aggregate language bias ---
    weighted_probs = []
    for sr in sentence_results:
        probs_t = torch.tensor([sr['all_probs'][l] for l in LABELS])
        weighted_probs.append(probs_t * sr['weight'])
    
    total = torch.stack(weighted_probs).sum(dim=0)
    if total.sum() > 0:
        normalized = total / total.sum()
    else:
        normalized = total
    
    language_label = LABELS[torch.argmax(total).item()]
    language_probs = {l: float(normalized[i]) for i, l in enumerate(LABELS)}
    
    # --- Step 5: Topic bias (if available) ---
    # Topic PMI is INFORMATIONAL ONLY — shown to users as context
    # about agenda-setting bias, NOT blended into the prediction.
    # Reason: PMI tells you which sources tend to cover a topic,
    # not how THIS article is biased. A left source writing about
    # a right-heavy topic doesn't make the article right-leaning.
    topic_result = None
    combined_label = language_label
    combined_probs = language_probs.copy()
    
    if topic_analyzer:
        topic_result = topic_analyzer.analyze(body)
        # No blending — language prediction stands on its own
        # Topic info is passed through for display to the user
    
    elapsed = time.time() - start
    
    return {
        'language_prediction': language_label,
        'language_probs': language_probs,
        'topic_result': topic_result,
        'combined_prediction': combined_label,
        'combined_probs': combined_probs,
        'sentences': sentence_results,
        'total_sentences': len(sentences),
        'filtered_sentences': filtered,
        'analyzed_sentences': len(sentence_results),
        'filter_strategy': filter_meta['strategy'],
        'elapsed_seconds': elapsed,
    }


def _empty_result(start, total=0, filtered=0):
    return {
        'language_prediction': 'Center',
        'language_probs': {'Left': 0, 'Center': 1, 'Right': 0},
        'topic_result': None,
        'combined_prediction': 'Center',
        'combined_probs': {'Left': 0, 'Center': 1, 'Right': 0},
        'sentences': [],
        'total_sentences': total,
        'filtered_sentences': filtered,
        'analyzed_sentences': 0,
        'filter_strategy': 'genuinely_not_political',
        'elapsed_seconds': time.time() - start,
    }


# ===================================================================
# PART 5: EVALUATION
# ===================================================================

def load_allsides_data(filepath):
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


def run_eval(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
             topic_analyzer=None, explain_top_k=2):
    """Run evaluation with the fast pipeline."""
    
    preds_lang = []
    preds_combined = []
    truths = []
    total_time = 0
    
    for i, article in enumerate(articles):
        result = analyze_article_fast(
            article['body'],
            bias_model, bias_tokenizer,
            pol_model, pol_tokenizer,
            topic_analyzer=topic_analyzer,
            explain_top_k=explain_top_k,
            headline=article.get('headline', ''),
        )
        
        total_time += result['elapsed_seconds']
        
        lang_pred = result['language_prediction']
        comb_pred = result['combined_prediction']
        true = article['true_label']
        
        preds_lang.append(lang_pred)
        preds_combined.append(comb_pred)
        truths.append(true)
        
        match_l = "✓" if lang_pred == true else "✗"
        match_c = "✓" if comb_pred == true else "✗"
        
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(articles)}]  {article['source']}")
        print(f"  True: {true:>8}  |  Language: {lang_pred:>8} {match_l}"
              f"  |  Combined: {comb_pred:>8} {match_c}")
        print(f"  ({result['analyzed_sentences']}/{result['total_sentences']} sents, "
              f"{result['elapsed_seconds']:.2f}s, filter: {result.get('filter_strategy', 'n/a')})")
        
        # Show topic info
        if result['topic_result'] and result['topic_result']['topic_id'] != -1:
            tr = result['topic_result']
            print(f"  Topic: {tr['topic_label']}")
            print(f"  Topic PMI: L={tr['pmi_scores'].get('Left',0):+.3f} "
                  f"C={tr['pmi_scores'].get('Center',0):+.3f} "
                  f"R={tr['pmi_scores'].get('Right',0):+.3f}")
            if tr['selection_bias']:
                print(f"  ⚠ Selection bias toward {tr['selection_bias']} "
                      f"(strength: {tr['selection_bias_strength']:.3f})")
        
        # Show top explainability tokens
        all_tokens = []
        for sr in result['sentences']:
            all_tokens.extend(sr.get('top_tokens', []))
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
    
    print(f"\n  Total time: {total_time:.1f}s ({total_time/len(articles):.2f}s/article)")
    
    metrics_lang = compute_metrics(preds_lang, truths)
    print_metrics(metrics_lang, "LANGUAGE-ONLY RESULTS")
    
    if topic_analyzer:
        metrics_comb = compute_metrics(preds_combined, truths)
        print_metrics(metrics_comb, "COMBINED (LANGUAGE + TOPIC) RESULTS")
    
    return metrics_lang


# ===================================================================
# MODEL LOADING
# ===================================================================

def load_models(bias_path="models/bias_detector", pol_path="models/politicalness_filter"):
    """Load both models and tokenizers."""
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
# MAIN
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-topics', action='store_true',
                        help='Build topic model from training data (run once)')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation on AllSides data')
    parser.add_argument('--dataset', default='preprocessed/article_bias_prediction.parquet')
    parser.add_argument('--allsides', default='allsides_data.json')
    parser.add_argument('--explain-top-k', type=int, default=2,
                        help='Number of sentences to explain (0=none)')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Articles to use for topic modeling')
    args = parser.parse_args()
    
    if args.build_topics:
        build_topic_model(
            dataset_path=args.dataset,
            sample_size=args.sample_size,
        )
        print("\n✓ Topic model built. Now run with --eval")
        return
    
    if args.eval:
        # Load AllSides data
        articles = load_allsides_data(args.allsides)
        print(f"Loaded {len(articles)} articles")
        print(f"Distribution: {dict(Counter(a['true_label'] for a in articles))}")
        
        # Load bias models
        bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()
        
        # Try to load topic model
        topic_analyzer = None
        if Path("topic_pmi_lookup.json").exists() and Path("topic_model").exists():
            topic_analyzer = TopicBiasAnalyzer()
        else:
            print("\n⚠ No topic model found. Run --build-topics first for combined analysis.")
            print("  Running language-only evaluation...\n")
        
        print("\n" + "#" * 70)
        print("# FAST PIPELINE EVALUATION (IG on top-k + Topic PMI)")
        print("#" * 70)
        
        run_eval(
            articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
            topic_analyzer=topic_analyzer,
            explain_top_k=args.explain_top_k,
        )
        return
    
    # Default: show usage
    print("Usage:")
    print("  Step 1 (once): python topic_bias_pipeline.py --build-topics")
    print("  Step 2:        python topic_bias_pipeline.py --eval")
    print("\nOptions:")
    print("  --explain-top-k N   Explain top N sentences (default: 2, 0=fast)")
    print("  --sample-size N     Articles for topic modeling (default: 10000)")
    print("  --allsides PATH     AllSides JSON file path")


if __name__ == "__main__":
    main()