"""
Explainability for Bias Detection using Integrated Gradients
============================================================
Replaces raw attention weights with gradient-based attribution
for more meaningful token importance scores.

Uses captum's LayerIntegratedGradients to measure how much each
token embedding contributes to the model's bias prediction.

Install: pip install captum --break-system-packages


This works excellently as is, the model however learns patterns in certain leaning text. Which results in the model returning things like "The, an" etc.
On the frontend we can just filter these out instead
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'shall',
                'can', 'that', 'this', 'these', 'those', 'it', 'its', 'not', 'no'}
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import spacy

# Conditionally import captum — falls back to occlusion method if unavailable
try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("⚠️  captum not installed. Using occlusion-based fallback.")
    print("   Install with: pip install captum")

nlp = spacy.load("en_core_web_sm")


# ===========================================================================
# Model Loading
# ===========================================================================

def load_models():
    """Load both models"""
    print("Loading models...")

    bias_model = AutoModelForSequenceClassification.from_pretrained(
        "models/bias_detector"
    )
    bias_tokenizer = AutoTokenizer.from_pretrained("models/bias_detector")

    pol_model = AutoModelForSequenceClassification.from_pretrained(
        "models/politicalness_filter"
    )
    pol_tokenizer = AutoTokenizer.from_pretrained("models/politicalness_filter")

    # Keep models in eval mode
    bias_model.eval()
    pol_model.eval()

    print("✓ Models loaded\n")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


# ===========================================================================
# Politicalness Filter (NLI-based — corrected)
# ===========================================================================

def check_politicalness(text, model, tokenizer):
    """
    Check if text is political using Political DEBATE (NLI model).
    Requires premise + hypothesis pair.
    """
    hypothesis = "This text is about politics."
    inputs = tokenizer(text, hypothesis, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Get the entailment index from the model's config
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    political_prob = float(probs[entail_idx])

    return {
        'is_political': political_prob > 0.5,
        'confidence': political_prob
    }


# ===========================================================================
# Sentence Splitting (from your existing code)
# ===========================================================================

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def merge_short_sentences(sentences, min_tokens=30, max_tokens=150):
    """
    Merge short sentences for better model context.
    Added max_tokens cap to prevent overly long merged chunks.
    """
    merged = []
    buffer = ""

    for sent in sentences:
        num_tokens = len(sent.split())

        if num_tokens < min_tokens:
            buffer += " " + sent
            # Flush if buffer is getting too long
            if len(buffer.split()) >= max_tokens:
                merged.append(buffer.strip())
                buffer = ""
        else:
            if buffer.strip():
                merged.append(buffer.strip())
                buffer = ""
            merged.append(sent)

    if buffer.strip():
        merged.append(buffer.strip())

    return merged


# ===========================================================================
# Integrated Gradients Attribution (Primary Method)
# ===========================================================================

def _forward_func(input_embeds, attention_mask, model):
    """
    Forward function for captum that takes embeddings as input.
    This lets IG compute gradients w.r.t. the embedding layer.
    """
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    return outputs.logits


def compute_ig_attributions(text, model, tokenizer, target_class=None, n_steps=50):
    """
    Compute Integrated Gradients attributions for each token.

    Args:
        text: Input text string
        model: The bias detection model
        tokenizer: The model's tokenizer
        target_class: Which class to explain (0=Left, 1=Center, 2=Right).
                      If None, uses the predicted class.
        n_steps: Number of interpolation steps for IG (higher = more precise)

    Returns:
        dict with prediction info and token attributions
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get prediction first
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    labels = ['Left', 'Center', 'Right']
    predicted_class = torch.argmax(probs).item()
    predicted_label = labels[predicted_class]

    if target_class is None:
        target_class = predicted_class

    # --- Integrated Gradients ---
    # We need to attribute w.r.t. the embedding layer
    embedding_layer = _get_embedding_layer(model)

    if embedding_layer is None:
        print("⚠️  Could not find embedding layer. Falling back to occlusion.")
        return compute_occlusion_attributions(text, model, tokenizer, target_class)

    lig = LayerIntegratedGradients(
        lambda input_ids, attention_mask: model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits,
        embedding_layer
    )

    # Baseline: pad token embeddings (represents "absence of information")
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)

    # Compute attributions
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=False
    )

    # Sum across embedding dimension to get per-token attribution
    # Shape: (1, seq_len, hidden_dim) -> (seq_len,)
    token_attributions = attributions.sum(dim=-1).squeeze(0)

    # Normalize to [0, 1] range using absolute values
    # (we care about magnitude of contribution, not direction)
    abs_attr = token_attributions.abs()
    if abs_attr.max() > 0:
        normalized = (abs_attr / abs_attr.max()).detach().numpy()
    else:
        normalized = abs_attr.detach().numpy()

    # Build token attribution list (filter special tokens + punctuation)
    token_attention = _build_token_list(tokens, normalized)

    return {
        'predicted_label': predicted_label,
        'confidence': float(probs[predicted_class]),
        'all_probs': {
            'Left': float(probs[0]),
            'Center': float(probs[1]),
            'Right': float(probs[2])
        },
        'target_class': labels[target_class],
        'token_attention': token_attention,
        'all_tokens': tokens,
        'method': 'integrated_gradients'
    }


def _get_embedding_layer(model):
    """
    Find the word embedding layer for different model architectures.
    Supports DeBERTa, BERT, RoBERTa, DistilBERT.
    """
    # DeBERTa / DeBERTa-v2 / DeBERTa-v3
    if hasattr(model, 'deberta'):
        return model.deberta.embeddings.word_embeddings
    # BERT
    if hasattr(model, 'bert'):
        return model.bert.embeddings.word_embeddings
    # RoBERTa
    if hasattr(model, 'roberta'):
        return model.roberta.embeddings.word_embeddings
    # DistilBERT
    if hasattr(model, 'distilbert'):
        return model.distilbert.embeddings.word_embeddings

    # Generic fallback: try to find it by walking the model
    for name, module in model.named_modules():
        if 'word_embedding' in name.lower() and hasattr(module, 'weight'):
            return module

    return None


# ===========================================================================
# Occlusion-based Attribution (Fallback if captum unavailable)
# ===========================================================================

def compute_occlusion_attributions(text, model, tokenizer, target_class=None):
    """
    Leave-one-out attribution: measure prediction change when each token
    is replaced with [PAD]. Slower but requires no extra dependencies.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get baseline prediction
    with torch.no_grad():
        outputs = model(**inputs)
        base_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    labels = ['Left', 'Center', 'Right']
    predicted_class = torch.argmax(base_probs).item()
    predicted_label = labels[predicted_class]

    if target_class is None:
        target_class = predicted_class

    base_score = float(base_probs[target_class])
    pad_id = tokenizer.pad_token_id or 0

    # For each token, replace with PAD and measure prediction drop
    importance_scores = np.zeros(len(tokens))

    for i in range(len(tokens)):
        if tokens[i] in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue

        # Create modified input with token i replaced
        modified_ids = input_ids.clone()
        modified_ids[0, i] = pad_id

        with torch.no_grad():
            mod_outputs = model(input_ids=modified_ids, attention_mask=attention_mask)
            mod_probs = torch.nn.functional.softmax(mod_outputs.logits, dim=-1)[0]

        # Importance = how much the target class probability drops
        importance_scores[i] = max(0, base_score - float(mod_probs[target_class]))

    # Normalize
    if importance_scores.max() > 0:
        normalized = importance_scores / importance_scores.max()
    else:
        normalized = importance_scores

    token_attention = _build_token_list(tokens, normalized)

    return {
        'predicted_label': predicted_label,
        'confidence': float(base_probs[predicted_class]),
        'all_probs': {
            'Left': float(base_probs[0]),
            'Center': float(base_probs[1]),
            'Right': float(base_probs[2])
        },
        'target_class': labels[target_class],
        'token_attention': token_attention,
        'all_tokens': tokens,
        'method': 'occlusion'
    }


# ===========================================================================
# Shared Utilities
# ===========================================================================

PUNCTUATION = {
    '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']',
    '-', '–', '—', '▁.', '▁,', '▁!', '▁?', '▁;', '▁:', '▁-'
}


def _build_token_list(tokens, scores):
    """
    Build sorted list of (token, score) pairs, filtering out
    special tokens and punctuation.
    """
    token_list = []

    for i, (token, score) in enumerate(zip(tokens, scores)):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue

        clean = token.replace('##', '').replace('▁', '').strip()
        if clean in PUNCTUATION or clean == '':
            continue
        if all(c in '.,!?;:\'"()[]<>-–—' for c in clean):
            continue

        token_list.append({
            'token': token.replace('##', ''),
            'clean_token': clean,
            'score': float(score),
            'position': i
        })

    token_list.sort(key=lambda x: x['score'], reverse=True)
    return token_list


def detect_bias(text, model, tokenizer, method='ig'):
    """
    Unified interface: detect bias with chosen explainability method.

    Args:
        method: 'ig' for Integrated Gradients, 'occlusion' for leave-one-out
    """
    if method == 'ig' and HAS_CAPTUM:
        return compute_ig_attributions(text, model, tokenizer)
    else:
        return compute_occlusion_attributions(text, model, tokenizer)


# ===========================================================================
# Visualization
# ===========================================================================

def visualize(text, result, top_k=10):
    """Print attribution visualization"""
    print("=" * 70)
    print(f"TEXT: {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"METHOD: {result['method']}")
    print("=" * 70)

    print(f"\nPrediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.3f}")

    print(f"\nAll Probabilities:")
    for label, prob in result['all_probs'].items():
        bar = "█" * int(prob * 40)
        print(f"  {label:7s} {prob:.3f} {bar}")

    print(f"\n{'=' * 70}")
    print(f"TOP {top_k} MOST IMPORTANT TOKENS")
    print(f"{'=' * 70}")
    print(f"{'Rank':<6} {'Token':<20} {'Attribution':<15} {'Visual'}")
    print("-" * 70)

    for i, item in enumerate(result['token_attention'][:top_k], 1):
        score = item['score']
        max_score = result['token_attention'][0]['score'] if result['token_attention'] else 1
        bar_len = int((score / max_score) * 30) if max_score > 0 else 0
        bar = "█" * bar_len

        print(f"{i:<6} {item['token']:<20} {score:<15.4f} {bar}")

    print("=" * 70)


def highlight_text(text, result, threshold=0.3):
    """
    Highlight important words in the original text.
    Threshold is relative to the max score (0.0 to 1.0).
    """
    tokens = result['all_tokens']
    token_dict = {item['position']: item['score'] for item in result['token_attention']}

    highlighted = []
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue

        clean = token.replace('##', '').replace('▁', ' ').strip()
        if not clean:
            continue

        score = token_dict.get(i, 0)
        if score >= threshold:
            highlighted.append(f"**{clean}**")
        else:
            highlighted.append(clean)

    return ' '.join(highlighted)


# ===========================================================================
# Sentence-Level Analysis with Aggregation
# ===========================================================================

def analyze_article(text, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                    method='ig', min_merge_tokens=30, low_conf_threshold=0.45):
    """
    Full pipeline: politicalness check → sentence split → per-sentence
    bias detection with explainability → weighted aggregation.
    """
    # Step 1: Politicalness check on full text (or first 512 tokens)
    pol_result = check_politicalness(text[:2000], pol_model, pol_tokenizer)
    if not pol_result['is_political']:
        return {
            'is_political': False,
            'political_confidence': pol_result['confidence'],
            'sentences': [],
            'overall_prediction': None
        }

    # Step 2: Split into sentences
    sentences = split_into_sentences(text)
    sentences = merge_short_sentences(sentences, min_tokens=min_merge_tokens)

    # Step 3: Per-sentence analysis
    sentence_results = []
    weighted_probs = []

    for sent in sentences:
        result = detect_bias(sent, bias_model, bias_tokenizer, method=method)

        # Compute weight (from your existing logic)
        weight = 1.0
        peak_conf = result['confidence']

        if peak_conf < low_conf_threshold:
            weight *= 0.3

        # Downweight quoted text
        if sent.strip().startswith('"') or sent.strip().startswith('\u201c'):
            weight *= 0.5

        probs_tensor = torch.tensor([
            result['all_probs']['Left'],
            result['all_probs']['Center'],
            result['all_probs']['Right']
        ])

        weighted_probs.append(probs_tensor * weight)

        sentence_results.append({
            'text': sent,
            'prediction': result['predicted_label'],
            'confidence': result['confidence'],
            'all_probs': result['all_probs'],
            'weight': weight,
            'top_tokens': result['token_attention'][:5],
            'method': result['method']
        })

    # Step 4: Aggregate
    if weighted_probs:
        total = torch.stack(weighted_probs).sum(dim=0)
        labels = ['Left', 'Center', 'Right']
        overall_idx = torch.argmax(total).item()
        overall_prediction = labels[overall_idx]
        overall_probs = {l: float(total[i]) for i, l in enumerate(labels)}
    else:
        overall_prediction = None
        overall_probs = {}

    return {
        'is_political': True,
        'political_confidence': pol_result['confidence'],
        'sentences': sentence_results,
        'overall_prediction': overall_prediction,
        'overall_weighted_probs': overall_probs,
        'num_sentences': len(sentences)
    }


# ===========================================================================
# Test Suite
# ===========================================================================

def test_on_sample_texts():
    """Test IG vs attention on sample texts"""
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    test_texts = [
        "The liberal party's welfare reform is merely a vote-seeking gambit that burdens taxpayers.",
        "Freedom House published a report Wednesday downgrading the United States democracy rating.",
        "Republican lawmakers blocked the progressive tax bill in a partisan vote.",
        "The senator announced plans for infrastructure investment at a town hall meeting.",
        "I love pizza on Fridays!",
        "Conservative media outlets distorted the facts about immigration policy.",
        "The government approved funding for education and healthcare programs.",
    ]

    print("\n" + "#" * 70)
    print("# EXPLAINABILITY TEST — INTEGRATED GRADIENTS")
    print("#" * 70)

    for i, text in enumerate(test_texts, 1):
        print(f"\n\n{'#' * 70}")
        print(f"# TEST {i}/{len(test_texts)}")
        print(f"{'#' * 70}\n")

        # Step 1: Politicalness
        pol_result = check_politicalness(text, pol_model, pol_tokenizer)
        print(f"Politicalness: {'Political' if pol_result['is_political'] else 'Not Political'}"
              f" ({pol_result['confidence']:.3f})")

        if not pol_result['is_political']:
            print("  → Skipping bias detection (not political)\n")
            continue

        print(f"\n{'─' * 70}")

        # Step 2: Bias detection with IG
        result = detect_bias(text, bias_model, bias_tokenizer, method='ig')
        visualize(text, result, top_k=10)

        print(f"\nHighlighted (threshold=0.3):")
        print(highlight_text(text, result, threshold=0.3))


def test_article_pipeline():
    """Test the full article analysis pipeline"""
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    article = (
        "The Department of Health and Human Services's Office for Civil Rights has released "
        "guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies "
        "against refusing to dispense abortion-inducing medication, stipulating that doing so "
        "is pregnancy discrimination. That includes discrimination based on current pregnancy, "
        "past pregnancy, potential or intended pregnancy, and medical conditions related to "
        "pregnancy or childbirth. HHS is committed to ensuring that everyone can access "
        "healthcare, free of discrimination."
    )

    print("\n" + "#" * 70)
    print("# FULL ARTICLE PIPELINE TEST")
    print("#" * 70)

    result = analyze_article(
        article, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
        method='ig'
    )

    if not result['is_political']:
        print("Article not classified as political. Skipping.")
        return

    print(f"\nPolitical confidence: {result['political_confidence']:.3f}")
    print(f"Sentences analyzed: {result['num_sentences']}")
    print(f"Overall prediction: {result['overall_prediction']}")
    print(f"Weighted probs: {result['overall_weighted_probs']}")

    print(f"\n{'─' * 70}")
    print("PER-SENTENCE BREAKDOWN:")
    print(f"{'─' * 70}")

    for i, sent_result in enumerate(result['sentences'], 1):
        print(f"\n  Sentence {i}: {sent_result['text'][:80]}...")
        print(f"  Prediction: {sent_result['prediction']} "
              f"(conf={sent_result['confidence']:.3f}, weight={sent_result['weight']:.2f})")
        print(f"  Top tokens: ", end="")
        for t in sent_result['top_tokens'][:3]:
            print(f"{t['clean_token']}({t['score']:.3f}) ", end="")
        print()


def compare_methods():
    """
    Side-by-side comparison of IG vs Occlusion on the same text.
    Useful for validating that IG gives sensible results.
    """
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    text = "Conservative media outlets distorted the facts about immigration policy."

    print("\n" + "#" * 70)
    print("# METHOD COMPARISON: IG vs OCCLUSION")
    print("#" * 70)

    for method_name in ['ig', 'occlusion']:
        print(f"\n{'=' * 70}")
        print(f"  METHOD: {method_name.upper()}")
        print(f"{'=' * 70}")

        if method_name == 'ig' and not HAS_CAPTUM:
            print("  (skipped — captum not installed)")
            continue

        result = detect_bias(text, bias_model, bias_tokenizer, method=method_name)
        visualize(text, result, top_k=8)


if __name__ == "__main__":
    # Compare IG vs Occlusion
    compare_methods()

    # Test on sample texts
    test_on_sample_texts()

    # Test full article pipeline
    test_article_pipeline()

    print("\n\n" + "=" * 70)
    print("✓ EXPLAINABILITY TEST COMPLETE")
    print("=" * 70)