"""
Explainability for bias detection using Integrated Gradients.
Measures how much each token embedding contributes to the model's bias prediction.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import spacy
from captum.attr import LayerIntegratedGradients

nlp = spacy.load("en_core_web_sm")

PUNCTUATION = {
    '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']',
    '-', '-', '—', '▁.', '▁,', '▁!', '▁?', '▁;', '▁:', '▁-'
}


def load_models():
    print("Loading models..")

    bias_model = AutoModelForSequenceClassification.from_pretrained("../models/demo_models/bias_detector")
    bias_tokenizer = AutoTokenizer.from_pretrained("../models/demo_models/bias_detector")

    pol_model = AutoModelForSequenceClassification.from_pretrained("../models/demo_models/politicalness_filter")
    pol_tokenizer = AutoTokenizer.from_pretrained("../models/demo_models/politicalness_filter")

    bias_model.eval()
    pol_model.eval()

    print("Models loaded\n")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


# politicalness filter using Political DEBATE (NLI model)
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


def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def merge_short_sentences(sentences, min_tokens=30, max_tokens=150):
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
        merged.append(buffer.strip())

    return merged


def _build_token_list(tokens, scores):
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


# Integrated Gradients attribution
def compute_ig_attributions(text, model, tokenizer, target_class=None, n_steps=50):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    labels = ['Left', 'Center', 'Right']
    predicted_class = torch.argmax(probs).item()
    if target_class is None:
        target_class = predicted_class

    # get the DeBERTa embedding layer
    embedding_layer = model.deberta.embeddings.word_embeddings

    lig = LayerIntegratedGradients(
        lambda input_ids, attention_mask: model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits,
        embedding_layer
    )

    # baseline = pad tokens (represents "absence of information")
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=False
    )

    # sum across embedding dim to get per-token attribution, then normalize
    token_attributions = attributions.sum(dim=-1).squeeze(0)
    abs_attr = token_attributions.abs()
    if abs_attr.max() > 0:
        normalized = (abs_attr / abs_attr.max()).detach().numpy()
    else:
        normalized = abs_attr.detach().numpy()

    return {
        'predicted_label': labels[predicted_class],
        'confidence': float(probs[predicted_class]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(labels)},
        'target_class': labels[target_class],
        'token_attention': _build_token_list(tokens, normalized),
        'all_tokens': tokens,
    }


def visualize(text, result, top_k=10):
    print(f"\nTEXT: {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"Prediction: {result['predicted_label']} ({result['confidence']:.3f})")
    print(f"Probs: L={result['all_probs']['Left']:.3f}  C={result['all_probs']['Center']:.3f}  R={result['all_probs']['Right']:.3f}")
    print(f"\nTop {top_k} tokens:")
    for i, item in enumerate(result['token_attention'][:top_k], 1):
        print(f"  {i}. {item['clean_token']:<20s} {item['score']:.4f}")


# full article pipeline: politicalness -> split -> per-sentence bias + IG -> aggregate
def analyze_article(text, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                    min_merge_tokens=30, low_conf_threshold=0.45):

    pol_result = check_politicalness(text[:2000], pol_model, pol_tokenizer)
    if not pol_result['is_political']:
        return {
            'is_political': False,
            'political_confidence': pol_result['confidence'],
            'sentences': [],
            'overall_prediction': None
        }

    sentences = split_into_sentences(text)
    sentences = merge_short_sentences(sentences, min_tokens=min_merge_tokens)

    sentence_results = []
    weighted_probs = []

    for sent in sentences:
        result = compute_ig_attributions(sent, bias_model, bias_tokenizer)

        weight = 1.0
        if result['confidence'] < low_conf_threshold:
            weight *= 0.3
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
        })

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


if __name__ == "__main__":
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    test_texts = [
        "The liberal party's welfare reform is merely a vote-seeking gambit that burdens taxpayers.",
        "Freedom House published a report Wednesday downgrading the United States democracy rating.",
        "Republican lawmakers blocked the progressive tax bill in a partisan vote.",
        "I love pizza on Fridays!",
        "Conservative media outlets distorted the facts about immigration policy.",
    ]

    # test IG on sample texts
    print("\n--- Sample Texts ---")
    for text in test_texts:
        pol = check_politicalness(text, pol_model, pol_tokenizer)
        print(f"\n{'Political' if pol['is_political'] else 'Not Political'} ({pol['confidence']:.3f}): {text[:80]}")
        if pol['is_political']:
            result = compute_ig_attributions(text, bias_model, bias_tokenizer)
            visualize(text, result, top_k=5)

    # test full article pipeline
    print("\n--- Article Pipeline ---")
    article = (
        "The Department of Health and Human Services's Office for Civil Rights has released "
        "guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies "
        "against refusing to dispense abortion-inducing medication, stipulating that doing so "
        "is pregnancy discrimination."
    )
    result = analyze_article(article, bias_model, bias_tokenizer, pol_model, pol_tokenizer)
    print(f"Overall: {result['overall_prediction']}")
    for s in result['sentences']:
        tokens_str = ", ".join(t['clean_token'] for t in s['top_tokens'][:3])
        print(f"  {s['prediction']} ({s['confidence']:.3f}) [{tokens_str}] {s['text'][:60]}...")