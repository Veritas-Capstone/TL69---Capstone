"""
Test attention weight extraction for explainability
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


def load_models():
    print("Loading models..")

    bias_model = AutoModelForSequenceClassification.from_pretrained(
        "../models/demo_models/bias_detector",
        output_attentions=True  # needed for attention extraction
    )
    bias_tokenizer = AutoTokenizer.from_pretrained("../models/demo_models/bias_detector")

    pol_model = AutoModelForSequenceClassification.from_pretrained("../models/demo_models/politicalness_filter")
    pol_tokenizer = AutoTokenizer.from_pretrained("../models/demo_models/politicalness_filter")

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


def extract_attention_weights(text, model, tokenizer, top_k=15):
    # uses CLS token attention from the last layer, averaged across heads
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()
    labels = ['Left', 'Center', 'Right']

    # last layer attention, average across heads, take CLS row
    last_layer_attention = outputs.attentions[-1]
    avg_attention = last_layer_attention[0].mean(dim=0)
    cls_attention = avg_attention[0].cpu().numpy()

    token_scores = []
    for i, (token, score) in enumerate(zip(tokens, cls_attention)):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue
        clean = token.replace('##', '').replace('▁', '').strip()
        if not clean:
            continue
        token_scores.append({
            'token': clean,
            'raw_token': token,
            'score': float(score),
            'position': i
        })

    token_scores.sort(key=lambda x: x['score'], reverse=True)

    return {
        'predicted_bias': labels[predicted_class],
        'confidence': confidence,
        'all_probs': {l: float(probs[0][i]) for i, l in enumerate(labels)},
        'top_tokens': token_scores[:top_k],
    }


def visualize_attention(text, result):
    print(f"\nText: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Prediction: {result['predicted_bias']} ({result['confidence']:.3f})")
    print(f"Probs: L={result['all_probs']['Left']:.3f} C={result['all_probs']['Center']:.3f} R={result['all_probs']['Right']:.3f}")
    print(f"\nTop attended tokens:")
    for i, t in enumerate(result['top_tokens'][:10], 1):
        print(f"  {i}. {t['token']:<20s} {t['score']:.4f}")


if __name__ == "__main__":
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

    test_texts = [
        "The liberal party's welfare reform is merely a vote-seeking gambit that burdens taxpayers.",
        "The weather today is sunny with a chance of rain.",
        "Conservative lawmakers blocked the progressive tax bill in a partisan vote.",
        "Freedom House published a report downgrading the United States democracy rating.",
        "I love pizza on Fridays!",
    ]

    for text in test_texts:
        pol = check_politicalness(text, pol_model, pol_tokenizer)
        print(f"\nPolitical: {'Yes' if pol['is_political'] else 'No'} ({pol['confidence']:.3f})")

        if not pol['is_political']:
            print(f"  Skipping: {text[:60]}")
            continue

        result = extract_attention_weights(text, bias_model, bias_tokenizer)
        visualize_attention(text, result)