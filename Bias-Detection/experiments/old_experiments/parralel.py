"""
Parallel processing prototype.
Runs politicalness filtering and bias detection in parallel to test
if threading helps with latency.
"""

import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import time
import re
import numpy as np


class ParallelBiasDetector:

    def __init__(self, models_dir="../models/demo_models", device=None):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        self.load_models()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def load_models(self):
        print("Loading models...")
        pol_dir = self.models_dir / "politicalness_filter"
        self.pol_tokenizer = AutoTokenizer.from_pretrained(pol_dir)
        self.pol_model = AutoModelForSequenceClassification.from_pretrained(pol_dir).to(self.device)
        self.pol_model.eval()

        bias_dir = self.models_dir / "bias_detector"
        self.bias_tokenizer = AutoTokenizer.from_pretrained(bias_dir)
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(
            bias_dir, output_attentions=True
        ).to(self.device)
        self.bias_model.eval()
        print("Models loaded")

    def check_politicalness(self, text):
        inputs = self.pol_tokenizer(text, return_tensors="pt", truncation=True,
                                     padding=True, max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.pol_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1).item()
        return {
            'is_political': predicted == 1,
            'confidence': probs[0][predicted].item(),
        }

    def detect_bias(self, text):
        inputs = self.bias_tokenizer(text, return_tensors="pt", truncation=True,
                                      padding=True, max_length=256).to(self.device)
        tokens = self.bias_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = self.bias_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1).item()
        labels = ['left', 'center', 'right']

        # extract attention from last layer, average across heads, take CLS row
        last_layer = outputs.attentions[-1]
        cls_attention = last_layer[0].mean(dim=0)[0].cpu().numpy()

        token_scores = []
        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_scores.append({'token': token, 'score': float(score), 'position': i})
        token_scores.sort(key=lambda x: x['score'], reverse=True)

        return {
            'bias': labels[predicted],
            'confidence': probs[0][predicted].item(),
            'all_probs': {l: probs[0][i].item() for i, l in enumerate(labels)},
            'attention_weights': token_scores[:20],
        }

    async def analyze_text_async(self, text):
        pol = self.check_politicalness(text)
        if not pol['is_political']:
            return {'is_political': False, 'confidence': pol['confidence'], 'bias': None}
        bias = self.detect_bias(text)
        return {
            'is_political': True, 'confidence': pol['confidence'],
            'bias': bias['bias'], 'bias_confidence': bias['confidence'],
            'all_probs': bias['all_probs'], 'important_tokens': bias['attention_weights'],
        }

    def analyze_text(self, text):
        return asyncio.run(self.analyze_text_async(text))

    def analyze_article(self, article):
        sentences = [s.strip() for s in re.split(r'[.!?]+', article) if s.strip()]
        print(f"Analyzing {len(sentences)} sentences...")

        results = []
        for sent in sentences:
            if len(sent.strip()) < 10:
                continue
            results.append(self.analyze_text(sent))

        political = [r for r in results if r['is_political']]
        if not political:
            return {'overall_bias': None, 'sentence_count': len(sentences)}

        bias_counts = {'left': 0, 'center': 0, 'right': 0}
        weighted = {'left': 0.0, 'center': 0.0, 'right': 0.0}
        for r in political:
            bias_counts[r['bias']] += 1
            for b in ['left', 'center', 'right']:
                weighted[b] += r['all_probs'][b] * r['bias_confidence']

        total = sum(weighted.values())
        if total > 0:
            weighted = {k: v / total for k, v in weighted.items()}

        return {
            'overall_bias': max(weighted, key=weighted.get),
            'sentence_count': len(sentences),
            'political_count': len(political),
            'bias_distribution': bias_counts,
            'weighted_probs': weighted,
        }


if __name__ == "__main__":
    detector = ParallelBiasDetector()

    test_texts = [
        "The liberal party's welfare reform is merely a vote-seeking gambit.",
        "The weather today is sunny with a chance of rain.",
        "Conservative lawmakers blocked the progressive tax bill.",
        "I went to the store to buy groceries.",
        "The right-wing media distorts the facts about immigration."
    ]

    print("\nSingle text analysis")
    for text in test_texts:
        result = detector.analyze_text(text)
        status = f"{result['bias']} ({result['bias_confidence']:.3f})" if result['is_political'] else "Not political"
        print(f"  {status}: {text[:60]}")

    print("\nArticle analysis")
    article = """
    Freedom House published a report Wednesday downgrading the United States
    from a democracy. Our nation already passed the tipping point where we
    might hope to match the deliberative legislative process of shore crabs.
    """
    result = detector.analyze_article(article)
    print(f"  Overall: {result['overall_bias']}")
    print(f"  Distribution: {result.get('bias_distribution', {})}")