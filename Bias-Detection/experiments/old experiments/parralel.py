"""
Parallel Processing Pipeline
Runs politicalness filtering and bias detection in parallel for optimal speed
"""

import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import numpy as np


class ParallelBiasDetector:
    """
    Parallel pipeline for efficient bias detection
    
    Pipeline:
    1. Politicalness filter (fast, binary) 
    2. Bias detection (slower, 3-class) - only if political
    
    Both run in parallel to minimize latency
    """
    
    def __init__(self, models_dir="models", device=None):
        self.models_dir = Path(models_dir)
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        self.load_models()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def load_models(self):
        """Load both models"""
        print("Loading models...")
        
        # Politicalness filter (fast, small)
        pol_dir = self.models_dir / "politicalness_filter"
        self.pol_tokenizer = AutoTokenizer.from_pretrained(pol_dir)
        self.pol_model = AutoModelForSequenceClassification.from_pretrained(
            pol_dir,
            output_attentions=False  # Don't need attention for filter
        ).to(self.device)
        self.pol_model.eval()
        
        # Bias detector (slower, larger)
        bias_dir = self.models_dir / "bias_detector"
        self.bias_tokenizer = AutoTokenizer.from_pretrained(bias_dir)
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(
            bias_dir,
            output_attentions=True  # Need attention for explainability
        ).to(self.device)
        self.bias_model.eval()
        
        print("✓ Models loaded")
    
    def check_politicalness(self, text: str) -> Dict:
        """
        Fast politicalness check
        Returns: {is_political: bool, confidence: float}
        """
        inputs = self.pol_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.pol_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
        
        # Assuming label 1 = political, 0 = non-political
        is_political = predicted_class == 1
        
        return {
            'is_political': is_political,
            'confidence': confidence,
            'probs': probs[0].cpu().numpy()
        }
    
    def detect_bias(self, text: str) -> Dict:
        """
        Detect bias (left/center/right)
        Returns: {bias: str, confidence: float, all_probs: dict, attention: array}
        """
        inputs = self.bias_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)
        
        # Get tokens for attention mapping
        tokens = self.bias_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            outputs = self.bias_model(**inputs)
        
        # Get predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
        
        # Label mapping
        labels = ['left', 'center', 'right']
        predicted_bias = labels[predicted_class]
        
        # Extract attention weights for explainability
        attention_weights = self._extract_token_importance(
            outputs.attentions,
            tokens
        )
        
        return {
            'bias': predicted_bias,
            'confidence': confidence,
            'all_probs': {
                'left': probs[0][0].item(),
                'center': probs[0][1].item(),
                'right': probs[0][2].item()
            },
            'attention_weights': attention_weights,
            'tokens': tokens
        }
    
    def _extract_token_importance(self, attentions: Tuple, tokens: List[str], top_k: int = 20) -> List[Dict]:
        """
        Extract token importance from attention weights
        
        Returns list of {token, score, position} sorted by importance
        """
        if not attentions:
            return []
        
        # Get last layer attention: (batch, heads, seq, seq)
        last_layer = attentions[-1]
        
        # Average across heads: (seq, seq)
        avg_attention = last_layer[0].mean(dim=0)
        
        # Get [CLS] token attention: (seq,)
        cls_attention = avg_attention[0].cpu().numpy()
        
        # Get top-k tokens (excluding special tokens)
        token_scores = []
        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_scores.append({
                    'token': token,
                    'score': float(score),
                    'position': i
                })
        
        # Sort by score
        token_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return token_scores[:top_k]
    
    async def analyze_text_async(self, text: str) -> Dict:
        """
        Async analysis of single text
        Runs politicalness check first, then bias detection if political
        """
        # Run politicalness check
        pol_result = self.check_politicalness(text)
        
        if not pol_result['is_political']:
            return {
                'is_political': False,
                'politicalness_confidence': pol_result['confidence'],
                'bias': None,
                'message': 'Text is not political'
            }
        
        # Run bias detection (only if political)
        bias_result = self.detect_bias(text)
        
        return {
            'is_political': True,
            'politicalness_confidence': pol_result['confidence'],
            'bias': bias_result['bias'],
            'bias_confidence': bias_result['confidence'],
            'all_bias_probs': bias_result['all_probs'],
            'important_tokens': bias_result['attention_weights'],
            'tokens': bias_result['tokens']
        }
    
    def analyze_text(self, text: str) -> Dict:
        """Synchronous wrapper for async analysis"""
        return asyncio.run(self.analyze_text_async(text))
    
    async def analyze_batch_async(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts in parallel"""
        tasks = [self.analyze_text_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Synchronous wrapper for batch analysis"""
        return asyncio.run(self.analyze_batch_async(texts))
    
    def analyze_article_sentences(self, article: str) -> Dict:
        """
        Analyze article sentence-by-sentence (your baseline approach)
        
        Returns aggregated results with per-sentence breakdown
        """
        # Split into sentences (using simple approach for now)
        sentences = self._split_sentences(article)
        
        print(f"Analyzing {len(sentences)} sentences...")
        
        # Filter political sentences first
        sentence_results = []
        for i, sent in enumerate(sentences):
            if len(sent.strip()) < 10:  # Skip very short
                continue
            
            result = self.analyze_text(sent)
            result['sentence'] = sent
            result['position'] = i
            sentence_results.append(result)
        
        # Count political vs non-political
        political_sents = [r for r in sentence_results if r['is_political']]
        
        if not political_sents:
            return {
                'is_political_article': False,
                'sentence_count': len(sentences),
                'political_sentence_count': 0,
                'overall_bias': None
            }
        
        # Aggregate bias from political sentences
        bias_counts = {'left': 0, 'center': 0, 'right': 0}
        weighted_probs = {'left': 0.0, 'center': 0.0, 'right': 0.0}
        
        for result in political_sents:
            bias = result['bias']
            confidence = result['bias_confidence']
            
            # Simple counting
            bias_counts[bias] += 1
            
            # Weighted by confidence
            for b in ['left', 'center', 'right']:
                weighted_probs[b] += result['all_bias_probs'][b] * confidence
        
        # Normalize weighted probs
        total_weight = sum(weighted_probs.values())
        if total_weight > 0:
            weighted_probs = {k: v/total_weight for k, v in weighted_probs.items()}
        
        # Overall bias (by weighted average)
        overall_bias = max(weighted_probs, key=weighted_probs.get)
        
        return {
            'is_political_article': True,
            'sentence_count': len(sentences),
            'political_sentence_count': len(political_sents),
            'non_political_sentence_count': len(sentences) - len(political_sents),
            'overall_bias': overall_bias,
            'bias_distribution': bias_counts,
            'weighted_probs': weighted_probs,
            'sentence_results': sentence_results,
            'confidence': weighted_probs[overall_bias]
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter
        TODO: Use spaCy for better results
        """
        import re
        # Simple split on . ! ?
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def benchmark_speed(self, texts: List[str]) -> Dict:
        """Benchmark processing speed"""
        print(f"\nBenchmarking with {len(texts)} texts...")
        
        # Sequential processing
        start = time.time()
        for text in texts:
            self.analyze_text(text)
        sequential_time = time.time() - start
        
        # Batch processing
        start = time.time()
        self.analyze_batch(texts)
        batch_time = time.time() - start
        
        results = {
            'num_texts': len(texts),
            'sequential_time': sequential_time,
            'batch_time': batch_time,
            'speedup': sequential_time / batch_time if batch_time > 0 else 0,
            'texts_per_second_sequential': len(texts) / sequential_time,
            'texts_per_second_batch': len(texts) / batch_time
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Sequential: {sequential_time:.2f}s ({results['texts_per_second_sequential']:.1f} texts/sec)")
        print(f"  Batch: {batch_time:.2f}s ({results['texts_per_second_batch']:.1f} texts/sec)")
        print(f"  Speedup: {results['speedup']:.2f}x")
        
        return results


def main():
    """Test parallel processing"""
    detector = ParallelBiasDetector()
    
    # Test texts
    test_texts = [
        "The liberal party's welfare reform is merely a vote-seeking gambit.",
        "The weather today is sunny with a chance of rain.",
        "Conservative lawmakers blocked the progressive tax bill.",
        "I went to the store to buy groceries.",
        "The right-wing media distorts the facts about immigration."
    ]
    
    print("="*60)
    print("TESTING PARALLEL BIAS DETECTION")
    print("="*60)
    
    # Test single text
    print("\n1. Single Text Analysis:")
    result = detector.analyze_text(test_texts[0])
    print(f"\nText: {test_texts[0]}")
    print(f"Is Political: {result['is_political']}")
    if result['is_political']:
        print(f"Bias: {result['bias']} (confidence: {result['bias_confidence']:.3f})")
        print(f"\nTop biased tokens:")
        for token_info in result['important_tokens'][:5]:
            print(f"  {token_info['token']:15s} - {token_info['score']:.4f}")
    
    # Test batch
    print("\n" + "="*60)
    print("2. Batch Analysis:")
    results = detector.analyze_batch(test_texts)
    for text, result in zip(test_texts, results):
        print(f"\nText: {text[:50]}...")
        print(f"  Political: {result['is_political']}")
        if result['is_political']:
            print(f"  Bias: {result['bias']} ({result['bias_confidence']:.3f})")
    
    # Test article analysis
    print("\n" + "="*60)
    print("3. Article Analysis (Sentence-by-Sentence):")
    article = """
    The recent protests led by women once again turned chaotic, with several reports 
    of emotional outbursts and irrational behavior. Freedom House published a report 
    Wednesday downgrading the United States from a democracy to whatever political 
    system lobsters have. Our nation already passed the tipping point where we might 
    hope to match the deliberative bicameral legislative process of shore crabs.
    """
    
    article_result = detector.analyze_article_sentences(article)
    print(f"\nArticle Analysis:")
    print(f"  Total sentences: {article_result['sentence_count']}")
    print(f"  Political sentences: {article_result['political_sentence_count']}")
    print(f"  Overall bias: {article_result['overall_bias']}")
    print(f"  Confidence: {article_result['confidence']:.3f}")
    print(f"\n  Bias distribution: {article_result['bias_distribution']}")
    print(f"  Weighted probs: {article_result['weighted_probs']}")
    
    # Benchmark
    print("\n" + "="*60)
    print("4. Speed Benchmark:")
    detector.benchmark_speed(test_texts * 10)  # 50 texts
    
    print("\n" + "="*60)
    print("✓ Parallel processing ready!")
    print("="*60)


if __name__ == "__main__":
    main()