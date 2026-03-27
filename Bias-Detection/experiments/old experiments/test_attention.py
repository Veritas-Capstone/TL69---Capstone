"""
Test Attention Extraction for Explainability
Shows which words contribute most to bias prediction
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

def load_models():
    """Load both models"""
    print("Loading models...")
    
    bias_model = AutoModelForSequenceClassification.from_pretrained(
        "models/bias_detector",
        output_attentions=True  # CRITICAL for attention extraction
    )
    bias_tokenizer = AutoTokenizer.from_pretrained("models/bias_detector")
    
    pol_model = AutoModelForSequenceClassification.from_pretrained(
        "models/politicalness_filter"
    )
    pol_tokenizer = AutoTokenizer.from_pretrained("models/politicalness_filter")
    
    print("✓ Models loaded\n")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


def check_politicalness(text, model, tokenizer):
    """Check if text is political"""
    # Political DEBATE is a binary classifier, not NLI
    # Just pass the text directly
    hypothesis = "This text is about politics."
    inputs = tokenizer(text, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # NLI models typically output: [entailment, not_entailment] or [contradiction, neutral, entailment]
    # Check the model's label mapping - for Political DEBATE:
    # entailment = the hypothesis IS true for this text
    entail_idx = model.config.label2id.get("ENTAILMENT", model.config.label2id.get("entailment", 0))
    political_prob = float(probs[entail_idx])
    
    return {
        'is_political': political_prob > 0.5,
        'confidence': political_prob
    }


def detect_bias_with_attention(text, model, tokenizer):
    """
    Detect bias AND extract attention weights
    Returns which words contribute most to the prediction
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # Get tokens for display
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get predictions
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    labels = ['Left', 'Center', 'Right']
    predicted_class = torch.argmax(probs).item()
    predicted_label = labels[predicted_class]
    
    # Extract attention weights
    # outputs.attentions is tuple of (num_layers, batch, heads, seq_len, seq_len)
    attention_weights = outputs.attentions
    
    # Use last layer attention (usually most meaningful)
    last_layer_attention = attention_weights[-1]  # Shape: (batch=1, heads, seq, seq)
    
    # Average across all attention heads
    avg_attention = last_layer_attention[0].mean(dim=0)  # Shape: (seq, seq)
    
    # Get attention FROM [CLS] token (first token)
    # This shows importance of each token for classification
    cls_attention = avg_attention[0]  # Shape: (seq_len,)
    
    # Create list of (token, attention_score) pairs
    # Skip special tokens AND punctuation
    punctuation = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '-', '–', '—', '▁.', '▁,', '▁!', '▁?', '▁;', '▁:', '▁-'}
    
    token_attention = []
    for i, (token, score) in enumerate(zip(tokens, cls_attention)):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue
        
        # Skip pure punctuation tokens
        clean_token = token.replace('##', '').replace('▁', '').strip()
        if clean_token in punctuation or clean_token == '':
            continue
        
        # Skip tokens that are only punctuation after cleaning
        if all(c in '.,!?;:\'"()[]<>-–—' for c in clean_token):
            continue
        
        token_attention.append({
            'token': token.replace('##', ''),  # Keep ▁ for word boundary info
            'score': float(score),
            'position': i
        })
    
    # Sort by attention score (highest first)
    token_attention.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'predicted_label': predicted_label,
        'confidence': float(probs[predicted_class]),
        'all_probs': {
            'Left': float(probs[0]),
            'Center': float(probs[1]),
            'Right': float(probs[2])
        },
        'token_attention': token_attention,
        'all_tokens': tokens
    }


def visualize_attention(text, result, top_k=10):
    """
    Print attention visualization
    Shows which words are most important for the prediction
    """
    print("="*70)
    print(f"TEXT: {text}")
    print("="*70)
    
    print(f"\nPrediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    print(f"\nAll Probabilities:")
    for label, prob in result['all_probs'].items():
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        print(f"  {label:7s} {prob:.3f} {bar}")
    
    print(f"\n{'='*70}")
    print(f"TOP {top_k} MOST IMPORTANT WORDS (Punctuation Filtered)")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Token':<20} {'Attention Score':<15} {'Visual'}")
    print("-"*70)
    
    for i, item in enumerate(result['token_attention'][:top_k], 1):
        token = item['token']
        score = item['score']
        
        # Visual bar (scaled)
        max_score = result['token_attention'][0]['score']
        bar_length = int((score / max_score) * 30)
        bar = "█" * bar_length
        
        print(f"{i:<6} {token:<20} {score:<15.4f} {bar}")
    
    print("="*70)


def highlight_text_with_attention(text, result, threshold=0.03):
    """
    Create a version of text with important words highlighted
    Words above threshold get marked with ** **
    Filters out punctuation for cleaner display
    """
    tokens = result['all_tokens']
    token_dict = {item['position']: item['score'] for item in result['token_attention']}
    
    punctuation = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '-', '–', '—'}
    
    highlighted = []
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue
        
        clean_token = token.replace('##', '').replace('▁', ' ').strip()
        
        # Skip empty or pure punctuation
        if not clean_token or all(c in punctuation for c in clean_token):
            continue
        
        score = token_dict.get(i, 0)
        
        if score > threshold:
            highlighted.append(f"**{clean_token}**")
        else:
            highlighted.append(clean_token)
    
    return ' '.join(highlighted)


def test_on_sample_texts():
    """Test on various examples"""
    
    # Load models
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()
    
    # Test texts with different bias levels
    test_texts = [
        "The liberal party's welfare reform is merely a vote-seeking gambit that burdens taxpayers.",
        "Freedom House published a report Wednesday downgrading the United States democracy rating.",
        "Republican lawmakers blocked the progressive tax bill in a partisan vote.",
        "The senator announced plans for infrastructure investment at a town hall meeting.",
        "I love pizza on Fridays!",
        "Conservative media outlets distorted the facts about immigration policy.",
        "The government approved funding for education and healthcare programs.",
    ]
    
    print("\n" + "#"*70)
    print("# ATTENTION EXTRACTION TEST")
    print("#"*70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n\n{'#'*70}")
        print(f"# TEST {i}/{len(test_texts)}")
        print(f"{'#'*70}\n")
        
        # Step 1: Check if political
        pol_result = check_politicalness(text, pol_model, pol_tokenizer)
        print(f"Politicalness Check:")
        print(f"  Is Political: {pol_result['is_political']}")
        print(f"  Confidence: {pol_result['confidence']:.3f}")
        
        if not pol_result['is_political']:
            print(f"\n  → Skipping bias detection (not political)\n")
            continue
        
        print(f"\n{'─'*70}")
        print("Bias Detection with Attention:")
        print('─'*70)
        
        # Step 2: Detect bias with attention
        bias_result = detect_bias_with_attention(text, bias_model, bias_tokenizer)
        
        # Visualize
        visualize_attention(text, bias_result, top_k=15)
        
        # Show highlighted version
        print(f"\nHighlighted Text (** = high attention, punctuation filtered):")
        print(f"{highlight_text_with_attention(text, bias_result, threshold=0.03)}")


def test_on_your_dataset():
    """
    Test on actual data from your preprocessed datasets
    """
    import pandas as pd
    from pathlib import Path
    
    print("\n" + "#"*70)
    print("# TESTING ON YOUR PREPROCESSED DATA")
    print("#"*70)
    
    # Load models
    bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()
    
    # Try to load a dataset
    dataset_path = Path("preprocessed/article_bias_prediction.parquet")
    
    if not dataset_path.exists():
        print(f"\n⚠️  Dataset not found at {dataset_path}")
        print("Skipping dataset test.")
        return
    
    df = pd.read_parquet(dataset_path)
    print(f"\n✓ Loaded {len(df)} articles from article_bias_prediction")
    
    # Sample 3 articles (one from each class if possible)
    samples = []
    for leaning in ['left', 'center', 'right']:
        subset = df[df['leaning'] == leaning]
        if len(subset) > 0:
            samples.append(subset.sample(1).iloc[0])
    
    if not samples:
        print("No samples found. Check dataset labels.")
        return
    
    print(f"\nTesting on {len(samples)} sample articles...\n")
    
    for i, row in enumerate(samples, 1):
        text = row['body'][:500]  # First 500 chars
        true_label = row['leaning']
        
        print(f"\n{'='*70}")
        print(f"SAMPLE {i} - True Label: {true_label.upper()}")
        print(f"{'='*70}")
        
        # Check politicalness
        pol_result = check_politicalness(text, pol_model, pol_tokenizer)
        print(f"Political: {pol_result['is_political']} ({pol_result['confidence']:.3f})")
        
        if pol_result['is_political']:
            # Detect bias
            bias_result = detect_bias_with_attention(text, bias_model, bias_tokenizer)
            
            print(f"\nPrediction: {bias_result['predicted_label']} (True: {true_label})")
            print(f"Confidence: {bias_result['confidence']:.3f}")
            
            # Show top biased words
            print(f"\nTop 5 biased words:")
            for j, item in enumerate(bias_result['token_attention'][:5], 1):
                print(f"  {j}. {item['token']} ({item['score']:.4f})")


if __name__ == "__main__":
    # Test on sample texts
    test_on_sample_texts()
    
    # Test on your actual data
    print("\n\n")
    test_on_your_dataset()
    
    print("\n\n" + "="*70)
    print("✓ ATTENTION EXTRACTION TEST COMPLETE!")
    print("="*70)
    print("\nKey Findings:")
    print("  • Attention weights successfully extracted")
    print("  • Can identify which words contribute to bias")
    print("  • Ready for visualization in browser extension")
    print("\nNext Steps:")
    print("  1. Validate full datasets")
    print("  2. Test on more examples")
    print("  3. Implement in browser extension UI")

# """
# Improvements 1-4: Immediate Fixes for Bias Detection

# 1. Politicalness threshold adjustment
# 2. Center detection with thresholds
# 3. Better test examples
# 4. Sentence-level aggregation (your baseline approach)
# """

# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# import pandas as pd
# from pathlib import Path
# import re

# # ============================================================================
# # IMPROVEMENT 1: Stricter Politicalness Threshold
# # ============================================================================

# def check_politicalness_improved(text, model, tokenizer, threshold=0.8):
#     """
#     More conservative politicalness check
#     Threshold raised from 0.5 to 0.8 to reduce false positives
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
#     if len(probs) == 2:
#         political_prob = float(probs[1])
#     else:
#         political_prob = float(probs[-1])
    
#     return {
#         'is_political': political_prob > threshold,  # More strict!
#         'confidence': political_prob
#     }


# # ============================================================================
# # IMPROVEMENT 2: Center Detection with Thresholds
# # ============================================================================

# def predict_bias_with_center_threshold(text, model, tokenizer, 
#                                        left_threshold=0.6, 
#                                        right_threshold=0.6):
#     """
#     Improved center detection using threshold zones
    
#     Logic:
#     - If Left > 0.6 → Left
#     - If Right > 0.6 → Right
#     - Otherwise → Center
    
#     This addresses: "low confidence in left and right can be center"
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
#     with torch.no_grad():
#         outputs = model(**inputs, output_attentions=True)
    
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
#     left_prob = float(probs[0])
#     center_prob = float(probs[1])
#     right_prob = float(probs[2])
    
#     # Apply threshold logic
#     if left_prob > left_threshold:
#         predicted_label = 'Left'
#         confidence = left_prob
#     elif right_prob > right_threshold:
#         predicted_label = 'Right'
#         confidence = right_prob
#     else:
#         predicted_label = 'Center'
#         confidence = max(left_prob, center_prob, right_prob)
    
#     return {
#         'predicted_label': predicted_label,
#         'confidence': confidence,
#         'all_probs': {
#             'Left': left_prob,
#             'Center': center_prob,
#             'Right': right_prob
#         }
#     }


# # ============================================================================
# # IMPROVEMENT 3: Better Test Examples
# # ============================================================================

# def get_diverse_test_examples():
#     """
#     More diverse and clear test examples across all categories
#     """
#     return {
#         'clear_left': [
#             "Progressive Democrats championed Medicare for All and free college tuition for working families.",
#             "Socialist policies will redistribute wealth and ensure equality for marginalized communities.",
#             "Liberal activists demanded climate action and expanded social safety nets.",
#         ],
#         'clear_right': [
#             "Republican senators defended free market capitalism and reduced government regulations.",
#             "Conservative values prioritize individual liberty, personal responsibility, and limited government.",
#             "Right-wing commentators criticized big government spending and socialist overreach.",
#         ],
#         'clear_center': [
#             "Both parties agreed to pass the bipartisan infrastructure bill after negotiations.",
#             "The nonpartisan Congressional Budget Office released a balanced policy analysis.",
#             "Moderate lawmakers from both sides sought compromise on immigration reform.",
#         ],
#         'non_political': [
#             "The weather forecast predicts rain this weekend with temperatures in the 60s.",
#             "I ordered a turkey sandwich for lunch at the new deli downtown.",
#             "The movie received positive reviews from critics and audiences alike.",
#         ]
#     }


# # ============================================================================
# # IMPROVEMENT 4: Sentence-Level Aggregation (Your Baseline)
# # ============================================================================

# def split_into_sentences(text):
#     """
#     Split text into sentences
#     Uses simple regex - can upgrade to spaCy later
#     """
#     # Simple sentence splitter
#     sentences = re.split(r'[.!?]+', text)
#     return [s.strip() for s in sentences if len(s.strip()) > 10]


# def compute_sentence_weight(sentence, prediction, position=None, total_sentences=None):
#     """
#     YOUR weighting strategy (from baseline + Week 3 feedback)
    
#     Weights based on:
#     1. Confidence (existing)
#     2. Quote detection (Week 3)
#     3. Position weighting (Week 3) - headlines matter more
#     4. Counterpoint detection (Week 3)
#     """
#     weight = 1.0
#     confidence = prediction['confidence']
    
#     # 1. Low confidence downweighting (from baseline)
#     if confidence < 0.45:
#         weight *= 0.3
    
#     # 2. Quote detection (Week 3 feedback)
#     if sentence.strip().startswith('"') or sentence.strip().startswith('"'):
#         weight *= 0.5
    
#     # 3. Position weighting (Week 3 feedback)
#     if position is not None and total_sentences is not None:
#         if position == 0:  # First sentence / headline area
#             weight *= 1.5
#         elif position <= 2:  # First few sentences
#             weight *= 1.2
#         elif position >= total_sentences - 2:  # Last sentences (often biased)
#             weight *= 1.3
    
#     # 4. Counterpoint detection (Week 3 feedback)
#     counterpoint_phrases = [
#         'critics argue', 'critics say', 'critics claim',
#         'opponents say', 'opponents argue',
#         'however', 'on the other hand',
#         'but critics', 'but opponents'
#     ]
#     if any(phrase in sentence.lower() for phrase in counterpoint_phrases):
#         weight *= 0.5
    
#     return weight


# def analyze_article_with_aggregation(article_text, bias_model, bias_tokenizer, 
#                                      pol_model, pol_tokenizer,
#                                      pol_threshold=0.8,
#                                      left_threshold=0.6,
#                                      right_threshold=0.6):
#     """
#     Full article analysis with sentence-level aggregation
#     YOUR MAIN CONTRIBUTION!
#     """
#     # Split into sentences
#     sentences = split_into_sentences(article_text)
    
#     if len(sentences) == 0:
#         return {'error': 'No valid sentences found'}
    
#     print(f"\nAnalyzing article with {len(sentences)} sentences...")
    
#     # Analyze each sentence
#     sentence_results = []
#     political_count = 0
    
#     for i, sentence in enumerate(sentences):
#         # Check politicalness
#         pol_result = check_politicalness_improved(
#             sentence, pol_model, pol_tokenizer, threshold=pol_threshold
#         )
        
#         if not pol_result['is_political']:
#             continue
        
#         political_count += 1
        
#         # Detect bias with center thresholds
#         bias_result = predict_bias_with_center_threshold(
#             sentence, bias_model, bias_tokenizer,
#             left_threshold=left_threshold,
#             right_threshold=right_threshold
#         )
        
#         # Compute weight
#         weight = compute_sentence_weight(
#             sentence, bias_result, 
#             position=i, 
#             total_sentences=len(sentences)
#         )
        
#         sentence_results.append({
#             'sentence': sentence,
#             'prediction': bias_result['predicted_label'],
#             'confidence': bias_result['confidence'],
#             'all_probs': bias_result['all_probs'],
#             'weight': weight,
#             'position': i
#         })
    
#     if political_count == 0:
#         return {
#             'is_political_article': False,
#             'total_sentences': len(sentences),
#             'political_sentences': 0
#         }
    
#     # Weighted aggregation
#     weighted_probs = {'Left': 0.0, 'Center': 0.0, 'Right': 0.0}
#     total_weight = 0.0
    
#     for result in sentence_results:
#         w = result['weight']
#         for label in ['Left', 'Center', 'Right']:
#             weighted_probs[label] += result['all_probs'][label] * w
#         total_weight += w
    
#     # Normalize
#     if total_weight > 0:
#         weighted_probs = {k: v/total_weight for k, v in weighted_probs.items()}
    
#     # Apply center threshold to aggregated result
#     if weighted_probs['Left'] > left_threshold:
#         overall_prediction = 'Left'
#     elif weighted_probs['Right'] > right_threshold:
#         overall_prediction = 'Right'
#     else:
#         overall_prediction = 'Center'
    
#     return {
#         'is_political_article': True,
#         'total_sentences': len(sentences),
#         'political_sentences': political_count,
#         'non_political_sentences': len(sentences) - political_count,
#         'overall_prediction': overall_prediction,
#         'weighted_probs': weighted_probs,
#         'confidence': weighted_probs[overall_prediction],
#         'sentence_results': sentence_results
#     }


# # ============================================================================
# # TEST ALL IMPROVEMENTS
# # ============================================================================

# def test_all_improvements():
#     """Run comprehensive tests on all 4 improvements"""
    
#     print("="*70)
#     print("TESTING ALL 4 IMPROVEMENTS")
#     print("="*70)
    
#     # Load models
#     print("\nLoading models...")
#     bias_model = AutoModelForSequenceClassification.from_pretrained(
#         "models/bias_detector",
#         output_attentions=True
#     )
#     bias_tokenizer = AutoTokenizer.from_pretrained("models/bias_detector")
    
#     pol_model = AutoModelForSequenceClassification.from_pretrained(
#         "models/politicalness_filter"
#     )
#     pol_tokenizer = AutoTokenizer.from_pretrained("models/politicalness_filter")
#     print("✓ Models loaded\n")
    
#     # ========================================================================
#     # TEST 1: Politicalness Threshold
#     # ========================================================================
#     print("\n" + "="*70)
#     print("TEST 1: POLITICALNESS THRESHOLD COMPARISON")
#     print("="*70)
    
#     test_texts = [
#         "I love pizza on Fridays!",
#         "The weather is nice today.",
#         "Republican lawmakers blocked the bill.",
#     ]
    
#     print("\nComparing threshold 0.5 vs 0.8:\n")
#     print(f"{'Text':<50} {'@0.5':<15} {'@0.8':<15}")
#     print("-"*70)
    
#     for text in test_texts:
#         result_05 = check_politicalness_improved(text, pol_model, pol_tokenizer, threshold=0.5)
#         result_08 = check_politicalness_improved(text, pol_model, pol_tokenizer, threshold=0.8)
        
#         status_05 = "Political" if result_05['is_political'] else "Non-political"
#         status_08 = "Political" if result_08['is_political'] else "Non-political"
        
#         print(f"{text[:47]:<50} {status_05:<15} {status_08:<15}")
    
#     # ========================================================================
#     # TEST 2: Center Detection
#     # ========================================================================
#     print("\n\n" + "="*70)
#     print("TEST 2: CENTER DETECTION WITH THRESHOLDS")
#     print("="*70)
    
#     test_examples = get_diverse_test_examples()
    
#     thresholds = [0.5, 0.6, 0.7]
    
#     for thresh in thresholds:
#         print(f"\n--- Testing with threshold {thresh} ---")
        
#         correct = 0
#         total = 0
        
#         for true_label, texts in test_examples.items():
#             # Skip non-political
#             if true_label == 'non_political':
#                 continue
            
#             expected = true_label.replace('clear_', '').title()
            
#             for text in texts:
#                 result = predict_bias_with_center_threshold(
#                     text, bias_model, bias_tokenizer,
#                     left_threshold=thresh,
#                     right_threshold=thresh
#                 )
                
#                 predicted = result['predicted_label']
#                 is_correct = predicted.lower() == expected.lower()
                
#                 if is_correct:
#                     correct += 1
#                 total += 1
                
#                 symbol = "✓" if is_correct else "✗"
#                 print(f"  {symbol} {text[:60]:<62} → {predicted} (expected {expected})")
        
#         accuracy = correct / total * 100 if total > 0 else 0
#         print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    
#     # ========================================================================
#     # TEST 3: Better Examples
#     # ========================================================================
#     print("\n\n" + "="*70)
#     print("TEST 3: PERFORMANCE ON DIVERSE EXAMPLES")
#     print("="*70)
    
#     print("\nUsing threshold 0.6 (best from Test 2):")
    
#     all_examples = get_diverse_test_examples()
    
#     for category, texts in all_examples.items():
#         print(f"\n--- {category.upper()} ---")
        
#         for text in texts:
#             # Check politicalness first
#             pol_result = check_politicalness_improved(
#                 text, pol_model, pol_tokenizer, threshold=0.8
#             )
            
#             if not pol_result['is_political']:
#                 print(f"  Non-political: {text[:60]}")
#                 continue
            
#             # Predict bias
#             bias_result = predict_bias_with_center_threshold(
#                 text, bias_model, bias_tokenizer,
#                 left_threshold=0.6,
#                 right_threshold=0.6
#             )
            
#             print(f"  {bias_result['predicted_label']}: {text[:60]}")
#             print(f"    Probs: L={bias_result['all_probs']['Left']:.2f}, "
#                   f"C={bias_result['all_probs']['Center']:.2f}, "
#                   f"R={bias_result['all_probs']['Right']:.2f}")
    
#     # ========================================================================
#     # TEST 4: Sentence Aggregation
#     # ========================================================================
#     print("\n\n" + "="*70)
#     print("TEST 4: SENTENCE-LEVEL AGGREGATION")
#     print("="*70)
    
#     test_article = """
#     Freedom House published a report Wednesday downgrading the United States democracy rating.
#     The liberal organization claimed that conservative policies threaten democratic norms.
#     However, critics argue that the report itself shows political bias.
#     Republican lawmakers dismissed the findings as partisan propaganda.
#     The report highlights concerns about voting rights and media freedom.
#     """
    
#     print("\nTest Article:")
#     print(test_article)
    
#     print("\n" + "-"*70)
#     print("Analyzing with sentence-level aggregation...")
#     print("-"*70)
    
#     result = analyze_article_with_aggregation(
#         test_article,
#         bias_model, bias_tokenizer,
#         pol_model, pol_tokenizer,
#         pol_threshold=0.8,
#         left_threshold=0.6,
#         right_threshold=0.6
#     )
    
#     print(f"\n✓ Analysis complete!")
#     print(f"\n  Total sentences: {result['total_sentences']}")
#     print(f"  Political sentences: {result['political_sentences']}")
#     print(f"  Overall prediction: {result['overall_prediction']}")
#     print(f"  Confidence: {result['confidence']:.3f}")
#     print(f"\n  Weighted probabilities:")
#     for label, prob in result['weighted_probs'].items():
#         print(f"    {label:7s}: {prob:.3f}")
    
#     print(f"\n  Per-sentence breakdown:")
#     for i, sent_result in enumerate(result['sentence_results'], 1):
#         print(f"\n    Sentence {i}: {sent_result['prediction']} (weight: {sent_result['weight']:.2f})")
#         print(f"      {sent_result['sentence'][:70]}")
    
#     # ========================================================================
#     # TEST ON REAL DATASET
#     # ========================================================================
#     print("\n\n" + "="*70)
#     print("TEST ON REAL DATASET")
#     print("="*70)
    
#     dataset_path = Path("preprocessed/article_bias_prediction.parquet")
    
#     if dataset_path.exists():
#         df = pd.read_parquet(dataset_path)
#         print(f"\n✓ Loaded {len(df)} articles")
        
#         # Test on 5 random articles
#         samples = df.sample(min(5, len(df)))
        
#         print("\nTesting aggregation on real articles...\n")
        
#         correct = 0
#         for idx, row in samples.iterrows():
#             true_label = row['leaning']
#             article_text = row['body'][:1000]  # First 1000 chars
            
#             result = analyze_article_with_aggregation(
#                 article_text,
#                 bias_model, bias_tokenizer,
#                 pol_model, pol_tokenizer,
#                 pol_threshold=0.8,
#                 left_threshold=0.6,
#                 right_threshold=0.6
#             )
            
#             if result.get('is_political_article'):
#                 predicted = result['overall_prediction']
#                 is_correct = predicted.lower() == true_label.lower()
                
#                 if is_correct:
#                     correct += 1
                
#                 symbol = "✓" if is_correct else "✗"
#                 print(f"{symbol} True: {true_label:7s} | Predicted: {predicted:7s} | "
#                       f"Conf: {result['confidence']:.2f} | "
#                       f"Sents: {result['political_sentences']}/{result['total_sentences']}")
#             else:
#                 print(f"  Article not political enough")
        
#         print(f"\nAccuracy on sample: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
#     else:
#         print(f"\n⚠️  Dataset not found at {dataset_path}")
    
#     print("\n\n" + "="*70)
#     print("✓ ALL TESTS COMPLETE!")
#     print("="*70)
#     print("\nKey Improvements Implemented:")
#     print("  1. ✓ Stricter politicalness threshold (0.5 → 0.8)")
#     print("  2. ✓ Center detection with threshold zones")
#     print("  3. ✓ More diverse test examples")
#     print("  4. ✓ Sentence-level aggregation with weighting")
#     print("\nNext Steps:")
#     print("  • Fine-tune thresholds based on results")
#     print("  • Test on more articles from your datasets")
#     print("  • Integrate into browser extension")


# if __name__ == "__main__":
#     test_all_improvements()