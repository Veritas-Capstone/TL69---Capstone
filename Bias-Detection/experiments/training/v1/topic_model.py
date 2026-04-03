"""
Topic selection bias via PMI + speed-optimized explainability.

Part 1 Build BERTopic model + PMI matrix from training data.
Part 2 Fast pipeline with topic bias + language bias + IG explainability.
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
    '-', '\u2013', '\u2014', '\u2581.', '\u2581,', '\u2581!', '\u2581?', '\u2581;', '\u2581:', '\u2581-'
}

# config
BUILD_TOPICS = False
RUN_EVAL = True
DATASET_PATH = "../../datasets/article_bias_prediction.parquet"
ALLSIDES_PATH = "../../datasets/other_data/allsides_data.json"
EXPLAIN_TOP_K = 2
SAMPLE_SIZE = 10000


# part 1: build topic model + PMI matrix (run once offline)
def build_topic_model(dataset_path=DATASET_PATH, output_path="topic_pmi_lookup.json",
                      sample_size=SAMPLE_SIZE, n_topics=50):
    import pandas as pd
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    print("Building topic model + PMI matrix\n")

    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} articles")
    print(f"Label distribution: {dict(df['leaning'].value_counts())}")

    label_map = {'left': 'Left', 'center': 'Center', 'right': 'Right'}
    df['label'] = df['leaning'].map(label_map)
    df = df.dropna(subset=['label', 'body'])

    # stratified sample for efficiency
    if len(df) > sample_size:
        df_sample = df.groupby('label').apply(
            lambda x: x.sample(min(len(x), sample_size // 3), random_state=42)
        ).reset_index(drop=True)
    else:
        df_sample = df

    print(f"Using {len(df_sample)} articles for topic modeling")

    texts = df_sample['body'].apply(lambda x: ' '.join(str(x).split()[:300])).tolist()
    labels = df_sample['label'].tolist()

    print("Fitting BERTopic..")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model, nr_topics=n_topics,
        min_topic_size=20, verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    print(f"Discovered {len(topic_info) - 1} topics")

    # compute PMI
    print("Computing PMI matrix..")
    total = len(topics)
    label_counts = Counter(labels)
    topic_counts = Counter(topics)
    joint_counts = Counter(zip(topics, labels))

    pmi_matrix = {}
    topic_descriptions = {}

    for topic_id in set(topics):
        if topic_id == -1:
            continue
        pmi_matrix[str(topic_id)] = {}
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

    # coverage ratios per topic
    coverage = {}
    for topic_id in set(topics):
        if topic_id == -1:
            continue
        topic_articles = [(t, l) for t, l in zip(topics, labels) if t == topic_id]
        label_dist = Counter(l for _, l in topic_articles)
        total_topic = len(topic_articles)
        coverage[str(topic_id)] = {l: round(label_dist.get(l, 0) / total_topic, 3) for l in LABELS}

    # show most biased topics
    for direction in ['Left', 'Right', 'Center']:
        print(f"\nMost {direction}-skewed topics:")
        sorted_topics = sorted(pmi_matrix.items(), key=lambda x: x[1].get(direction, 0), reverse=True)
        for tid, scores in sorted_topics[:5]:
            desc = topic_descriptions.get(tid, {}).get('label', '?')
            print(f"  Topic {tid}: PMI({direction})={scores[direction]:+.3f}  [{desc}]")

    # save
    lookup = {
        'pmi_matrix': pmi_matrix, 'topic_descriptions': topic_descriptions,
        'coverage': coverage, 'label_distribution': {l: c for l, c in label_counts.items()},
        'total_articles': total, 'num_topics': len(pmi_matrix),
    }
    with open(output_path, 'w') as f:
        json.dump(lookup, f, indent=2)
    print(f"\nSaved PMI lookup to {output_path} ({len(pmi_matrix)} topics)")

    model_path = "topic_model"
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True,
                     save_embedding_model=embedding_model)
    print(f"Saved BERTopic model to {model_path}/")
    return topic_model, lookup


# part 2: topic bias lookup at inference time
class TopicBiasAnalyzer:
    def __init__(self, lookup_path="topic_pmi_lookup.json", model_path="topic_model"):
        print("Loading topic model..")
        with open(lookup_path, 'r') as f:
            self.lookup = json.load(f)
        self.pmi_matrix = self.lookup['pmi_matrix']
        self.descriptions = self.lookup['topic_descriptions']
        self.coverage = self.lookup['coverage']

        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
        print(f"Topic model loaded ({self.lookup['num_topics']} topics)")

    def analyze(self, text):
        truncated = ' '.join(text.split()[:300])
        topics, probs = self.topic_model.transform([truncated])
        topic_id = topics[0]

        if topic_id == -1:
            return {'topic_id': -1, 'topic_label': 'No clear topic',
                    'pmi_scores': {'Left': 0, 'Center': 0, 'Right': 0},
                    'coverage': {'Left': 0.33, 'Center': 0.33, 'Right': 0.33},
                    'selection_bias': None, 'selection_bias_strength': 0}

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
            'pmi_scores': pmi, 'coverage': cov,
            'selection_bias': max_pmi_label if max_pmi_value > 0.3 else None,
            'selection_bias_strength': max_pmi_value if max_pmi_value > 0.3 else 0,
        }


# part 3: IG explainability
def ig_attribution(text, model, tokenizer, target_class=None, n_steps=30):
    from captum.attr import LayerIntegratedGradients

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    embedding_layer = model.deberta.embeddings.word_embeddings

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
        inputs=input_ids, baselines=baseline_ids, target=target_class,
        additional_forward_args=(attention_mask,), n_steps=n_steps,
    )

    attr_scores = attributions.sum(dim=-1).abs()[0]
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    return {
        'predicted_label': LABELS[pred], 'confidence': float(probs[pred]),
        'all_probs': {l: float(probs[i]) for i, l in enumerate(LABELS)},
        'top_tokens': [], 'method': 'prediction_only',
    }


def _build_filtered_token_list(tokens, scores):
    result = []
    for i, (token, score) in enumerate(zip(tokens, scores)):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            continue
        clean = token.replace('##', '').replace('\u2581', '').strip()
        if not clean or clean in PUNCTUATION:
            continue
        if all(c in '.,!?;:\'"()[]<>-\u2013\u2014' for c in clean):
            continue
        if clean.lower() in STOP_WORDS:
            continue
        result.append({'token': clean, 'score': float(score), 'position': i})
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


# part 4: combined pipeline
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
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)[0]
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    political_prob = float(probs[entail_idx])
    return {'is_political': political_prob > 0.5, 'confidence': political_prob}


def batch_check_politicalness(sentences, model, tokenizer, threshold=0.5):
    hypothesis = "This text is about politics."
    inputs = tokenizer([(s, hypothesis) for s in sentences],
                       return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
    label2id = model.config.label2id
    entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
    results = []
    for i, sent in enumerate(sentences):
        conf = float(probs[i, entail_idx])
        results.append({'sentence': sent, 'is_political': conf > threshold, 'confidence': conf})
    return results


def smart_political_filter(sentences, model, tokenizer, headline=None,
                           primary_threshold=0.5, fallback_threshold=0.3, min_sentences=1):
    # cascading fallback: primary threshold -> relaxed -> top-k forced -> full text -> not political
    if not sentences:
        return [], [], {'strategy': 'empty_input', 'filtered': 0, 'total': 0}

    results = batch_check_politicalness(sentences, model, tokenizer, threshold=primary_threshold)
    passed = [(r['sentence'], r['confidence']) for r in results if r['is_political']]

    if len(passed) >= min_sentences:
        return ([s for s, _ in passed], [c for _, c in passed],
                {'strategy': 'primary_threshold', 'passed': len(passed),
                 'total': len(sentences), 'filtered': len(sentences) - len(passed)})

    # relax threshold
    passed_relaxed = [(r['sentence'], r['confidence']) for r in results if r['confidence'] > fallback_threshold]
    if len(passed_relaxed) >= min_sentences:
        return ([s for s, _ in passed_relaxed], [c for _, c in passed_relaxed],
                {'strategy': 'relaxed_threshold', 'passed': len(passed_relaxed),
                 'total': len(sentences), 'filtered': len(sentences) - len(passed_relaxed)})

    # force top-k by confidence
    sorted_results = sorted(results, key=lambda r: r['confidence'], reverse=True)
    top_k = sorted_results[:max(min_sentences, 2)]
    if top_k[0]['confidence'] > 0.1:
        return ([r['sentence'] for r in top_k], [r['confidence'] for r in top_k],
                {'strategy': 'top_k_forced', 'passed': len(top_k),
                 'total': len(sentences), 'filtered': len(sentences) - len(top_k)})

    # full text + headline fallback
    full_text = ' '.join(sentences)
    full_conf = batch_check_politicalness([full_text], model, tokenizer, threshold=0.0)[0]['confidence']
    headline_conf = 0.0
    if headline:
        headline_conf = batch_check_politicalness([headline], model, tokenizer, threshold=0.0)[0]['confidence']

    if full_conf > 0.2 or headline_conf > 0.4:
        return (sentences, [max(full_conf, 0.1)] * len(sentences),
                {'strategy': 'full_text_fallback', 'passed': len(sentences),
                 'total': len(sentences), 'filtered': 0})

    # genuinely not political
    return ([], [], {'strategy': 'genuinely_not_political', 'passed': 0,
                     'total': len(sentences), 'filtered': len(sentences)})


def analyze_article_fast(body, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                         topic_analyzer=None, explain_top_k=2, headline=None):
    start = time.time()

    sentences = split_and_merge(body)
    if not sentences:
        return _empty_result(start)

    # politicalness filter with cascading fallbacks
    political_sents, political_confs, filter_meta = smart_political_filter(
        sentences, pol_model, pol_tokenizer, headline=headline)
    filtered = filter_meta['filtered']

    if not political_sents:
        return _empty_result(start, len(sentences), filtered)

    # batch bias prediction
    bias_inputs = bias_tokenizer(political_sents, return_tensors="pt",
                                  truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        bias_probs = torch.nn.functional.softmax(bias_model(**bias_inputs).logits, dim=-1)

    sentence_results = []
    for i, (sent, pol_conf) in enumerate(zip(political_sents, political_confs)):
        probs = bias_probs[i]
        pred_class = torch.argmax(probs).item()
        all_probs = {l: float(probs[j]) for j, l in enumerate(LABELS)}

        weight = 1.0
        peak = float(probs[pred_class])
        if peak < 0.45: weight *= 0.3
        if pol_conf < 0.5: weight *= pol_conf
        if sent.strip().startswith('"') or sent.strip().startswith('\u201c'): weight *= 0.5
        if len(sent.split()) < 10: weight *= 0.5

        sentence_results.append({
            'text': sent[:100], 'full_text': sent, 'political_conf': pol_conf,
            'predicted_label': LABELS[pred_class], 'confidence': peak,
            'all_probs': all_probs, 'weight': weight, 'top_tokens': [],
        })

    # IG on top-k most biased sentences only
    if explain_top_k > 0:
        scored = [(1.0 - sr['all_probs']['Center'], j) for j, sr in enumerate(sentence_results)]
        scored.sort(reverse=True)
        for _, idx in scored[:explain_top_k]:
            sr = sentence_results[idx]
            attr = ig_attribution(sr['full_text'], bias_model, bias_tokenizer)
            sr['top_tokens'] = attr['top_tokens']
            sr['method'] = 'integrated_gradients'

    # aggregate language bias
    weighted_probs = []
    for sr in sentence_results:
        probs_t = torch.tensor([sr['all_probs'][l] for l in LABELS])
        weighted_probs.append(probs_t * sr['weight'])

    total = torch.stack(weighted_probs).sum(dim=0)
    normalized = total / total.sum() if total.sum() > 0 else total

    language_label = LABELS[torch.argmax(total).item()]
    language_probs = {l: float(normalized[i]) for i, l in enumerate(LABELS)}

    # topic bias
    topic_result = None
    if topic_analyzer:
        topic_result = topic_analyzer.analyze(body)

    return {
        'language_prediction': language_label, 'language_probs': language_probs,
        'topic_result': topic_result,
        'combined_prediction': language_label, 'combined_probs': language_probs.copy(),
        'sentences': sentence_results,
        'total_sentences': len(sentences), 'filtered_sentences': filtered,
        'analyzed_sentences': len(sentence_results),
        'filter_strategy': filter_meta['strategy'],
        'elapsed_seconds': time.time() - start,
    }


def _empty_result(start, total=0, filtered=0):
    return {
        'language_prediction': 'Center', 'language_probs': {'Left': 0, 'Center': 1, 'Right': 0},
        'topic_result': None,
        'combined_prediction': 'Center', 'combined_probs': {'Left': 0, 'Center': 1, 'Right': 0},
        'sentences': [], 'total_sentences': total, 'filtered_sentences': filtered,
        'analyzed_sentences': 0, 'filter_strategy': 'genuinely_not_political',
        'elapsed_seconds': time.time() - start,
    }


# part 5: evaluation
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
                'source': side.get('source', 'Unknown'), 'headline': side.get('headline', ''),
                'body': body, 'bias_detail': bias_detail, 'true_label': ALLSIDES_MAP[bias_detail],
            })
    return articles


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
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    confusion = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for p, g in zip(predictions, ground_truths):
        confusion[g][p] += 1
    return {'accuracy': correct / total if total > 0 else 0, 'macro_f1': macro_f1,
            'correct': correct, 'total': total, 'per_class': class_metrics, 'confusion': confusion}


def print_metrics(metrics, title="Results"):
    print(f"\n{title}")
    print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    for label in LABELS:
        m = metrics['per_class'][label]
        print(f"  {label:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")


def run_eval(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
             topic_analyzer=None, explain_top_k=2):
    preds_lang, truths = [], []
    total_time = 0

    for i, article in enumerate(articles):
        result = analyze_article_fast(
            article['body'], bias_model, bias_tokenizer, pol_model, pol_tokenizer,
            topic_analyzer=topic_analyzer, explain_top_k=explain_top_k,
            headline=article.get('headline', ''))

        total_time += result['elapsed_seconds']
        lang_pred = result['language_prediction']
        true = article['true_label']
        preds_lang.append(lang_pred)
        truths.append(true)

        match = "Y" if lang_pred == true else "X"
        print(f"  [{i+1}/{len(articles)}] {match} True: {true:>7}  Pred: {lang_pred:>7}  "
              f"({result['analyzed_sentences']}/{result['total_sentences']} sents, "
              f"{result['elapsed_seconds']:.2f}s)  {article['source']}")

        if result['topic_result'] and result['topic_result']['topic_id'] != -1:
            tr = result['topic_result']
            print(f"           Topic: {tr['topic_label']}  "
                  f"PMI: L={tr['pmi_scores'].get('Left',0):+.3f} C={tr['pmi_scores'].get('Center',0):+.3f} R={tr['pmi_scores'].get('Right',0):+.3f}")

        # show top explainability tokens
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
            print(f"           Tokens: {', '.join(token_strs)}")

    print(f"\n  Time: {total_time:.1f}s ({total_time/len(articles):.2f}s/article)")

    metrics = compute_metrics(preds_lang, truths)
    print_metrics(metrics, "Language-Only Results")

    if topic_analyzer:
        print_metrics(compute_metrics(preds_lang, truths), "Combined Results")

    return metrics


def load_models(bias_path="../../models/demo_models/bias_detector", pol_path="../../models/demo_models/politicalness_filter"):
    print("Loading models...")
    bias_model = AutoModelForSequenceClassification.from_pretrained(bias_path)
    bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
    bias_model.eval()
    pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
    pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
    pol_model.eval()
    print("Models loaded")
    return bias_model, bias_tokenizer, pol_model, pol_tokenizer


if __name__ == "__main__":
    if BUILD_TOPICS:
        build_topic_model(dataset_path=DATASET_PATH, sample_size=SAMPLE_SIZE)
        print("\nTopic model built. ")

    if RUN_EVAL:
        articles = load_allsides_data(ALLSIDES_PATH)
        print(f"Loaded {len(articles)} articles")
        print(f"Distribution: {dict(Counter(a['true_label'] for a in articles))}")

        bias_model, bias_tokenizer, pol_model, pol_tokenizer = load_models()

        topic_analyzer = None
        if Path("topic_pmi_lookup.json").exists() and Path("topic_model").exists():
            topic_analyzer = TopicBiasAnalyzer()
        else:
            print("\nNo topic model found. \n")

        run_eval(articles, bias_model, bias_tokenizer, pol_model, pol_tokenizer,
                 topic_analyzer=topic_analyzer, explain_top_k=EXPLAIN_TOP_K)
