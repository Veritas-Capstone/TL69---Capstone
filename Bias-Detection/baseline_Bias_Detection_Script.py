''' 
This script uses Model 4 as the baseline and takes in as input an article, splits it into sentences, feeds each sentence to the model, and aggregates the predicted bias for each sentence.
Todo-> 
1) Replace Naive Sentence splitting with a better algorithm, perhaps one from a real sentence segmentation library to prevent sentences real context being cut off midway 
2) Merge small sentences (connected to part 1) 
3) Introduce a weighted aggregate system instead of majority voting (essentially give less weight to low confidence results), maybe even ignore them comepletely outright 
4) ADVANCED : Weight quotes and counterpoints less as the model might think they are part of the problem when actually they arent 
5) Smoothing across consectuive sentences (CHAT RECOMMENDATION) 
6) 
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
nlp = spacy.load("en_core_web_sm")


# -----------------------
# Sentence Splitting
# -----------------------
def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# -----------------------
# Merge Short Sentences 
# -----------------------
def merge_short_sentences(sentences, min_tokens=30):
    merged = []
    buffer = ""

    for sent in sentences:
        num_tokens = len(sent.split())

        if num_tokens < min_tokens:
            # small sentence → merge into buffer
            buffer += " " + sent
        else:
            # long sentence → flush buffer first
            if buffer.strip():
                merged.append(buffer.strip())
                buffer = ""
            merged.append(sent)

    # leftover buffer
    if buffer.strip():
        merged.append(buffer.strip())

    return merged


# -----------------------
# Load Model Once
# -----------------------
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "matous-volf/political-leaning-deberta-large"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        use_fast=False
    )
    model.eval()
    return model, tokenizer


# -----------------------
# Sentence-Level Prediction
# -----------------------
def baseline_bias_detection(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    return predicted, probs[0]


# -----------------------
# Apply Downweighting Rules
# -----------------------
def compute_weight(sentence, probs, low_conf_threshold=0.45):
    peak = torch.max(probs).item()

    weight = 1.0

    # 1. Downweight low-confidence sentences
    if peak < low_conf_threshold:
        weight *= 0.3   # reduce weight by 70%

    # 2. Downweight quotes / counterpoints
    if sentence.startswith('"') or sentence.startswith("“"):
        weight *= 0.5

    return weight


# -----------------------
# Weighted Aggregation
# -----------------------
def aggregate_bias_scores(probabilities_list, weights):
    """
    Weighted sum of probability vectors.
    """
    weighted_probs = []

    for probs, w in zip(probabilities_list, weights):
        weighted_probs.append(probs * w)

    total = torch.stack(weighted_probs).sum(dim=0)
    idx = torch.argmax(total).item()

    label_map = {0: "Left", 1: "Center", 2: "Right"}
    return label_map[idx], total


# -----------------------
# Main Driver
# -----------------------
def main():
    text = (
        "The Department of Health and Human Services's Office for Civil Rights has released " "guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies " "against refusing to dispense abortion-inducing medication, stipulating that doing so " "is pregnancy discrimination. That includes discrimination based on current pregnancy, " "past pregnancy, potential or intended pregnancy, and medical conditions related to " "pregnancy or childbirth. HHS is committed to ensuring that everyone can access " "healthcare, free of discrimination."
    )

    model, tokenizer = load_model()

    # Split + merge sentences
    sentences = split_into_sentences(text)
    sentences = merge_short_sentences(sentences)

    probability_vectors = []
    weights = []

    for sentence in sentences:
        pred, probs = baseline_bias_detection(model, tokenizer, sentence)
        probability_vectors.append(probs)

        # compute per-sentence weight
        w = compute_weight(sentence, probs)
        weights.append(w)

        print(f"Sentence: {sentence}")
        print(f"Probs: {probs.tolist()}, Weight: {w}")
        print()

    # Aggregate with weights
    overall_bias, totals = aggregate_bias_scores(probability_vectors, weights)

    print("======== FINAL RESULT ========")
    print("Overall Predicted Bias:", overall_bias)
    print("Weighted Probabilities:", totals.tolist())


if __name__ == "__main__":
    main()