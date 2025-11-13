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
def SplitIntoSentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


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

    label_map = {0: "Left", 1: "Center", 2: "Right"}

    print(f"Sentence: {sentence}")
    print("Predicted Bias:", label_map[predicted])
    print("Probabilities:", probs[0].tolist(), "\n")

    return label_map[predicted], probs[0]


# -----------------------
# Weighted Aggregate
# -----------------------
def AggregateBiasScores(probabilities_list):
    """
    probabilities_list = [tensor([0.8, 0.1, 0.1]), tensor([0.2,0.3,0.5]), ...]
    We sum all probability vectors and choose the highest total.
    """
    total = torch.stack(probabilities_list).sum(dim=0)
    idx = torch.argmax(total).item()

    label_map = {0: "Left", 1: "Center", 2: "Right"}
    return label_map[idx], total


# -----------------------
# Main Driver
# -----------------------
def main():
    text = (
        "The Department of Health and Human Services's Office for Civil Rights has released "
        "guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies "
        "against refusing to dispense abortion-inducing medication, stipulating that doing so "
        "is pregnancy discrimination. That includes discrimination based on current pregnancy, "
        "past pregnancy, potential or intended pregnancy, and medical conditions related to "
        "pregnancy or childbirth. HHS is committed to ensuring that everyone can access "
        "healthcare, free of discrimination."
    )

    model, tokenizer = load_model()

    sentences = SplitIntoSentences(text)

    probability_vectors = []

    for sentence in sentences:
        _, probs = baseline_bias_detection(model, tokenizer, sentence)
        probability_vectors.append(probs)

    overall_bias, totals = AggregateBiasScores(probability_vectors)

    print("======== FINAL RESULT ========")
    print("Overall Predicted Bias:", overall_bias)
    print("Aggregated Probabilities:", totals.tolist())


main()
