'''
This script uses Model 4 as the baseline and takes in as input the article, splits them up into sentences and feeds them to the model, the aggregate of the bias scores makes up the actual bias of the article or selected passage of text. Right now the aggregation is done by taking the mode of the sentence bias scores (predominant bias in the article/passage) and the sentence breakup is done using a naive method of splitting by periods. We use sentence-based segmentation with a small merge threshold — e.g., merge sentences <30 tokens with neighbors to avoid fragments and then at the end we take a weighted aggregate to get a general bias for the article/passage ie give more weight to a bias when the model is more confident about it and less weight to neutral sentences
'''

def SplitIntoSentences(text):
    sentences = text.split('. ')
    # Add the period back to each sentence except the last if it was removed
    sentences = [s if s.endswith('.') else s + '.' for s in sentences[:-1]] + [sentences[-1]]
    return sentences


def baseline_bias_detection(sentence):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-deberta-large")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)

    text = sentence

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted = torch.argmax(logits, dim=1).item()

    label_map = {0: "Left", 1: "Center", 2: "Right"}
    print("Predicted Bias:", label_map[predicted])

    # # Show confidence scores
    # probs = torch.nn.functional.softmax(logits, dim=1)
    # for label_id, label_name in label_map.items():
    #     print(f"{label_name}: {probs[0, label_id].item():.4f}")

    return label_map[predicted]


def AggregateBiasScores(bias_scores):
    from collections import Counter
    count = Counter(bias_scores)
    predominant_bias = count.most_common(1)[0][0]
    return predominant_bias


def main():
    text = "The Department of Health and Human Services's Office for Civil Rights has released guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies against refusing to dispense abortion-inducing medication, stipulating that doing so is pregnancy discrimination. That includes discrimination based on current pregnancy, past pregnancy, potential or intended pregnancy, and medical conditions related to pregnancy or childbirth. HHS is committed to ensuring that everyone can access healthcare, free of discrimination,"

    sentences = SplitIntoSentences(text)
    bias_scores = []
    for sentence in sentences:
        bias = baseline_bias_detection(sentence)
        bias_scores.append(bias)

    overall_bias = AggregateBiasScores(bias_scores)
    print("Overall Predicted Bias for the article/passage:", overall_bias)