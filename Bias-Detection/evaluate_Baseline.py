import pandas as pd
from tqdm import tqdm  

# ------ Load Dataset ------
df = pd.read_csv("datasets/allsides_balanced_news_headlines-texts.csv")
df = df[["text", "bias_rating"]]  # Keep only required columns
df = df.dropna()

label_map = {"left": 0, "center": 1, "right": 2}
df["label"] = df["bias_rating"].map(label_map)
print("Dataset loaded with", len(df), "samples.")


# ------ Load Baseline Model 1 (BERT politicalBiasBERT) ------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer_a = AutoTokenizer.from_pretrained("bert-base-cased")
model_a = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

def predict_model_a(text):
    inputs = tokenizer_a(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_a(**inputs).logits
    return torch.argmax(logits.softmax(dim=-1), dim=-1).item()



# ------ Load Baseline Model 2 (RoBERTa peekayitachi/roberta-political-bias) ------
tokenizer_b = AutoTokenizer.from_pretrained("peekayitachi/roberta-political-bias")
model_b = AutoModelForSequenceClassification.from_pretrained("peekayitachi/roberta-political-bias")


def predict_model_b(text):
    inputs = tokenizer_b(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_b(**inputs).logits
    return torch.argmax(logits, dim=1).item()


# ------ Predict and Evaluate ------\
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_a.to(device)
model_b.to(device)


preds_a = []
preds_b = []

for text in tqdm(df["text"]): 
    preds_a.append(predict_model_a(text))
    preds_b.append(predict_model_b(text))


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


y_true = df["label"].tolist()

# Convert true labels to numeric indices (if needed)
label_map = {"left": 0, "center": 1, "right": 2}
if isinstance(y_true[0], str):
    y_true = [label_map[y] for y in y_true]

print("\n=== Model A (BERT politicalBiasBERT) ===")
print("Accuracy:", accuracy_score(y_true, preds_a))
print(classification_report(y_true, preds_a, target_names=["left", "center", "right"]))
print(confusion_matrix(y_true, preds_a))
cm_a = confusion_matrix(y_true, preds_a)

print("\n=== Model B (RoBERTa political bias) ===")
print("Accuracy:", accuracy_score(y_true, preds_b))
print(classification_report(y_true, preds_b, target_names=["left", "center", "right"]))
print(confusion_matrix(y_true, preds_b))
cm_b = confusion_matrix(y_true, preds_b)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Confusion Matrix Model A ---
plt.figure(figsize=(6,4))
sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Left","Center","Right"], yticklabels=["Left","Center","Right"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Model A (politicalBiasBERT)")
plt.show()

# --- Confusion Matrix Model B ---
plt.figure(figsize=(6,4))
sns.heatmap(cm_b, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Left","Center","Right"], yticklabels=["Left","Center","Right"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Model B (RoBERTa political-bias)")
plt.show()

# --- Precision / Recall Bar Charts ---
prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(y_true, preds_a, labels=[0,1,2])
prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_true, preds_b, labels=[0,1,2])

metrics_a = pd.DataFrame({"precision": prec_a, "recall": rec_a, "f1": f1_a}, index=["Left","Center","Right"])
metrics_b = pd.DataFrame({"precision": prec_b, "recall": rec_b, "f1": f1_b}, index=["Left","Center","Right"])

metrics_a.plot(kind="bar", figsize=(7,4), ylim=(0,1), title="Model A Metrics by Class")
plt.show()

metrics_b.plot(kind="bar", figsize=(7,4), ylim=(0,1), title="Model B Metrics by Class")
plt.show()

# --- Predicted vs True Class Distribution ---
plt.figure(figsize=(6,4))
plt.hist([preds_a, preds_b, y_true], label=["Model A predictions","Model B predictions","True labels"], bins=[-0.5,0.5,1.5,2.5])
plt.xticks([0,1,2], ["Left","Center","Right"])
plt.title("Class Distribution Comparison")
plt.legend()
plt.show()