import pandas as pd
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# !Change this depending on which dataset you want to test on
# ------ Load Dataset ------

# allsides
# df = pd.read_csv("datasets/allsides_balanced_news_headlines-texts.csv")
# df = df[["text", "bias_rating"]].dropna()

# babe
df = pd.read_excel("datasets/final_labels_SG2.xlsx")
df = df[["text", "type"]].dropna() 

label_map = {"left": 0, "center": 1, "right": 2}
df["label"] = df["type"].map(label_map)
print("Dataset loaded with", len(df), "samples.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load models
tokenizer_a = AutoTokenizer.from_pretrained("bert-base-cased")
model_a = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT").to(device)

tokenizer_b = AutoTokenizer.from_pretrained("peekayitachi/roberta-political-bias")
model_b = AutoModelForSequenceClassification.from_pretrained("peekayitachi/roberta-political-bias").to(device)

tokenizer_c = AutoTokenizer.from_pretrained("launch/POLITICS")
model_c = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-politics").to(device)

tokenizer_d = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
model_d = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-deberta-large").to(device)

# Prediction Functions
def predict_model_a(text):
    inputs = tokenizer_a(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_a(**inputs).logits
    return torch.argmax(logits.softmax(dim=-1), dim=-1).item()


def predict_model_b(text):
    inputs = tokenizer_b(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_b(**inputs).logits
    return torch.argmax(logits, dim=1).item()


def predict_model_c(text):
    inputs = tokenizer_c(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_c(**inputs).logits
    return torch.argmax(logits, dim=1).item()

def predict_model_d(text):
    inputs = tokenizer_d(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_d(**inputs).logits
    return torch.argmax(logits, dim=1).item()

# ------ Predict and Evaluate ------\
preds_a, preds_b, preds_c, preds_d = [], [], [], []

for text in tqdm(df["text"]):
    preds_a.append(predict_model_a(text))
    preds_b.append(predict_model_b(text))
    preds_c.append(predict_model_c(text))
    preds_d.append(predict_model_d(text))


# Eval
y_true = df["label"].tolist()

print("\n=== Model A (BERT politicalBiasBERT) ===")
print("Accuracy:", accuracy_score(y_true, preds_a))
print(classification_report(y_true, preds_a, target_names=["left", "center", "right"]))
cm_a = confusion_matrix(y_true, preds_a)


print("\n=== Model B (RoBERTa political bias) ===")
print("Accuracy:", accuracy_score(y_true, preds_b))
print(classification_report(y_true, preds_b, target_names=["left", "center", "right"]))
cm_b = confusion_matrix(y_true, preds_b)


print("\n=== Model C (RoBERTa matous-volf) ===")
print("Accuracy:", accuracy_score(y_true, preds_c))
print(classification_report(y_true, preds_c, target_names=["left", "center", "right"]))
cm_c = confusion_matrix(y_true, preds_c)

print("\n=== Model D (DeBERTa V3 Large matous-volf) ===")
print("Accuracy:", accuracy_score(y_true, preds_d))
print(classification_report(y_true, preds_d, target_names=["left", "center", "right"]))
cm_d = confusion_matrix(y_true, preds_d)

# Metrics
# Overall accuracy
acc_a = accuracy_score(y_true, preds_a)
acc_b = accuracy_score(y_true, preds_b)
acc_c = accuracy_score(y_true, preds_c)
acc_d = accuracy_score(y_true, preds_d)

# Per-class P/R/F1
prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(y_true, preds_a, labels=[0,1,2], zero_division=0)
prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_true, preds_b, labels=[0,1,2], zero_division=0)
prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_true, preds_c, labels=[0,1,2], zero_division=0)
prec_d, rec_d, f1_d, _ = precision_recall_fscore_support(y_true, preds_d, labels=[0,1,2], zero_division=0)

# Macro averages (for concise model vs model comparison)
macro_prec_a, macro_rec_a, macro_f1_a, _ = precision_recall_fscore_support(y_true, preds_a, average="macro", zero_division=0)
macro_prec_b, macro_rec_b, macro_f1_b, _ = precision_recall_fscore_support(y_true, preds_b, average="macro", zero_division=0)
macro_prec_c, macro_rec_c, macro_f1_c, _ = precision_recall_fscore_support(y_true, preds_c, average="macro", zero_division=0)
macro_prec_d, macro_rec_d, macro_f1_d, _ = precision_recall_fscore_support(y_true, preds_d, average="macro", zero_division=0)


# Graphs:
f1_df = pd.DataFrame({
    "Model A": f1_a,
    "Model B": f1_b,
    "Model C": f1_c,
    "Model D": f1_d
}, index=["Left", "Center", "Right"])
f1_df.plot(kind="bar", figsize=(8,5), ylim=(0,1), title="F1 Score by Class")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()

metrics_df = pd.DataFrame({
    "A Precision": prec_a, "A Recall": rec_a, "A Accuracy": [acc_a, acc_a, acc_a],
    "B Precision": prec_b, "B Recall": rec_b, "B Accuracy": [acc_b, acc_b, acc_b],
    "C Precision": prec_c, "C Recall": rec_c, "C Accuracy": [acc_c, acc_c, acc_c],
    "D Precision": prec_d, "D Recall": rec_d, "D Accuracy": [acc_d, acc_d, acc_d],
}, index=["Left", "Center", "Right"])
metrics_df.plot(kind="bar", figsize=(12,6), ylim=(0,1), title="Precision and Recall by Class")
plt.ylabel("Score")
plt.xlabel("Class")
plt.legend(ncol=4, fontsize=9)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14,12))
sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", ax=axes[0,0], xticklabels=["Left","Center","Right"], yticklabels=["Left","Center","Right"])
axes[0,0].set_title("Model A (BERT)")
axes[0,0].set_xlabel("Predicted")
axes[0,0].set_ylabel("True")

sns.heatmap(cm_b, annot=True, fmt="d", cmap="Reds", ax=axes[0,1], xticklabels=["Left","Center","Right"], yticklabels=["Left","Center","Right"])
axes[0,1].set_title("Model B (RoBERTa)")
axes[0,1].set_xlabel("Predicted")

sns.heatmap(cm_c, annot=True, fmt="d", cmap="Greens", ax=axes[1,0], xticklabels=["Left","Center","Right"], yticklabels=["Left","Center","Right"])
axes[1,0].set_title("Model C (RoBERTa matous-volf)")
axes[1,0].set_xlabel("Predicted")
axes[1,0].set_ylabel("True")

sns.heatmap(cm_d, annot=True, fmt="d", cmap="Purples", ax=axes[1,1], xticklabels=["Left","Center","Right"], yticklabels=["Left","Center","Right"])
axes[1,1].set_title("Model D (DeBERTa V3 Large)")
axes[1,1].set_xlabel("Predicted")

plt.suptitle("Confusion Matrices for All Models")
plt.tight_layout()
plt.show()

# --- Predicted vs True Class Distribution ---
plt.figure(figsize=(6,4))
plt.hist([preds_a, preds_b, preds_c, preds_d, y_true], label=["Model A","Model B","Model C","Model D","True"], bins=[-0.5,0.5,1.5,2.5], alpha=0.75)
plt.xticks([0,1,2], ["Left","Center","Right"])
plt.title("Class Distribution Comparison")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()