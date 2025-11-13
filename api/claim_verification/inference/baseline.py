# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv (3.12.9)
#     language: python
#     name: python3
# ---

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from torch.utils.data import Subset
import ast

MODEL = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
DATA_PATH_AVERITEC = "../data/processed/averitec_sample.csv"
DATA_PATH_FEVER   = "../data/processed/fever_train_claims_sample.csv"

LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}


# %%
class ClaimBatchDataset(Dataset):
    """
    Each item = (claim: str, evidences: List[str], label_id: int)
    Use for inference + aggregation (run NLI over all evidences, then aggregate).
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df["evidence"] = df["evidence"].apply(
            lambda s: ast.literal_eval(s) if isinstance(s, str) else s
        )
        self.claims = df["claim"].tolist()
        self.evidences = df["evidence"].tolist()
        self.labels = [LABEL_MAP[str(l).upper()] for l in df["label"].tolist()]

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        return self.claims[idx], self.evidences[idx], self.labels[idx]



# %%

def aggregate(probs, margin=0.05, min_conf=0.5):
    """
    Aggregate per-evidence probabilities into a claim-level prediction.
    This is pretty rudimentary for now.
    """
    probs = np.asarray(probs)
    E, R = probs[:,2].max(), probs[:,0].max()
    if abs(E - R) < margin and max(E, R) >= min_conf: return "NOT ENOUGH INFO"
    if E >= min_conf and E >= R + margin: return "SUPPORTED"
    if R >= min_conf and R >= E + margin: return "REFUTED"
    return "NOT ENOUGH INFO"

def run_inference(data_path=DATA_PATH_AVERITEC):
    ds = ClaimBatchDataset(data_path)
    model_nli = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model_nli.eval()
    predictions = []
    true_labels = []
    for claim, evid_list, true_label in ds:
        if not evid_list: 
            continue
        enc = tokenizer(evid_list, [claim]*len(evid_list),
                        padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model_nli(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = aggregate(probs)
        classes = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]
        predictions.append(pred)
        true_labels.append(classes[true_label])
        print(f"Claim: {claim}\nEvidences: {evid_list}\nTrue: {classes[true_label]}\nPredicted: {pred}\n")
    print(classification_report(true_labels, predictions, labels=classes, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, labels=classes)
    plt.show()
    

run_inference(DATA_PATH_FEVER)
    

