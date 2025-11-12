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
import ast

MODEL = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
DATA_PATH = "../data/processed/averitec_sample.csv"
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


class PairwiseExpansionDataset(Dataset):
    """
    Expands a CSV row into multiple pairs:
      (claim, evidence_sentence, label_id)
    Better for initially training and fine-tuning the model.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        pairs = []
        for _, row in df.iterrows():
            claim = row["claim"]
            evid_list = ast.literal_eval(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]
            label_id = LABEL_MAP[str(row["label"]).upper()]
            for ev in (evid_list or []):
                pairs.append((claim, ev, label_id))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_pairwise(batch, tokenizer, max_length=256):
    """
    Collate for PairwiseExpansionDataset.
    batch: list of (claim, evidence_sent, label_id)
    Returns tokenized tensors and labels.
    """
    claims, evids, labels = zip(*batch)
    enc = tokenizer(
        list(evids), list(claims),
        padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return enc, labels



# %%

def aggregate(probs, margin=0.1, min_conf=0.7):
    """
    Aggregate per-evidence probabilities into a claim-level prediction.
    This is pretty rudimentary for now.
    """
    probs = np.asarray(probs)
    E, R = probs[:,2].max(), probs[:,0].max()
    print(f"Max SUP: {E:.4f}, Max REF: {R:.4f}")
    if abs(E - R) < margin and max(E, R) >= min_conf: return "NOT ENOUGH INFO"
    if E >= min_conf and E >= R + margin: return "SUPPORTED"
    if R >= min_conf and R >= E + margin: return "REFUTED"
    return "NOT ENOUGH INFO"

ds = ClaimBatchDataset(DATA_PATH)
model_nli = AutoModelForSequenceClassification.from_pretrained(MODEL)
model_nli.eval()
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
    print(f"Claim: {claim}\nEvidences: {evid_list}\nTrue: {classes[true_label]}, Pred: {pred}\n")

