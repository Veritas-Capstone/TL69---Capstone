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

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import ast
import timeit
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "FacebookAI/roberta-large-mnli"
DATA_PATH_AVERITEC = "../data/processed/averitec_20.csv"
DATA_PATH_FEVER = "../data/processed/fever_train_claims_20.csv"

LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}
LABELS = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]
PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PKG_ROOT / "models"

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
    E, R = probs[:, 2].max(), probs[:, 0].max()
    if abs(E - R) < margin and max(E, R) >= min_conf:
        return "NOT ENOUGH INFO"
    if E >= min_conf and E >= R + margin:
        return "SUPPORTED"
    if R >= min_conf and R >= E + margin:
        return "REFUTED"
    return "NOT ENOUGH INFO"

def resolve_latest_checkpoint(dataset: str) -> Path:
    """
    Return the path to the latest checkpoint for a given dataset split
    (expects models/<dataset>/latest.pt).
    """
    ckpt = MODELS_DIR / dataset / "latest.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found for dataset '{dataset}' at {ckpt}")
    return ckpt


def list_available_checkpoints():
    """
    Return a mapping of dataset -> latest checkpoint path for all available runs.
    """
    if not MODELS_DIR.exists():
        return {}
    latest = {}
    for dataset_dir in MODELS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        candidate = dataset_dir / "latest.pt"
        if candidate.exists():
            latest[dataset_dir.name] = candidate
    return latest


def load_claim_verifier(
    model_name: str = MODEL,
    state_dict_path: Union[str, Path, None] = None,
    map_location: Union[str, torch.device, None] = "cpu",
):
    """
    Load tokenizer + model. If `state_dict_path` is provided, initialize from
    the given .pt checkpoint (state_dict) after loading the base HF weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if state_dict_path:
        ckpt_path = Path(state_dict_path)
        if not ckpt_path.is_absolute():
            ckpt_path = ckpt_path.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(state_dict)

    model.eval()
    return tokenizer, model


def verify_claim(
    claim: str,
    evidence_list: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    max_length: int = 256,
) -> Tuple[str, dict]:
    """
    Run the NLI model over a claim and all evidence concatenated together.
    This replaces per-evidence aggregation with a single pass over the joined text.
    """
    evidence_list = [e.strip() for e in evidence_list if e and e.strip()]
    if not evidence_list:
        empty_probs = torch.tensor([0.0, 0.0, 1.0])
        return "NOT ENOUGH INFO", {
            LABELS[i]: float(empty_probs[i]) for i in range(len(LABELS))
        }

    # Join evidence sentences so the model reasons over the combined context once.
    combined_evidence = " ".join(evidence_list)
    enc = tokenizer(
        combined_evidence,
        claim,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    label_idx = torch.argmax(probs).item()
    label = LABELS[label_idx]
    probs_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return label, probs_dict


def run_inference(
    data_path: str = DATA_PATH_AVERITEC,
    tokenizer: AutoTokenizer = None,
    model_nli: AutoModelForSequenceClassification = None,
):
    timer_start = timeit.default_timer()

    if tokenizer is None or model_nli is None:
        tokenizer, model_nli = load_claim_verifier()

    ds = ClaimBatchDataset(data_path)
    predictions = []
    true_labels = []
    for i, (claim, evid_list, true_label) in enumerate(ds, start=1):
        if not evid_list:
            continue

        pred_label, _ = verify_claim(
            claim,
            evid_list,
            tokenizer=tokenizer,
            model=model_nli,
        )
        predictions.append(pred_label)
        true_labels.append(LABELS[true_label])
        print(
            f"Claim: {claim}\nEvidences: {evid_list}\n"
            f"True: {LABELS[true_label]}\nPredicted: {pred_label}\n"
        )

        if i % 50 == 0:
            print(f"Iteration: {i}/{len(ds)}")
            print(f"Time Elapsed: {timeit.default_timer() - timer_start}")

    print(classification_report(true_labels, predictions, labels=LABELS, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, labels=LABELS)
    plt.show()

    end = timeit.default_timer()
    print(f"Time Elapsed: {end - timer_start}")


if __name__ == "__main__":
    tok, model = load_claim_verifier()
    run_inference(DATA_PATH_AVERITEC, tokenizer=tok, model_nli=model)
