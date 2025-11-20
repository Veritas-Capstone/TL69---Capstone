# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import os
import json
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from .eval_helpers import collate_multi_evidence, PairwiseExpansionDataset, collate_pairwise

# Base HF model (same as training)
MODEL = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

DATA_SET = "averitec"
LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}


def eval_averitec(
    init_weights="hf",
    data_set=DATA_SET,
    data_path=None,
    batch_size=8,
    max_length=256,
    output_root="claim_verification/eval_metrics",
):
    """
    Evaluate RoBERTa on the *entire* AveriTeC dataset.

    Parameters
    ----------
    init_weights : str
        "hf" -> load base HF roberta-large-mnli,
        any other string -> treated as path to a .pt checkpoint to load.
    data_set : str
        Dataset name (just used to name output dirs).
    data_path : str or None
        Path to the CSV for AveriTeC. If None, uses ../data/processed/{data_set}.csv
    """

    # Figure out package root: .../api/claim_verification
    pkg_root = Path(__file__).resolve().parents[1]
    default_data_dir = pkg_root / "data" / "processed"

    if data_path is None:
        # If you want full dataset: f"{data_set}.csv"
        # If you specifically want the 20% split: f"{data_set}_20.csv"
        data_path = default_data_dir / f"{data_set}_20.csv"

    print(f"[EVAL] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Same dataset object as training, but we use ALL examples (no split)
    pair_ds = PairwiseExpansionDataset(df, LABEL_MAP)
    # print(f"[EVAL] Total pairwise examples: {len(pair_ds)}")
    print(f"[EVAL] Total pairwise examples: {len(df)}")

    # Multiple Evidence input
    eval_loader = DataLoader(
        df.to_dict("records"),  # or df.to_dict("records") if you prefer dicts over Series
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_multi_evidence(
            batch,
            tokenizer,
            max_length=max_length,
            label_map=LABEL_MAP,   # <-- important
        ),
    )
    # Pairwise
    # eval_loader = DataLoader(
    #     pair_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=lambda batch: collate_pairwise(batch, tokenizer, max_length=max_length),
    # )

    num_labels = len(LABEL_MAP)
    print("Number of Labels:", num_labels)

    # ---- Load model ----
    print("[EVAL] Loading model:")
    print(f"       MODEL = {MODEL}")
    print(f"       init_weights = {init_weights!r}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=num_labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device Used for Evaluation:", device)

    # If init_weights is not "hf", treat it as a checkpoint path
    if init_weights != "hf":
        ckpt_path = init_weights
        print(f"[EVAL] Loading checkpoint weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    all_preds = []
    all_labels = []

    # i = 0
    # check_every = num_labels // 20
    with torch.no_grad():
        for enc, labels in tqdm(eval_loader, desc="Evaluating AveriTeC", unit="batch"):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # if i % check_every == 0:
            #     print(f"Iteration: {i}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print("\n[EVAL] Classification report (per class):")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=list(LABEL_MAP.keys()),
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print("[EVAL] Confusion matrix:\n", cm)

    # ---- Save metrics & confusion matrix ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make model tag for naming outputs
    if init_weights == "hf":
        model_tag = "hf_baseline"
    else:
        # e.g. ../models/averitec/latest.pt -> latest
        base_name = os.path.basename(init_weights)
        model_tag = os.path.splitext(base_name)[0]

    out_dir = os.path.join(output_root, f"{data_set}_{model_tag}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "model_name": MODEL,
        "init_weights": init_weights,
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "num_examples": int(len(all_labels)),
        "label_map": LABEL_MAP,
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[EVAL] Metrics saved to: {metrics_path}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(LABEL_MAP.keys()),
    )
    disp.plot(cmap="Blues")
    plt.title(f"AveriTeC Confusion Matrix ({model_tag})")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[EVAL] Confusion matrix saved to: {cm_path}")

    return {
        "out_dir": out_dir,
        "metrics_path": metrics_path,
        "confusion_matrix_path": cm_path,
    }


if __name__ == "__main__":
    # Default: evaluate HF baseline on full AveriTeC
    eval_averitec()

