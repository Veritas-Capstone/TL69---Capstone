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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from .eval_helpers import collate_multi_evidence, PairwiseExpansionDataset, collate_pairwise
from api.claim_verification.training.training_joint_helpers import (
    JointEvidenceDataset,
    collate_joint_batch,
)
from api.claim_verification.models.claim_evidence_attention import (
    ClaimEvidenceAttentionModel,
    upgrade_attention_state_dict,
)

# Base HF model (same as training)
MODEL = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

DATA_SET = "averitec"
LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}
PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_ROOT / "data" / "processed"
EVAL_OUT_DIR = PKG_ROOT / "eval_metrics"
MODELS_DIR = PKG_ROOT / "models"


def _looks_like_attention_checkpoint(state_dict):
    """
    Heuristic: attention checkpoints save encoder.* keys; HF classifiers use roberta.*.
    """
    return any(k.startswith("encoder.") or k.startswith("W_c") for k in state_dict.keys())


def eval_averitec(
    init_weights="hf",
    data_set=DATA_SET,
    data_path=None,
    batch_size=8,
    max_length=256,
    max_evidence=5,
    use_attention_model=None,
    output_root=None,
    num_heads=4,
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
        Path to the CSV for AveriTeC. If None, uses <pkg>/data/processed/{data_set}.csv
    use_attention_model : bool | None
        If True, evaluate the attention model (JointEvidenceDataset + ClaimEvidenceAttentionModel).
        If None, auto-detect based on checkpoint keys.
    """

    if data_path is None:
        data_path = DATA_DIR / f"{data_set}.csv"
    data_path = Path(data_path)

    if output_root is None:
        output_root = EVAL_OUT_DIR
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[EVAL] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # ---- Resolve init weights ----
    if init_weights != "hf":
        ckpt_path = Path(init_weights)
        if not ckpt_path.is_absolute():
            ckpt_path = (MODELS_DIR / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt_path = None
        ckpt_state = None

    # Auto-detect architecture if not specified
    if use_attention_model is None:
        use_attention_model = ckpt_state is not None and _looks_like_attention_checkpoint(ckpt_state)

    # Dataset + loader based on architecture
    print(f"[EVAL] Total examples: {len(df)}")
    if use_attention_model:
        eval_ds = JointEvidenceDataset(df, LABEL_MAP, max_evidence=max_evidence)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_joint_batch(
                batch,
                tokenizer,
                max_length=max_length,
            ),
        )
    else:
        eval_loader = DataLoader(
            df.to_dict("records"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_multi_evidence(
                batch,
                tokenizer,
                max_length=max_length,
                label_map=LABEL_MAP,
            ),
        )

    num_labels = len(LABEL_MAP)

    # ---- Load model ----
    print("[EVAL] Loading model:")
    print(f"       MODEL = {MODEL}")
    print(f"       init_weights = {init_weights!r}")
    print(f"       architecture = {'attention' if use_attention_model else 'pairwise'}")

    if use_attention_model:
        encoder = AutoModel.from_pretrained(MODEL)
        model = ClaimEvidenceAttentionModel(
            encoder,
            hidden_dim=encoder.config.hidden_size,
            num_labels=num_labels,
            num_heads=num_heads,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=num_labels,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device Used for Evaluation:", device)

    if ckpt_path:
        print(f"[EVAL] Loading checkpoint weights from: {ckpt_path}")
        if use_attention_model:
            ckpt_state = upgrade_attention_state_dict(ckpt_state, num_heads=num_heads)
            missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
            if missing:
                print(f"[EVAL] Missing keys (ignored): {len(missing)}")
            if unexpected:
                print(f"[EVAL] Unexpected keys (ignored): {len(unexpected)}")
        else:
            model.load_state_dict(ckpt_state)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {data_set}"):
            if use_attention_model:
                claim_enc, ev_enc, evid_mask, labels = batch
                claim_enc = {k: v.to(device) for k, v in claim_enc.items()}
                ev_enc = {k: v.to(device) for k, v in ev_enc.items()}
                evid_mask = evid_mask.to(device)
                labels = labels.to(device)
                outputs = model(claim_enc, ev_enc, evid_mask)
                logits = outputs["logits"]
            else:
                enc, labels = batch
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
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
        labels=[0, 1, 2],
    )
    print(report_text)
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
        labels=[0, 1, 2],
        output_dict=True,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print("[EVAL] Confusion matrix:\n", cm)

    # ---- Save metrics & confusion matrix ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make model tag for naming outputs
    if init_weights == "hf":
        model_tag = "hf_baseline"
    else:
        base_name = os.path.basename(str(ckpt_path))
        model_tag = os.path.splitext(base_name)[0]

    out_dir = output_root / f"{data_set}_{model_tag}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": MODEL,
        "init_weights": str(init_weights),
        "use_attention_model": use_attention_model,
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "num_examples": int(len(all_labels)),
        "label_map": LABEL_MAP,
        "classification_report": report_dict,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[EVAL] Metrics saved to: {metrics_path}")

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"[EVAL] Classification report saved to: {report_path}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(LABEL_MAP.keys()),
        # labels=[0, 1, 2],
    )
    disp.plot(cmap="Blues")
    plt.title(f"AveriTeC Confusion Matrix ({model_tag})")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
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
