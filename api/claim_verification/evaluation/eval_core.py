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

from .eval_helpers import collate_multi_evidence
from api.claim_verification.training.training_joint_helpers import (
    JointEvidenceDataset,
    collate_joint_batch,
)
from api.claim_verification.models.claim_evidence_attention import ClaimEvidenceAttentionModel


def _looks_like_attention_checkpoint(state_dict):
    return any(k.startswith("encoder.") or k.startswith("W_c") for k in state_dict.keys())


def eval_claim_model(
    init_weights="hf",
    data_set="averitec",
    data_path=None,
    batch_size=8,
    max_length=256,
    max_evidence=5,
    use_attention_model=None,
    output_root=None,
    num_heads=4,
    model_name="FacebookAI/roberta-large-mnli",
    label_map=None,
    data_dir=None,
    models_dir=None,
    eval_out_dir=None,
):
    if label_map is None:
        label_map = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}

    if data_path is None:
        if data_dir is None:
            raise ValueError("data_path is required when data_dir is not provided")
        data_path = Path(data_dir) / f"{data_set}.csv"
    data_path = Path(data_path)

    if output_root is None:
        if eval_out_dir is None:
            raise ValueError("output_root is required when eval_out_dir is not provided")
        output_root = eval_out_dir
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[EVAL] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # ---- Resolve init weights ----
    if init_weights != "hf":
        ckpt_path = Path(init_weights)
        if not ckpt_path.is_absolute():
            if models_dir is None:
                ckpt_path = ckpt_path.resolve()
            else:
                ckpt_path = (Path(models_dir) / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt_path = None
        ckpt_state = None

    # Auto-detect architecture if not specified
    if use_attention_model is None:
        use_attention_model = ckpt_state is not None and _looks_like_attention_checkpoint(ckpt_state)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[EVAL] Total examples: {len(df)}")
    if use_attention_model:
        eval_ds = JointEvidenceDataset(df, label_map, max_evidence=max_evidence)
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
                label_map=label_map,
            ),
        )

    num_labels = len(label_map)

    print("[EVAL] Loading model:")
    print(f"       MODEL = {model_name}")
    print(f"       init_weights = {init_weights!r}")
    print(f"       architecture = {'attention' if use_attention_model else 'pairwise'}")

    if use_attention_model:
        encoder = AutoModel.from_pretrained(model_name)
        model = ClaimEvidenceAttentionModel(
            encoder,
            hidden_dim=encoder.config.hidden_size,
            num_labels=num_labels,
            num_heads=num_heads,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device Used for Evaluation:", device)

    if ckpt_path:
        print(f"[EVAL] Loading checkpoint weights from: {ckpt_path}")
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

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

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
        target_names=list(label_map.keys()),
        zero_division=0,
        labels=list(range(len(label_map))),
    )
    print(report_text)
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=list(label_map.keys()),
        zero_division=0,
        labels=list(range(len(label_map))),
        output_dict=True,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(label_map))))
    print("[EVAL] Confusion matrix:\n", cm)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if init_weights == "hf":
        model_tag = "hf_baseline"
    else:
        base_name = os.path.basename(str(ckpt_path))
        model_tag = os.path.splitext(base_name)[0]

    out_dir = output_root / f"{data_set}_{model_tag}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": model_name,
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
        "label_map": label_map,
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
        display_labels=list(label_map.keys()),
    )
    disp.plot(cmap="Blues")
    plt.title(f"{data_set} Confusion Matrix ({model_tag})")
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
