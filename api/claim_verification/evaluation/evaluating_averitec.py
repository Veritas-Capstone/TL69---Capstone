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
import re
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

from .eval_helpers import (
    collate_multi_evidence,
    PairwiseExpansionDataset,
    collate_pairwise,
    parse_evidence_field,
)
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
    nei_threshold=None,
    nei_margin=None,
    fever_score=False,
    gold_data_path=None,
    run_name=None,
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

    if gold_data_path is None and fever_score and str(data_path).endswith("_er.csv"):
        candidate = Path(str(data_path).replace("_er.csv", ".csv"))
        if candidate.exists():
            gold_data_path = candidate
        else:
            print(f"[EVAL] Warning: gold_data_path not found at {candidate}")

    if gold_data_path is not None:
        gold_data_path = Path(gold_data_path)
        if not gold_data_path.exists():
            raise FileNotFoundError(f"Gold data not found at {gold_data_path}")

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

    # If checkpoint encodes multi-head count, align num_heads automatically
    if use_attention_model and ckpt_state is not None and "W_c" in ckpt_state:
        try:
            ckpt_heads = int(ckpt_state["W_c"].shape[0])
            if ckpt_heads != num_heads:
                print(f"[EVAL] Overriding num_heads={num_heads} -> {ckpt_heads} to match checkpoint")
                num_heads = ckpt_heads
        except Exception:
            pass

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
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            if nei_threshold is not None or nei_margin is not None:
                # Apply NEI post-processing based on confidence/margin
                nei_id = LABEL_MAP["NOT ENOUGH INFO"]
                top2 = torch.topk(probs, k=2, dim=-1)
                top1_prob = top2.values[:, 0]
                top2_prob = top2.values[:, 1]
                margin = top1_prob - top2_prob
                mask = torch.zeros_like(top1_prob, dtype=torch.bool)
                if nei_threshold is not None:
                    mask |= top1_prob < float(nei_threshold)
                if nei_margin is not None:
                    mask |= margin < float(nei_margin)
                preds = torch.where(mask, torch.tensor(nei_id, device=preds.device), preds)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # if i % check_every == 0:
            #     print(f"Iteration: {i}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    def _normalize_text(text):
        text = re.sub(r"\s+", " ", str(text)).strip().lower()
        return text

    def _evidence_match(gold_list, pred_list):
        if not gold_list or not pred_list:
            return False
        gold_norm = [_normalize_text(x) for x in gold_list if isinstance(x, str) and x.strip()]
        pred_norm = [_normalize_text(x) for x in pred_list if isinstance(x, str) and x.strip()]
        if not gold_norm or not pred_norm:
            return False
        for g in gold_norm:
            for p in pred_norm:
                if g in p or p in g:
                    return True
        return False

    fever_score_value = None
    evidence_hit_rate = None

    if fever_score or gold_data_path is not None:
        gold_df = None
        if gold_data_path is not None:
            gold_df = pd.read_csv(gold_data_path)

        # Pre-parse predicted evidence from eval data (respect max_evidence)
        pred_evidence = [
            parse_evidence_field(row.get("evidence"))[:max_evidence]
            for _, row in df.iterrows()
        ]

        gold_evidence = []
        if gold_df is None:
            gold_evidence = [
                parse_evidence_field(row.get("evidence")) for _, row in df.iterrows()
            ]
        else:
            # Build claim+label -> list of evidence lists
            lookup = {}
            for _, row in gold_df.iterrows():
                key = (row.get("claim"), row.get("label"))
                lookup.setdefault(key, []).append(parse_evidence_field(row.get("evidence")))
            for _, row in df.iterrows():
                key = (row.get("claim"), row.get("label"))
                items = lookup.get(key)
                if items:
                    gold_evidence.append(items.pop(0))
                else:
                    gold_evidence.append([])

        nei_id = LABEL_MAP["NOT ENOUGH INFO"]
        fever_hits = 0
        evidence_hits = 0
        evidence_total = 0
        for idx in range(len(all_labels)):
            gold_label = int(all_labels[idx])
            pred_label = int(all_preds[idx])
            if gold_label == nei_id:
                evidence_ok = True
            else:
                evidence_ok = _evidence_match(gold_evidence[idx], pred_evidence[idx])
                evidence_total += 1
                if evidence_ok:
                    evidence_hits += 1
            if pred_label == gold_label and evidence_ok:
                fever_hits += 1

        fever_score_value = float(fever_hits / len(all_labels)) if len(all_labels) else 0.0
        evidence_hit_rate = (
            float(evidence_hits / evidence_total) if evidence_total else 0.0
        )
        print(f"[EVAL] FEVER score: {fever_score_value:.4f}")
        print(f"[EVAL] Evidence hit rate (non-NEI): {evidence_hit_rate:.4f}")

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

    if run_name is None:
        run_name = data_set
    out_dir = output_root / f"{run_name}_{model_tag}_{run_id}"
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
        "fever_score": fever_score_value,
        "evidence_hit_rate": evidence_hit_rate,
        "gold_data_path": str(gold_data_path) if gold_data_path is not None else None,
        "run_name": run_name,
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
