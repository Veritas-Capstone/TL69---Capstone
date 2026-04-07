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
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import ast
import argparse
import timeit
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

from api.claim_verification.models.claim_evidence_attention import (
    ClaimEvidenceAttentionModel,
    upgrade_attention_state_dict,
)

MODEL = "FacebookAI/roberta-large-mnli"

LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}
LABELS = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]
PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PKG_ROOT / "models"
DATA_PATH_AVERITEC = PKG_ROOT / "data" / "processed" / "averitec_sample.csv"
DATA_PATH_FEVER = PKG_ROOT / "data" / "processed" / "fever_train_claims_sample.csv"

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
        has_attention_keys = any(
            k.startswith("W_c")
            or k.startswith("W_e")
            or k.startswith("v.")
            or k.startswith("head_gate")
            or k.startswith("encoder.")
            for k in state_dict.keys()
        )
        if has_attention_keys:
            print(
                "[ClaimVerifier] Attention-style checkpoint detected. "
                "Skipping state_dict load for baseline classifier."
            )
        else:
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    print(
                        "[ClaimVerifier] Checkpoint loaded with "
                        f"{len(missing)} missing and {len(unexpected)} unexpected keys; "
                        "using base weights for missing params."
                    )
            except RuntimeError:
                print(
                    "[ClaimVerifier] Incompatible checkpoint; "
                    "falling back to base HF weights."
                )

    model.eval()
    return tokenizer, model


def _load_encoder_into_attention_model(
    model: ClaimEvidenceAttentionModel,
    state_dict_path: Union[str, Path],
    map_location: Union[str, torch.device, None] = "cpu",
):
    state_dict = torch.load(state_dict_path, map_location=map_location)
    if any(k.startswith("encoder.") for k in state_dict):
        encoder_state = {
            k[len("encoder."):] : v for k, v in state_dict.items() if k.startswith("encoder.")
        }
    else:
        encoder_state = {
            k.replace("roberta.", "") if k.startswith("roberta.") else k: v
            for k, v in state_dict.items()
            if not k.startswith("classifier")
        }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    return missing, unexpected


def load_attention_verifier(
    model_name: str = MODEL,
    state_dict_path: Union[str, Path, None] = None,
    map_location: Union[str, torch.device, None] = "cpu",
    num_heads: int = 4,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_encoder = AutoModel.from_pretrained(model_name)
    model = ClaimEvidenceAttentionModel(
        base_encoder,
        hidden_dim=base_encoder.config.hidden_size,
        num_labels=len(LABELS),
        num_heads=num_heads,
    )

    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=map_location)
        has_attention = any(
            k.startswith("W_c") or k.startswith("W_e") or k.startswith("v.") or k.startswith("classifier.")
            for k in state_dict.keys()
        )
        if has_attention:
            state_dict = upgrade_attention_state_dict(state_dict, num_heads=model.num_heads)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[AttentionLoad] Missing keys: {len(missing)} (often harmless)")
            if unexpected:
                print(f"[AttentionLoad] Unexpected keys ignored: {len(unexpected)}")
        else:
            # If no attention-specific keys, this isn't an attention checkpoint
            raise ValueError("State dict does not contain attention head weights.")
    else:
        raise ValueError("Attention model requires a checkpoint with head weights.")

    model.eval()
    return tokenizer, model


def verify_claim(
    claim: str,
    evidence_list: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: Union[str, torch.device, None] = None,
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

    if device is not None:
        model = model.to(device)
        enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    label_idx = torch.argmax(probs).item()
    label = LABELS[label_idx]
    probs_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return label, probs_dict


def verify_claim_attention(
    claim: str,
    evidence_list: List[str],
    tokenizer: AutoTokenizer,
    model: ClaimEvidenceAttentionModel,
    device: Union[str, torch.device, None] = None,
    max_evidence: int = 5,
    max_length: int = 256,
) -> Tuple[str, dict, List[Tuple[str, float]]]:
    """
    Run the attention model over a claim + evidence list and return
    label, probs, and attention weights per evidence.
    """
    evidence_list = [e.strip() for e in evidence_list if e and e.strip()]
    if not evidence_list:
        empty_probs = torch.tensor([0.0, 0.0, 1.0])
        return "NOT ENOUGH INFO", {LABELS[i]: float(empty_probs[i]) for i in range(len(LABELS))}, []

    evidence_list = evidence_list[:max_evidence]
    claim_enc = tokenizer(
        [claim],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ev_enc = tokenizer(
        evidence_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    for k in ev_enc:
        ev_enc[k] = ev_enc[k].unsqueeze(0)  # [1, K, L]
    evid_mask = torch.ones(1, len(evidence_list), dtype=torch.long)

    if device is not None:
        model = model.to(device)
        claim_enc = {k: v.to(device) for k, v in claim_enc.items()}
        ev_enc = {k: v.to(device) for k, v in ev_enc.items()}
        evid_mask = evid_mask.to(device)

    with torch.no_grad():
        outputs = model(claim_enc, ev_enc, evid_mask)
        logits = outputs["logits"]
        attn_weights = outputs["attn_weights"].squeeze(0)  # [H, K] or [K]
        head_weights = outputs.get("head_weights")  # [B, H] or None
        probs = torch.softmax(logits, dim=-1)

    label_idx = int(torch.argmax(probs, dim=-1).item())
    probs_dict = {LABELS[i]: float(probs[0, i]) for i in range(len(LABELS))}

    # Fuse per-head attention for a single evidence ranking
    if attn_weights.dim() == 2:
        # attn_weights: [H, K]
        if head_weights is not None:
            fused = torch.einsum("h,hk->k", head_weights.squeeze(0), attn_weights)
        else:
            fused = attn_weights.mean(dim=0)
        attn_for_display = fused
    else:
        # attn_weights: [K]
        attn_for_display = attn_weights

    weights = [(evidence_list[i], float(attn_for_display[i])) for i in range(len(evidence_list))]
    weights.sort(key=lambda x: x[1], reverse=True)
    return LABELS[label_idx], probs_dict, weights


def render_attention_table(claim: str, weights: List[Tuple[str, float]], bar_width: int = 20) -> str:
    """
    Render a simple ASCII table showing attention weights per evidence.
    """
    lines = []
    lines.append(f"Claim: \"{claim}\"")
    lines.append("+----------------------+-----------------+----------------------+")
    lines.append("| Evidence Sentence    | Attention Weight| Highlighting         |")
    lines.append("+----------------------+-----------------+----------------------+")

    max_w = max((w for _, w in weights), default=1.0)
    for ev, w in weights:
        attn_str = f"{w:.2f}"
        bar_len = int((w / max_w) * bar_width) if max_w > 0 else 0
        bar = "#" * bar_len
        ev_short = ev if len(ev) <= 50 else ev[:47] + "..."
        lines.append(f"| {ev_short:<20} | {attn_str:>15} | {bar:<20} |")

    lines.append("+----------------------+-----------------+----------------------+")
    return "\n".join(lines)


def run_inference(
    data_path: str = DATA_PATH_AVERITEC,
    tokenizer: AutoTokenizer = None,
    model_nli: AutoModelForSequenceClassification = None,
    use_attention: bool = False,
    attention_state: Union[str, Path, None] = None,
    max_evidence: int = 5,
    show_plots: bool = True,
    verbose: bool = True,
):
    timer_start = timeit.default_timer()

    if use_attention:
        tokenizer, model_attn = load_attention_verifier(
            state_dict_path=attention_state,
        )
    else:
        if tokenizer is None or model_nli is None:
            tokenizer, model_nli = load_claim_verifier()

    ds = ClaimBatchDataset(data_path)
    predictions = []
    true_labels = []
    for i, (claim, evid_list, true_label) in enumerate(ds, start=1):
        if not evid_list:
            continue

        if use_attention:
            pred_label, probs_dict, _weights = verify_claim_attention(
                claim,
                evid_list,
                tokenizer=tokenizer,
                model=model_attn,
                max_evidence=max_evidence,
            )
        else:
            pred_label, probs_dict = verify_claim(
                claim,
                evid_list,
                tokenizer=tokenizer,
                model=model_nli,
            )
        predictions.append(pred_label)
        true_labels.append(LABELS[true_label])
        # Minimal per-claim output: confidence per label (no attention table / evidence listing)
        probs_str = ", ".join(f"{k}={probs_dict.get(k, 0.0):.3f}" for k in LABELS)
        if verbose:
            print(f"Claim: {claim}")
            print(f"Probs: {probs_str}")
            print(f"True: {LABELS[true_label]} | Predicted: {pred_label}\n")
        else:
            print(f"Claim: {claim}")
            print(f"Probs: {probs_str}\n")
        if i % 50 == 0:
            print(f"Iteration: {i}/{len(ds)}")
            print(f"Time Elapsed: {timeit.default_timer() - timer_start}")

    print(classification_report(true_labels, predictions, labels=LABELS, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, labels=LABELS)
    if show_plots:
        plt.show()

    end = timeit.default_timer()
    print(f"Time Elapsed: {end - timer_start}")


def _resolve_attention_checkpoint(name_or_path: Union[str, Path, None]) -> Union[Path, None]:
    if name_or_path is None:
        return None
    p = Path(name_or_path)
    if p.exists():
        return p
    return resolve_latest_checkpoint(str(name_or_path))


def _cli():
    parser = argparse.ArgumentParser(description="Run claim verification baseline on a CSV.")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with claim/evidence/label.")
    parser.add_argument("--attention", action="store_true", help="Use attention model (requires checkpoint).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path or dataset name.")
    parser.add_argument("--max-evidence", type=int, default=5)
    parser.add_argument("--show-plots", action="store_true", help="Show confusion matrix plot.")
    parser.add_argument("--verbose", action="store_true", help="Print per-example details.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _cli()
    print(f"[Baseline] Package root: {PKG_ROOT}")

    if args.csv:
        if args.attention:
            ckpt = _resolve_attention_checkpoint(args.checkpoint)
            if ckpt is None:
                raise ValueError("Attention mode requires --checkpoint (path or dataset name).")
            print(f"[Main] Using attention model with checkpoint: {ckpt}")
            run_inference(
                args.csv,
                use_attention=True,
                attention_state=ckpt,
                max_evidence=args.max_evidence,
                show_plots=args.show_plots,
                verbose=args.verbose,
            )
        else:
            tok, model = load_claim_verifier()
            run_inference(
                args.csv,
                tokenizer=tok,
                model_nli=model,
                max_evidence=args.max_evidence,
                show_plots=args.show_plots,
                verbose=args.verbose,
            )
    else:
        try:
            ckpt = resolve_latest_checkpoint("averitec_80")
            print(f"[Main] Using attention model with checkpoint: {ckpt}")
            run_inference(
                DATA_PATH_AVERITEC,
                use_attention=True,
                attention_state=ckpt,
                max_evidence=5,
            )
        except Exception as exc:
            print(f"[Main] Falling back to base classifier due to: {exc}")
            tok, model = load_claim_verifier()
            run_inference(DATA_PATH_AVERITEC, tokenizer=tok, model_nli=model)
