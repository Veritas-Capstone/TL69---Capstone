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

"""
LLM-based Claim Verification on AveriTeC using local Mistral (Ollama).

- Claim-level LLM judgments (SUPPORTED / REFUTED / NOT ENOUGH INFO + confidence)
- Caching per claim to avoid re-querying the LLM
- Post-processing that:
    * trusts REFUTED as-is
    * only accepts SUPPORTED when confidence is very high
    * maps most NOT ENOUGH INFO cases to REFUTED, keeping only very confident NEI

Run:
    python llm_baseline.py
"""

import os
import sys
import ast
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from torch.utils.data import Dataset

# NEW: for confusion matrix & metrics “popup”
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report

# =========================
# Config (file-anchored)
# =========================

HERE = Path(__file__).resolve()          # .../inference/llm_baseline.py
API_DIR = HERE.parent.parent             # .../api/claim_verification
DATA_DIR = API_DIR / "data" / "processed"

# Env overrides or defaults
DATA_PATH = Path(
    os.environ.get("AVERIC_PATH", str(DATA_DIR / "averitec_sample.csv"))
).resolve()

CACHE_PATH = Path(
    os.environ.get("AVERIC_CACHE", str(DATA_DIR / "llm_claim_cache.jsonl"))
).resolve()

# Option: clear cache every run (default True)
CLEAR_CACHE_EACH_RUN = os.environ.get("CLEAR_LLM_CACHE", "1") == "1"

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
print("[Info] Resolved DATA_PATH:", DATA_PATH, "| exists:", DATA_PATH.exists())
print("[Info] Resolved CACHE_PATH:", CACHE_PATH)
assert DATA_PATH.exists(), (
    f"Dataset not found at {DATA_PATH}\n"
    f"Tip: export AVERIC_PATH=/absolute/path/to/averitec_sample.csv"
)

# Ollama host: accept with or without http://
_raw_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_HOST = (
    _raw_host if _raw_host.startswith(("http://", "https://"))
    else f"http://{_raw_host}"
)

# Use a model that actually exists on this server by default
# (override with: export LLM_MODEL="mistral:7b-instruct-v0.3-q3_K_M" etc.)
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral:7b")

LABELS = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]
LABEL_SET = set(LABELS)
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

# Post-processing thresholds
SUPPORTED_MIN_CONF = 0.90        # how strict we are about calling something SUPPORTED
NEI_TO_REFUTED_THRESHOLD = 0.75  # NEI with conf < this => REFUTED

# =========================
# Health check
# =========================

def _ollama_ok(host: str) -> bool:
    try:
        r = requests.get(f"{host}/api/version", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False

print(f"[Info] Using LLM_MODEL={LLM_MODEL} via {OLLAMA_HOST}")
if not _ollama_ok(OLLAMA_HOST):
    print(
        f"[FATAL] Ollama not reachable at {OLLAMA_HOST}. "
        f"Start `ollama serve` on this node or fix OLLAMA_HOST."
    )
    sys.exit(1)

# =========================
# Dataset
# =========================

class ClaimDataset(Dataset):
    """
    Each item = (claim: str, evidences: List[str], gold_label: str)
    Evidence column is a list serialized as a string in the CSV.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        # Parse evidence list safely
        def parse_ev(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return [x]
            return []

        df["evidence"] = df["evidence"].apply(parse_ev)

        self.claims = df["claim"].tolist()
        self.evidences = df["evidence"].tolist()
        self.labels = [str(l).upper() for l in df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.claims)

    def __getitem__(self, idx: int):
        return self.claims[idx], self.evidences[idx], self.labels[idx]

# =========================
# Prompting
# =========================

SYSTEM_PROMPT = (
    "You are an expert fact-checking assistant for political and news claims. "
    "You will be given a CLAIM and a list of EVIDENCE sentences drawn from reliable sources. "
    "Your job is to decide whether the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO "
    "based *only* on the provided evidence.\n\n"
    "Definitions:\n"
    "- SUPPORTED: The evidence clearly and directly shows the claim is true. "
    "There is no substantial contradiction in the provided evidence.\n"
    "- REFUTED: The evidence clearly and directly shows the claim is false. "
    "It contradicts the main statement of the claim.\n"
    "- NOT ENOUGH INFO: The evidence is mixed, ambiguous, or lacks the necessary information "
    "to confidently support or refute the claim. If evidence points in different directions, "
    "or only loosely relates to the claim, use NOT ENOUGH INFO.\n\n"
    "Important:\n"
    "- You must only use the provided EVIDENCE. Do not rely on outside knowledge.\n"
    "- Be conservative about predicting SUPPORTED. Only use SUPPORTED if the evidence clearly "
    "and directly entails the claim.\n"
    "- When in doubt, prefer NOT ENOUGH INFO over SUPPORTED.\n\n"
    "Output strict JSON with keys: label, confidence, rationale. "
    "Allowed labels: SUPPORTED, REFUTED, NOT ENOUGH INFO."
)

def build_user_prompt(claim: str, evidences: List[str]) -> str:
    # Number the evidence sentences to make them easier to reference in the rationale.
    lines = [
        "Decide whether the CLAIM is SUPPORTED, REFUTED, or NOT ENOUGH INFO "
        "using only the EVIDENCE list.",
        "",
        f"CLAIM: {claim}",
        "",
        "EVIDENCE:",
    ]
    for i, ev in enumerate(evidences):
        lines.append(f"  [{i}] {ev}")
    lines.append("")
    lines.append(
        "Return JSON ONLY in the form:\n"
        '{"label":"SUPPORTED","confidence":0.93,'
        '"rationale":"Very concise explanation based only on the evidence."}'
    )
    return "\n".join(lines)

# =========================
# JSON extraction
# =========================

def extract_first_json(s: str) -> dict:
    """Return the first balanced JSON object found in s. Falls back to {}."""
    if not s:
        return {}
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except Exception:
        pass

    start_idx = s.find("{")
    while start_idx != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start_idx, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = s[start_idx : i + 1]
                        try:
                            return json.loads(cand)
                        except Exception:
                            break
        start_idx = s.find("{", start_idx + 1)
    return {}

# =========================
# LLM call & result type
# =========================

@dataclass
class ClaimResult:
    claim_idx: int
    label: str
    confidence: float
    rationale: str

def llm_classify_claim(
    idx: int,
    claim: str,
    evidences: List[str],
) -> Optional[ClaimResult]:
    """Call local LLM via Ollama /api/generate to classify a single claim+evidence set."""

    # Combine system prompt + user prompt into a single prompt string
    prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(claim, evidences)

    payload = {
        "model": LLM_MODEL,
        "options": {
            "temperature": 0,
            "repeat_penalty": 1.05,
        },
        "format": "json",       # ask Ollama to give us JSON
        "stream": False,
        "prompt": prompt,
    }

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Standard Ollama /api/generate response has "response" with the text
        content = ""
        if isinstance(data, dict):
            content = data.get("response", "") or ""
        out = extract_first_json(content)
    except Exception as e:
        print(f"[WARN] LLM call failed for idx={idx}: {e}")
        return None

    raw_label = str(out.get("label", "")).upper().strip()
    if raw_label not in LABEL_SET:
        return None

    try:
        conf = float(out.get("confidence", 0.0))
    except Exception:
        return None
    conf = float(np.clip(conf, 0.0, 1.0))

    rationale = str(out.get("rationale", ""))

    return ClaimResult(
        claim_idx=idx,
        label=raw_label,
        confidence=conf,
        rationale=rationale,
    )

# =========================
# Post-processing
# =========================

def postprocess_claim_label(
    raw_label: str,
    conf: float,
) -> str:
    """
    Post-processing with asymmetric behaviour:

    - If model says REFUTED -> keep REFUTED (we trust refutations).
    - If model says SUPPORTED:
        * require very high confidence (SUPPORTED_MIN_CONF),
        * otherwise demote to NOT ENOUGH INFO.
    - If model says NOT ENOUGH INFO:
        * if confidence < NEI_TO_REFUTED_THRESHOLD, map to REFUTED
          (since in this dataset ambiguous cases are more often refuted),
        * otherwise keep as NOT ENOUGH INFO.
    """
    label = raw_label.upper().strip()
    if label not in LABEL_SET:
        return "NOT ENOUGH INFO"

    conf = float(np.clip(conf, 0.0, 1.0))

    if label == "REFUTED":
        return "REFUTED"

    if label == "SUPPORTED":
        if conf >= SUPPORTED_MIN_CONF:
            return "SUPPORTED"
        else:
            return "NOT ENOUGH INFO"

    # label == "NOT ENOUGH INFO"
    if conf < NEI_TO_REFUTED_THRESHOLD:
        return "REFUTED"
    else:
        return "NOT ENOUGH INFO"

# =========================
# Cache helpers
# =========================

def load_cache(path: Path) -> Dict[int, ClaimResult]:
    if not path.exists():
        return {}
    cache: Dict[int, ClaimResult] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cache[d["claim_idx"]] = ClaimResult(
                claim_idx=d["claim_idx"],
                label=d["label"],
                confidence=d["confidence"],
                rationale=d.get("rationale", ""),
            )
    return cache

def append_cache(path: Path, res: ClaimResult):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res.__dict__) + "\n")

# =========================
# Runner
# =========================

def run_llm_over_dataset(
    ds: ClaimDataset,
    throttle_s: float = 0.0,
    use_cache: bool = True,
) -> Dict[int, ClaimResult]:
    """
    Run the LLM once per claim+evidence set, with per-claim caching.
    Returns dict keyed by claim_idx.
    """
    cache = load_cache(CACHE_PATH) if use_cache else {}
    out: Dict[int, ClaimResult] = dict(cache)

    for idx in range(len(ds)):
        if idx in out:
            continue
        claim, evidences, _gold = ds[idx]
        res = llm_classify_claim(idx, claim, evidences)
        if res is None:
            # skip failures; we'll treat as NEI in evaluation
            continue
        out[idx] = res
        if use_cache:
            append_cache(CACHE_PATH, res)
        if throttle_s > 0:
            time.sleep(throttle_s)

    return out

# =========================
# Metrics helpers
# =========================

def compute_metrics(preds: List[str], golds: List[str]) -> Dict[str, Any]:
    assert len(preds) == len(golds)
    n = len(preds)
    correct = sum(int(p == g) for p, g in zip(preds, golds))
    acc = correct / n if n else 0.0

    per_class = {}
    f1s = []
    for L in LABELS:
        tp = sum(1 for p, g in zip(preds, golds) if p == L and g == L)
        fp = sum(1 for p, g in zip(preds, golds) if p == L and g != L)
        fn = sum(1 for p, g in zip(preds, golds) if p != L and g == L)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[L] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(1 for g in golds if g == L),
        }
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class}

# =========================
# Visualization
# =========================

def plot_confusion_matrix_and_metrics(
    golds: List[str],
    preds: List[str],
    metrics: Dict[str, Any],
    labels: List[str],
    save_prefix: str = "llm_baseline_",
):
    """
    Use sklearn + matplotlib to show:
    - Confusion matrix heatmap
    - Metrics table (accuracy, macro-F1, per-class)
    Also saves both as PNGs to disk.
    """
    # Confusion matrix (sklearn)
    cm = sk_confusion_matrix(golds, preds, labels=labels)

    # ---- Confusion matrix heatmap ----
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.figure.colorbar(im, ax=ax_cm)
    ax_cm.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Gold label",
        xlabel="Predicted label",
        title="Confusion Matrix – LLM Baseline on Averitec",
    )

    # Rotate x-axis labels for readability
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig_cm.tight_layout()
    cm_path = f"{save_prefix}_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=150)
    print(f"[Info] Saved confusion matrix figure to {cm_path}")

    # ---- Metrics table popup ----
    # We'll build a small table figure for overall + per-class metrics.
    overall_acc = metrics["accuracy"]
    overall_macro_f1 = metrics["macro_f1"]

    # Flatten per-class metrics into rows
    table_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for L in labels:
        m = metrics["per_class"][L]
        table_data.append([
            L,
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['support']}",
        ])

    # Add overall row
    table_data.append([
        "OVERALL",
        "-", "-",
        f"{overall_macro_f1:.3f} (macro F1)",
        f"{overall_acc:.3f} (acc)",
    ])

    fig_tbl, ax_tbl = plt.subplots(figsize=(7, 0.6 * len(table_data) + 1))
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=None,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.2)
    ax_tbl.set_title("LLM Baseline – Metrics Overview", pad=12)

    fig_tbl.tight_layout()
    metrics_path = f"{save_prefix}_metrics_table.png"
    fig_tbl.savefig(metrics_path, dpi=150)
    print(f"[Info] Saved metrics table figure to {metrics_path}")

    # If you’re running with a GUI (e.g., locally or X11 forwarding),
    # this will “popup” the figures. On headless servers you’ll just get PNGs.
    try:
        plt.show()
    except Exception:
        # On a headless cluster, show() may fail; PNGs are still saved.
        pass

# =========================
# Main
# =========================

if __name__ == "__main__":
    print(f"[Info] Loading dataset from {DATA_PATH}")

    if CLEAR_CACHE_EACH_RUN and CACHE_PATH.exists():
        print(f"[Info] Clearing cache file at {CACHE_PATH}")
        CACHE_PATH.unlink()

    ds = ClaimDataset(str(DATA_PATH))

    results = run_llm_over_dataset(ds, throttle_s=0.0, use_cache=True)
    print(f"[Info] Claim-level LLM results: {len(results)} rows (cached+new)")

    # Debug: predicted label distribution (AFTER post-processing)
    post_labels = []
    for i in range(len(ds)):
        res = results.get(i)
        if res is None:
            post_labels.append("NOT ENOUGH INFO")
        else:
            post_labels.append(postprocess_claim_label(res.label, res.confidence))
    print("[Debug] Predicted label distribution:", dict(Counter(post_labels)))

    # Build gold + preds
    golds: List[str] = []
    preds: List[str] = []
    for i in range(len(ds)):
        _claim, _evidences, gold = ds[i]
        gold = gold.upper()
        if gold not in LABEL_SET:
            gold = "NOT ENOUGH INFO"
        golds.append(gold)

        res = results.get(i)
        if res is None:
            preds.append("NOT ENOUGH INFO")
        else:
            preds.append(postprocess_claim_label(res.label, res.confidence))

    metrics = compute_metrics(preds, golds)

    # Instead of printing a giant matrix + per-class numbers, we:
    # 1) Save/“popup” confusion matrix heatmap
    # 2) Save/“popup” a metrics table
    plot_confusion_matrix_and_metrics(
        golds=golds,
        preds=preds,
        metrics=metrics,
        labels=LABELS,
        save_prefix="llm_baseline_",
    )

    # If you still want a quick text summary in the terminal, you can leave this:
    print(
        "[Report] Claim-level metrics (global):",
        {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
    )
