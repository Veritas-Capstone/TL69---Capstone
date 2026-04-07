import json
from pathlib import Path

import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ER_DIR = DATA_DIR / "ER evidence data"

TOP_K_EVIDENCE = 2

DEFAULTS = [
    {
        "name": "fever",
        "er_path": ER_DIR / "fever_full_er.csv",
        "inputs": [
            PROCESSED_DIR / "fever_train_claims_80.csv",
            PROCESSED_DIR / "fever_train_claims_20.csv",
        ],
    },
    {
        "name": "averitec",
        "er_path": ER_DIR / "averitec_full_er.csv",
        "inputs": [
            PROCESSED_DIR / "averitec_80.csv",
            PROCESSED_DIR / "averitec_20.csv",
        ],
    },
]


def _clean_evidence_list(raw, top_k=TOP_K_EVIDENCE):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        evid_list = raw
    else:
        s = str(raw).strip()
        if not s:
            return []
        try:
            evid_list = json.loads(s)
        except Exception:
            evid_list = [s]

    cleaned = []
    for ev in evid_list:
        if not isinstance(ev, str):
            continue
        lines = []
        for line in ev.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("url:"):
                continue
            if "http://" in line or "https://" in line:
                continue
            lines.append(line)
        text = " ".join(lines).strip()
        if text:
            cleaned.append(text)

    if top_k is not None and len(cleaned) > top_k:
        cleaned = cleaned[:top_k]
    return cleaned


def _load_er_map(er_path: Path, top_k=TOP_K_EVIDENCE):
    df = pd.read_csv(er_path)
    if "claim" not in df.columns or "evidence" not in df.columns:
        raise ValueError(f"ER file missing claim/evidence columns: {er_path}")
    df["claim"] = df["claim"].astype(str).str.strip()
    # If duplicates exist, keep first
    df = df.drop_duplicates(subset=["claim"])
    df["evidence"] = df["evidence"].apply(lambda x: json.dumps(_clean_evidence_list(x, top_k=top_k)))
    er_map = df.set_index("claim")["evidence"].to_dict()
    return er_map


def _apply_er_to_dataset(
    input_path: Path,
    output_path: Path,
    er_map: dict,
    fallback_to_original=True,
    top_k=TOP_K_EVIDENCE,
):
    df = pd.read_csv(input_path)
    if "claim" not in df.columns or "evidence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Input missing required columns: {input_path}")

    df["claim"] = df["claim"].astype(str).str.strip()
    df["evidence_er"] = df["claim"].map(er_map)

    found = df["evidence_er"].notna().sum()
    total = len(df)

    if fallback_to_original:
        df["evidence"] = df["evidence_er"].fillna(df["evidence"])
    else:
        df["evidence"] = df["evidence_er"].fillna("[]")

    # Ensure evidence is cleaned + trimmed even when falling back
    df["evidence"] = df["evidence"].apply(lambda x: json.dumps(_clean_evidence_list(x, top_k=top_k)))

    df = df[["claim", "evidence", "label"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} | coverage: {found}/{total} ({found/total:.2%})")


def main():
    for cfg in DEFAULTS:
        er_path = cfg["er_path"]
        if not er_path.exists():
            print(f"[WARN] ER file missing: {er_path}")
            continue
        er_map = _load_er_map(er_path, top_k=TOP_K_EVIDENCE)
        for inp in cfg["inputs"]:
            if not inp.exists():
                print(f"[WARN] Input missing: {inp}")
                continue
            out = inp.with_name(inp.stem + "_er.csv")
            _apply_er_to_dataset(inp, out, er_map, fallback_to_original=True, top_k=TOP_K_EVIDENCE)


if __name__ == "__main__":
    main()
