"""Data preprocessing helpers for claim verification.

This file contains small, safe helper functions to load raw text data,
clean text, and produce datasets for modelling. It's intentionally
lightweight so you can adapt to your project's needs.
"""
from typing import List
import re
import json
from pathlib import Path


def load_raw(path: str):
    """Load raw JSON-lines or JSON file containing records.

    Returns a list of dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix in {".jsonl", ".ndjson"}:
        out = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    else:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)


def clean_text(text: str) -> str:
    """Basic, deterministic text cleanup for NLP pipelines.

    - normalize whitespace
    - remove control characters
    - strip leading/trailing whitespace
    """
    if text is None:
        return ""
    # normalize line endings and whitespace
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove control characters except newline
    s = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]+", " ", s)
    # collapse multiple whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def save_processed(records: List[dict], path: str):
    p = Path(path)
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def example_usage():
    """Demonstrate a small preprocessing pipeline for a list of records.

    This function is intentionally tiny and useful as a quick check.
    """
    sample = [{"id": 1, "text": " Hello\nWorld!  "}, {"id": 2, "text": None}]
    for r in sample:
        r["text_clean"] = clean_text(r.get("text") or "")
    return sample
