# eval_helpers.py
import json
import ast
import math
import pandas as pd  # if not already imported
import torch
from torch.utils.data import Dataset

def parse_evidence_field(evidence_raw):
    """
    Robustly parse the 'evidence' column from CSV.

    Handles:
      - JSON strings: '["e1", "e2"]'
      - Python list repr: "['e1', 'e2']"
      - Empty strings / NaN -> []
    """
    # If it's already a list, just return it
    if isinstance(evidence_raw, list):
        return evidence_raw

    # Handle missing/NaN
    if evidence_raw is None:
        return []

    # pandas NaN detection
    try:
        if isinstance(evidence_raw, float) and math.isnan(evidence_raw):
            return []
    except Exception:
        pass

    # Convert to string and strip whitespace
    s = str(evidence_raw).strip()
    if s == "" or s.lower() == "nan":
        return []

    # First try JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass  # fall through to Python literal

    # Then try Python list literal (e.g. "['e1', 'e2']")
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
        # if it’s a single string, wrap it
        if isinstance(val, str):
            return [val]
        return []
    except (ValueError, SyntaxError):
        # As a last resort, treat the whole thing as a single evidence string
        return [s]
    

class PairwiseExpansionDataset(Dataset):
    """
    Expands a CSV row into multiple pairs:
      (claim, evidence_sentence, label_id)
    Better for initially training and fine-tuning the model.
    """
    def __init__(self, df, LABEL_MAP):
        self.LABEL_MAP = LABEL_MAP
        pairs = []
        
        for _, row in df.iterrows():
            claim = str(row["claim"]).strip()
            label_id = LABEL_MAP[row["label"].upper()]

            # Parse JSON list from CSV
            evid_list = parse_evidence_field(row["evidence"])

            # Only keep string evidence
            evid_list = [ev.strip() for ev in evid_list if isinstance(ev, str) and ev.strip()]

            for ev in evid_list:
                pairs.append((claim, ev, label_id))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_multi_evidence(batch, tokenizer, max_length=512, label_map=None):
    texts = []
    labels = []

    for item in batch:
        # item is a pandas Series if you pass df, or a dict if you use df.to_dict("records")
        claim = item["claim"]

        # Parse the JSON list from the "evidence" column
        evidence_list = parse_evidence_field(item["evidence"])

        # NORMAL JOIN (no special separator token between evidence sentences)
        # if len(evidence_list) == 0:
        #     continue
        evidence_text = " ".join(evidence_list)
        # evidence_text = " </s> ".join(evidence_list)  # Seperator token version

        # RoBERTa-style input: claim </s></s> evidence_blob
        # input_text = f"{claim} </s></s> {evidence_text}"      # Aviertec RoBERTa input?
        input_text = f"{evidence_text} </s></s> {claim}"      # Fever RoBERTa input?

        texts.append(input_text)

        # Map string label -> int if label_map is given
        if label_map is not None:
            labels.append(label_map[item["label"]])
        else:
            labels.append(item["label"])

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = torch.tensor(labels)

    # Return (enc, labels) to match your eval loop
    return enc, labels

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