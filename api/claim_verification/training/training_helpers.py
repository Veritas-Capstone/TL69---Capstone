import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

class PairwiseExpansionDataset(Dataset):
    """
    Expands a CSV row into multiple pairs:
      (claim, evidence_sentence, label_id)
    Better for initially training and fine-tuning the model.
    """
    def __init__(
        self,
        df,
        LABEL_MAP,
        nei_fill=False,
        nei_fill_prob=1.0,
        nei_fill_k=2,
        nei_fill_seed=10,
    ):
        self.LABEL_MAP = LABEL_MAP
        pairs = []
        rng = random.Random(nei_fill_seed)
        evidence_pool = []
        if nei_fill:
            for _, row in df.iterrows():
                try:
                    evid_list = json.loads(row["evidence"])
                except Exception:
                    evid_list = []
                evidence_pool.extend(
                    [ev.strip() for ev in evid_list if isinstance(ev, str) and ev.strip()]
                )
        
        for _, row in df.iterrows():
            claim = str(row["claim"]).strip()
            label_id = LABEL_MAP[row["label"].upper()]

            # Parse JSON list from CSV
            evid_list = json.loads(row["evidence"])

            # Only keep string evidence
            evid_list = [ev.strip() for ev in evid_list if isinstance(ev, str) and ev.strip()]

            # Ensure NOT ENOUGH INFO rows still contribute a training example.
            # Otherwise NEI rows with empty evidence are dropped and the model never
            # sees the NEI label during training.
            if not evid_list:
                if (
                    nei_fill
                    and label_id == LABEL_MAP.get("NOT ENOUGH INFO")
                    and rng.random() < nei_fill_prob
                ):
                    if evidence_pool:
                        if len(evidence_pool) >= nei_fill_k:
                            evid_list = rng.sample(evidence_pool, k=nei_fill_k)
                        else:
                            evid_list = [rng.choice(evidence_pool) for _ in range(nei_fill_k)]
                    else:
                        evid_list = [" "]
                else:
                    evid_list = [" "]

            for ev in evid_list:
                pairs.append((claim, ev, label_id))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


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
