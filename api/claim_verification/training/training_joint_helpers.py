import json
import random
import torch
from torch.utils.data import Dataset

class JointEvidenceDataset(Dataset):
    """
    Each sample is:
        (claim, list of evidence sentences, label_id)
    Used for multi-evidence attention models.
    """
    def __init__(
        self,
        df,
        LABEL_MAP,
        max_evidence=10,
        nei_fill=False,
        nei_fill_prob=1.0,
        nei_fill_k=2,
        nei_fill_seed=10,
    ):
        self.LABEL_MAP = LABEL_MAP
        self.max_evidence = max_evidence
        self.samples = []
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

            # Load and clean evidence list
            evid_list = json.loads(row["evidence"])
            evid_list = [ev.strip() for ev in evid_list if isinstance(ev, str) and ev.strip()]

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
                    evid_list = [" "]  # handle NEI with no evidence

            # Truncate to max_evidence length
            evid_list = evid_list[:self.max_evidence]

            self.samples.append((claim, evid_list, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_joint_batch(batch, tokenizer, max_length=256):
    """
    Collate for JointEvidenceDataset.
    Tokenizes:
      - claims: shape [B, L]
      - evidence sentences: shape [B, K, L]
    Returns:
      claim_inputs, evidence_inputs, evidence_mask, labels
    """
    claims, evidence_lists, labels = zip(*batch)
    B = len(claims)
    K = max(len(evs) for evs in evidence_lists)

    # Pad evidence lists
    padded_evids = []
    evid_mask = []
    for ev_list in evidence_lists:
        padded = ev_list + [""] * (K - len(ev_list))
        padded_evids.extend(padded)
        evid_mask.append([1]*len(ev_list) + [0]*(K - len(ev_list)))

    # Tokenize claims
    claim_enc = tokenizer(
        list(claims), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    # Tokenize evidence: flat then reshape
    ev_enc = tokenizer(
        padded_evids, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    for k in ev_enc:
        ev_enc[k] = ev_enc[k].view(B, K, -1)

    evid_mask = torch.tensor(evid_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return claim_enc, ev_enc, evid_mask, labels
