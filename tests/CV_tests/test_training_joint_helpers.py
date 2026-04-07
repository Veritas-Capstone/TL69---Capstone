import json
import unittest

import pandas as pd
import torch

from api.claim_verification.training.training_joint_helpers import (
    JointEvidenceDataset,
    collate_joint_batch,
)


class DummyTokenizer:
    def __call__(self, *args, padding=True, truncation=True, max_length=8, return_tensors="pt"):
        if not args:
            raise ValueError("Expected at least one positional arg")
        texts = args[0]
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestTrainingJointHelpers(unittest.TestCase):
    def test_joint_dataset_nei_fill_disabled(self):
        df = pd.DataFrame(
            [
                {"claim": "c1", "evidence": json.dumps([]), "label": "NOT ENOUGH INFO"},
                {"claim": "c2", "evidence": json.dumps(["e1"]), "label": "SUPPORTED"},
            ]
        )
        label_map = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}
        ds = JointEvidenceDataset(df, label_map, max_evidence=3, nei_fill=False)
        self.assertEqual(len(ds), 2)
        claim, evid, label = ds[0]
        self.assertEqual(label, 1)
        self.assertEqual(evid, [" "])

    def test_joint_dataset_nei_fill_enabled(self):
        df = pd.DataFrame(
            [
                {"claim": "c1", "evidence": json.dumps([]), "label": "NOT ENOUGH INFO"},
                {"claim": "c2", "evidence": json.dumps(["e1", "e2"]), "label": "SUPPORTED"},
            ]
        )
        label_map = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}
        ds = JointEvidenceDataset(
            df, label_map, max_evidence=2, nei_fill=True, nei_fill_prob=1.0, nei_fill_k=2, nei_fill_seed=1
        )
        claim, evid, label = ds[0]
        self.assertEqual(label, 1)
        self.assertEqual(len(evid), 2)
        self.assertTrue(all(isinstance(x, str) and x.strip() for x in evid))

    def test_collate_joint_batch_shapes(self):
        batch = [
            ("c1", ["e1", "e2"], 0),
            ("c2", ["e3"], 2),
        ]
        tokenizer = DummyTokenizer()
        claim_enc, ev_enc, evid_mask, labels = collate_joint_batch(
            batch, tokenizer, max_length=5
        )
        self.assertEqual(claim_enc["input_ids"].shape, (2, 5))
        self.assertEqual(ev_enc["input_ids"].shape, (2, 2, 5))
        self.assertEqual(evid_mask.shape, (2, 2))
        self.assertEqual(labels.shape, (2,))


if __name__ == "__main__":
    unittest.main()
