import unittest

import torch

from api.claim_verification.models.claim_evidence_attention import upgrade_attention_state_dict


class TestUpgradeAttentionStateDict(unittest.TestCase):
    def test_upgrade_repeats_weights_and_drops_legacy(self):
        wc = torch.randn(4, 4)
        we = torch.randn(4, 4)
        vw = torch.randn(1, 4)
        state = {
            "W_c.weight": wc,
            "W_c.bias": torch.zeros(4),
            "W_e.weight": we,
            "W_e.bias": torch.zeros(4),
            "v.weight": vw,
            "v.bias": torch.zeros(1),
        }
        upgraded = upgrade_attention_state_dict(state, num_heads=3)
        self.assertIn("W_c", upgraded)
        self.assertIn("W_e", upgraded)
        self.assertIn("v", upgraded)
        self.assertEqual(upgraded["W_c"].shape, (3, 4, 4))
        self.assertEqual(upgraded["W_e"].shape, (3, 4, 4))
        self.assertEqual(upgraded["v"].shape, (3, 4))
        for legacy_key in ["W_c.weight", "W_c.bias", "W_e.weight", "W_e.bias", "v.weight", "v.bias"]:
            self.assertNotIn(legacy_key, upgraded)


if __name__ == "__main__":
    unittest.main()
