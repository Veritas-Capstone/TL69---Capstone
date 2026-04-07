import unittest

from api.claim_verification.data_preprocessing.apply_er_evidence import _clean_evidence_list


class TestEREvidenceCleaning(unittest.TestCase):
    def test_removes_urls_and_trims_top_k(self):
        raw = [
            "Title\nSome content line.\nURL: https://example.com",
            "Another entry with http://foo.bar\nMore text.",
            "Keep me",
        ]
        cleaned = _clean_evidence_list(raw, top_k=2)
        self.assertEqual(len(cleaned), 2)
        for item in cleaned:
            self.assertNotIn("http://", item)
            self.assertNotIn("https://", item)
            self.assertNotIn("URL:", item)


if __name__ == "__main__":
    unittest.main()
