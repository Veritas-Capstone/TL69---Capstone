import math
import unittest

from api.claim_verification.evaluation.eval_helpers import parse_evidence_field


class TestEvalHelpers(unittest.TestCase):
    def test_parse_evidence_field_json_list(self):
        raw = '["e1", "e2"]'
        self.assertEqual(parse_evidence_field(raw), ["e1", "e2"])

    def test_parse_evidence_field_python_list(self):
        raw = "['e1', 'e2']"
        self.assertEqual(parse_evidence_field(raw), ["e1", "e2"])

    def test_parse_evidence_field_empty_and_nan(self):
        self.assertEqual(parse_evidence_field(""), [])
        self.assertEqual(parse_evidence_field("nan"), [])
        self.assertEqual(parse_evidence_field(None), [])
        self.assertEqual(parse_evidence_field(float("nan")), [])

    def test_parse_evidence_field_fallback_string(self):
        raw = "single evidence"
        self.assertEqual(parse_evidence_field(raw), ["single evidence"])


if __name__ == "__main__":
    unittest.main()
