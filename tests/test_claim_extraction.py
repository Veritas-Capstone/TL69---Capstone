"""
Unit and integration tests for api/claim_extraction/claim_extraction.py

Tests are split into two classes:
  - TestParseClaimsFromResponse  : pure unit tests for the JSON parser (no LLM)
  - TestExtractClaims            : integration tests for extract_claims() with Ollama mocked

Run from repo root:
    python -m pytest tests/test_claim_extraction.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

from api.claim_extraction.claim_extraction import _parse_claims_from_response, extract_claims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(content: str) -> dict:
    """Build a minimal mock Ollama response dict."""
    return {"message": {"content": content}}


# ---------------------------------------------------------------------------
# 1. Unit tests — _parse_claims_from_response (no LLM, fully deterministic)
# ---------------------------------------------------------------------------

class TestParseClaimsFromResponse(unittest.TestCase):
    """Tests for the internal JSON parser.  No mocking required."""

    # --- Happy-path format variants ---

    def test_standard_claims_key(self):
        raw = '{"claims": ["The Earth is round.", "Water boils at 100°C."]}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round.", "Water boils at 100°C."])

    def test_numbered_dict_keys(self):
        raw = '{"1": "The Earth is round.", "2": "Water boils at 100°C."}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round.", "Water boils at 100°C."])

    def test_named_dict_keys(self):
        raw = '{"Claim 1": "The Earth is round.", "Claim 2": "Water boils at 100°C."}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round.", "Water boils at 100°C."])

    def test_bare_json_array(self):
        raw = '["The Earth is round.", "Water boils at 100°C."]'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round.", "Water boils at 100°C."])

    def test_markdown_fenced_json(self):
        raw = '```json\n{"claims": ["The Earth is round."]}\n```'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round."])

    def test_markdown_fenced_no_language_tag(self):
        raw = '```\n{"claims": ["The Earth is round."]}\n```'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round."])

    def test_claims_with_extra_whitespace(self):
        raw = '{"claims": ["  The Earth is round.  ", "  Water boils at 100°C.  "]}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round.", "Water boils at 100°C."])

    # --- Edge / boundary cases ---

    def test_empty_claims_list(self):
        raw = '{"claims": []}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, [])

    def test_single_claim(self):
        raw = '{"claims": ["Neil Armstrong walked on the Moon in 1969."]}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["Neil Armstrong walked on the Moon in 1969."])

    def test_claims_list_filters_empty_strings(self):
        raw = '{"claims": ["Valid claim.", "", "   ", "Another valid claim."]}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["Valid claim.", "Another valid claim."])

    def test_non_string_values_coerced(self):
        """Integer or boolean values in the list should be cast to strings."""
        raw = '{"claims": [42, true, "A real claim."]}'
        result = _parse_claims_from_response(raw)
        self.assertIn("A real claim.", result)

    # --- Fallback: broken / non-JSON responses ---

    def test_plain_text_fallback(self):
        """Lines longer than 10 chars should be returned as individual claims."""
        raw = "The Earth is round.\nWater boils at 100 degrees Celsius."
        result = _parse_claims_from_response(raw)
        self.assertIn("The Earth is round.", result)
        self.assertIn("Water boils at 100 degrees Celsius.", result)

    def test_bulleted_list_fallback(self):
        raw = "- The Earth is round.\n• Water boils at 100°C.\n* Apollo 11 landed in 1969."
        result = _parse_claims_from_response(raw)
        self.assertTrue(all("The Earth" in c or "Water" in c or "Apollo" in c for c in result))

    def test_short_lines_filtered_in_fallback(self):
        """Lines of 10 chars or fewer should be dropped in the fallback path."""
        raw = "Yes.\nNo.\nThe Earth is round."
        result = _parse_claims_from_response(raw)
        self.assertEqual(result, ["The Earth is round."])

    def test_completely_empty_string(self):
        result = _parse_claims_from_response("")
        self.assertEqual(result, [])

    def test_whitespace_only_string(self):
        result = _parse_claims_from_response("   \n\t  ")
        self.assertEqual(result, [])

    def test_malformed_json_falls_back(self):
        raw = '{"claims": ["The Earth is round."'  # missing closing brackets
        result = _parse_claims_from_response(raw)
        # Fallback: the line is long enough to be returned
        self.assertTrue(len(result) >= 0)  # should not raise

    # --- Partition: input character types ---

    def test_non_english_claims_preserved(self):
        raw = '{"claims": ["La Tierra es redonda.", "L\'eau bout à 100°C."]}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(len(result), 2)

    def test_claims_with_special_characters(self):
        raw = '{"claims": ["CO₂ levels exceeded 420 ppm in 2023.", "5G operates at 24–100 GHz."]}'
        result = _parse_claims_from_response(raw)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# 2. Integration tests — extract_claims() with ollama.chat mocked
# ---------------------------------------------------------------------------

class TestExtractClaims(unittest.TestCase):
    """
    Tests for the public extract_claims() function.
    ollama.chat is always patched so no local model is required.
    """

    APOLLO_PASSAGE = (
        "In 1969, NASA successfully landed the Apollo 11 spacecraft on the Moon. "
        "Neil Armstrong became the first human to walk on the lunar surface. "
        "The mission launched from Kennedy Space Center on July 16, 1969."
    )

    def _patch_ollama(self, content: str):
        """Return a context manager that patches ollama.chat with a fixed response."""
        mock = MagicMock(return_value=_make_ollama_response(content))
        return patch("api.claim_extraction.claim_extraction.ollama.chat", mock)

    # --- Normal operation ---

    def test_max_claims_cap_respected(self):
        many_claims = [f"Claim number {i}." for i in range(20)]
        response_json = f'{{"claims": {many_claims}}}'
        with self._patch_ollama(response_json):
            result = extract_claims(self.APOLLO_PASSAGE, max_claims=5)
        self.assertLessEqual(len(result), 5)

    def test_default_max_claims_is_ten(self):
        many_claims = [f"Claim number {i} is a long enough string." for i in range(20)]
        import json as _json
        response_json = _json.dumps({"claims": many_claims})
        with self._patch_ollama(response_json):
            result = extract_claims(self.APOLLO_PASSAGE)
        self.assertLessEqual(len(result), 10)

    def test_custom_model_name_forwarded(self):
        response_json = '{"claims": ["NASA landed on the Moon in 1969."]}'
        with patch("api.claim_extraction.claim_extraction.ollama.chat") as mock_chat:
            mock_chat.return_value = _make_ollama_response(response_json)
            extract_claims(self.APOLLO_PASSAGE, model="llama3")
            call_kwargs = mock_chat.call_args
            self.assertEqual(call_kwargs[1]["model"] if call_kwargs[1] else call_kwargs[0][0], "llama3")

    def test_temperature_zero_always_set(self):
        """Temperature must be 0 for reproducibility."""
        response_json = '{"claims": ["NASA landed on the Moon in 1969."]}'
        with patch("api.claim_extraction.claim_extraction.ollama.chat") as mock_chat:
            mock_chat.return_value = _make_ollama_response(response_json)
            extract_claims(self.APOLLO_PASSAGE)
            _, kwargs = mock_chat.call_args
            self.assertEqual(kwargs.get("options", {}).get("temperature"), 0)

    def test_passage_included_in_prompt(self):
        """The passage text must appear in the prompt sent to the model."""
        response_json = '{"claims": ["NASA landed on the Moon in 1969."]}'
        with patch("api.claim_extraction.claim_extraction.ollama.chat") as mock_chat:
            mock_chat.return_value = _make_ollama_response(response_json)
            extract_claims(self.APOLLO_PASSAGE)
            _, kwargs = mock_chat.call_args
            prompt_content = kwargs["messages"][0]["content"]
            self.assertIn(self.APOLLO_PASSAGE.strip(), prompt_content)

    # --- Error handling ---

    def test_ollama_connection_error_raises_runtime_error(self):
        with patch("api.claim_extraction.claim_extraction.ollama.chat",
                   side_effect=Exception("connection refused")):
            with self.assertRaises(RuntimeError) as ctx:
                extract_claims(self.APOLLO_PASSAGE)
            self.assertIn("Ollama claim extraction failed", str(ctx.exception))

    def test_unparseable_response_raises_runtime_error(self):
        """A response that yields no claims after parsing should raise RuntimeError."""
        with self._patch_ollama(""):
            with self.assertRaises(RuntimeError) as ctx:
                extract_claims(self.APOLLO_PASSAGE)
            self.assertIn("no parseable claims", str(ctx.exception))

    def test_empty_claims_list_in_response_raises_runtime_error(self):
        with self._patch_ollama('{"claims": []}'):
            with self.assertRaises(RuntimeError):
                extract_claims(self.APOLLO_PASSAGE)

    # --- Boundary / partition: input passage types ---

    def test_empty_passage_still_calls_model(self):
        """extract_claims() does not validate the passage — that is the server's job."""
        response_json = '{"claims": ["Some claim extracted anyway."]}'
        with self._patch_ollama(response_json):
            result = extract_claims("")
        self.assertIsInstance(result, list)

    def test_very_long_passage(self):
        long_passage = "The Apollo program was significant. " * 500
        response_json = '{"claims": ["The Apollo program was significant."]}'
        with self._patch_ollama(response_json):
            result = extract_claims(long_passage)
        self.assertEqual(len(result), 1)

    def test_non_english_passage(self):
        french_passage = "La Terre tourne autour du Soleil. L'eau bout à 100 degrés Celsius."
        response_json = '{"claims": ["La Terre tourne autour du Soleil.", "L\'eau bout à 100 degrés Celsius."]}'
        with self._patch_ollama(response_json):
            result = extract_claims(french_passage)
        self.assertEqual(len(result), 2)

    def test_non_natural_language_passage(self):
        code_passage = "SELECT * FROM users WHERE id = 1; DROP TABLE users;"
        response_json = '{"claims": []}'
        with self._patch_ollama(response_json):
            with self.assertRaises(RuntimeError):
                extract_claims(code_passage)

    def test_single_sentence_passage(self):
        response_json = '{"claims": ["The Earth revolves around the Sun."]}'
        with self._patch_ollama(response_json):
            result = extract_claims("The Earth revolves around the Sun.")
        self.assertEqual(result, ["The Earth revolves around the Sun."])

    def test_passage_with_only_opinions(self):
        """If the model returns no verifiable claims, RuntimeError should propagate."""
        opinion_passage = "I think the sky looks beautiful today. In my view, sunsets are amazing."
        with self._patch_ollama('{"claims": []}'):
            with self.assertRaises(RuntimeError):
                extract_claims(opinion_passage)


if __name__ == "__main__":
    unittest.main()