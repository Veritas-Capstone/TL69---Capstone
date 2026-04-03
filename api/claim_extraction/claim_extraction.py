"""
Claim extraction using a local Ollama LLM.

Mirrors the structure of api/claim_verification/inference/baseline.py so the
server can import and call it the same way it calls the verifier.

Usage:
    from api.claim_extraction.claim_extraction import extract_claims

    claims = extract_claims(passage, model="mistral", max_claims=10)
"""

import json
import re
from typing import List

import ollama

# Default model — matches the options tested in the original script:
# mistral, mixtral, gemma:7b, llama3, phi3
MODEL = "mistral"

EXTRACTION_PROMPT_TEMPLATE = """\
Your task is to extract factual claims from a passage of text.

Rules:
- Extract only atomic, self-contained, independently verifiable claims.
- Split any compound statements into separate claims.
- Omit opinions, questions, and rhetorical statements that cannot be fact-checked.
- Each claim must be understandable on its own without needing extra context.
- Return ONLY a JSON object with a single key "claims" whose value is a list of claim strings.
- Do not include any explanation, preamble, or markdown formatting.

Passage:
{passage}
"""


def _parse_claims_from_response(raw: str) -> List[str]:
    """
    Robustly parse the LLM JSON response into a flat list of claim strings.
    Handles:
      - {"claims": ["...", "..."]}
      - {"1": "...", "2": "..."}  /  {"Claim 1": "..."}
      - ["...", "..."]  (bare top-level array)
      - Markdown-fenced JSON blocks
    Falls back to line-by-line extraction if JSON parsing fails entirely.
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: treat non-empty lines as individual claims
        lines = [ln.strip().strip("-•*").strip() for ln in cleaned.splitlines()]
        return [ln for ln in lines if len(ln) > 10]

    if isinstance(parsed, dict) and "claims" in parsed:
        value = parsed["claims"]
        if isinstance(value, list):
            return [str(c).strip() for c in value if str(c).strip()]

    if isinstance(parsed, dict):
        return [str(v).strip() for v in parsed.values() if str(v).strip()]

    if isinstance(parsed, list):
        return [str(c).strip() for c in parsed if str(c).strip()]

    return []


def extract_claims(
    passage: str,
    model: str = MODEL,
    max_claims: int = 10,
) -> List[str]:
    """
    Use a local Ollama model to extract atomic, verifiable claims from a passage.

    Args:
        passage:    The input text to extract claims from.
        model:      Ollama model name (e.g. "mistral", "llama3", "phi3").
        max_claims: Maximum number of claims to return.

    Returns:
        A list of claim strings, capped at max_claims.

    Raises:
        RuntimeError: If the Ollama call fails or returns an unparseable response.
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(passage=passage.strip())

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
            format="json",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama claim extraction failed (model={model}): {exc}"
        ) from exc

    raw_content = response["message"]["content"]
    claims = _parse_claims_from_response(raw_content)

    if not claims:
        raise RuntimeError(
            f"Claim extraction returned no parseable claims. Raw response:\n{raw_content}"
        )

    return claims[:max_claims]