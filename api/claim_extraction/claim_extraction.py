"""
Claim extraction using a local Ollama LLM.

Mirrors the structure of api/claim_verification/inference/baseline.py so the
server can import and call it the same way it calls the verifier.

Two public functions:

    extract_claims(passage, model, max_claims)
        → List[str]   (original interface, unchanged — drop-in compatible)

    extract_claims_with_provenance(passage, model, max_claims_per_sentence)
        → ExtractionResult  (new — includes per-sentence claim mapping)

The provenance format passed downstream to the ER module:

    {
      "passage": "<full original passage>",
      "sentence_claims": [
        {
          "sentence_id": 0,
          "sentence": "Marie Curie was born in Warsaw in 1867.",
          "claims": ["Marie Curie was born in Warsaw in 1867."]
        },
        ...
      ]
    }

Strategy: per-sentence extraction (one LLM call per sentence).
  - Provenance is exact and deterministic — no post-hoc matching needed.
  - Avoids a second embedding pass.
  - Works even when a sentence contains a compound claim that gets split.

Usage:
    from api.claim_extraction.claim_extraction import (
        extract_claims,
        extract_claims_with_provenance,
    )

    # Original interface (unchanged):
    claims = extract_claims(passage, model="mistral", max_claims=10)

    # New interface with provenance:
    result = extract_claims_with_provenance(passage, model="mistral")
    result.to_er_payload()   # dict ready to send to the ER module
    result.all_claims()      # flat list of all claims (convenience)
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import ollama

# Default model — matches the options tested in the original script:
# mistral, mixtral, gemma:7b, llama3, phi3
MODEL = "mistral"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_TEMPLATE = """\
Your task is to extract factual claims from a passage of text.

Rules:
- Extract all atomic, self-contained, independently verifiable claims.
- Omit opinions, questions, and rhetorical statements that cannot be fact-checked.
- Each claim must be understandable on its own without needing extra context.
- Do not split compound facts that belong together into separate claims.
- Return ONLY a JSON object with a single key "claims" whose value is a list of claim strings.
- Do not include any explanation, preamble, or markdown formatting.

Example
-------
Passage:
"Marie Curie was born in Warsaw in 1867 and became the first woman to win a Nobel Prize. \
She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."

Output:
{{"claims": [
  "Marie Curie was born in Warsaw in 1867.",
  "Marie Curie was the first woman to win a Nobel Prize.",
  "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."
]}}

Notice how the example:
- Replaces "She" and "her" with "Marie Curie" in every claim.
- Keeps compound facts from the same sentence together rather than over-splitting.

Passage:
{passage}
"""

# Lighter prompt used for sentence-level extraction. It includes a tiny
# local context window so pronouns in the target sentence can be resolved
# without paying the cost of full-passage extraction for every sentence.
SINGLE_SENTENCE_WITH_CONTEXT_PROMPT_TEMPLATE = """\
Your task is to extract factual claims from one target sentence.

You are also given a small amount of surrounding context. Use that context
only to resolve references such as pronouns, titles, or abbreviated names.
Extract claims from the TARGET SENTENCE only.

Rules:
- Extract all atomic, self-contained, independently verifiable claims from the TARGET SENTENCE only.
- Do not extract claims from the surrounding context sentences.
- Omit opinions, questions, and rhetorical statements that cannot be fact-checked.
- Replace pronouns and other references in the TARGET SENTENCE with the entity they refer to using the surrounding context when needed.
- If a reference is ambiguous, keep the original wording instead of guessing.
- Do not split compound facts that belong together.
- Return ONLY a JSON object with a single key "claims" whose value is a list of claim strings.
- Do not include any explanation, preamble, or markdown formatting.
- If the TARGET SENTENCE contains no verifiable factual claims, return {{"claims": []}}.

Previous sentence:
{previous_sentence}

TARGET SENTENCE:
{target_sentence}

Next sentence:
{next_sentence}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentenceClaims:
    """Claims extracted from a single sentence, with its position in the passage."""
    sentence_id: int
    sentence: str
    claims: List[str] = field(default_factory=list)
    error: Optional[str] = None  # set if extraction failed for this sentence


@dataclass
class ExtractionResult:
    """Full extraction result with per-sentence provenance."""
    passage: str
    sentence_claims: List[SentenceClaims] = field(default_factory=list)

    def all_claims(self) -> List[str]:
        """Flat list of every extracted claim across all sentences."""
        return [
            claim
            for sc in self.sentence_claims
            for claim in sc.claims
        ]

    def to_er_payload(self) -> dict:
        """
        Serialise to the dict format expected by the Evidence Retrieval module.

        Shape:
            {
              "passage": str,
              "sentence_claims": [
                {"sentence_id": int, "sentence": str, "claims": [str, ...]},
                ...
              ]
            }

        Sentences where extraction errored are included with an empty claims
        list so the ER module can account for all original sentences if needed.
        """
        return {
            "passage": self.passage,
            "sentence_claims": [
                {
                    "sentence_id": sc.sentence_id,
                    "sentence": sc.sentence,
                    "claims": sc.claims,
                }
                for sc in self.sentence_claims
            ],
        }

    def to_flat_payload(self) -> dict:
        """
        Alternative flat format: each claim is an object carrying its source
        sentence. Useful if the ER module processes claims one-by-one.

        Shape:
            {
              "passage": str,
              "claims": [
                {"claim": str, "sentence_id": int, "source_sentence": str},
                ...
              ]
            }
        """
        return {
            "passage": self.passage,
            "claims": [
                {
                    "claim": claim,
                    "sentence_id": sc.sentence_id,
                    "source_sentence": sc.sentence,
                }
                for sc in self.sentence_claims
                for claim in sc.claims
            ],
        }


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Split a passage into sentences.

    Tries nltk.sent_tokenize first (handles abbreviations like "U.S." and
    "Dr." correctly). Falls back to a simple regex splitter if nltk is not
    available so the module has no hard dependency.
    """
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            # Punkt tokenizer data not downloaded yet — download silently
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except ImportError:
        pass

    # Fallback: split on ". ", "! ", "? " but preserve the punctuation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# LLM call + response parsing (unchanged from original)
# ---------------------------------------------------------------------------

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


def _call_ollama(prompt: str, model: str) -> str:
    """Single Ollama call; raises RuntimeError on failure."""
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
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def _extract_from_text(text: str, model: str, prompt_template: str) -> List[str]:
    """Extract claims from an arbitrary text snippet using the given template."""
    prompt = prompt_template.format(passage=text.strip())
    raw = _call_ollama(prompt, model)
    return _parse_claims_from_response(raw)


def extract_claims(
    passage: str,
    model: str = MODEL,
    max_claims: int = 10,
) -> List[str]:
    """
    Use a local Ollama model to extract atomic, verifiable claims from a passage.
    Original interface — unchanged for backward compatibility.

    Args:
        passage:    The input text to extract claims from.
        model:      Ollama model name (e.g. "mistral", "llama3", "phi3").
        max_claims: Maximum number of claims to return (default: 10).

    Returns:
        A list of claim strings, capped at max_claims.

    Raises:
        RuntimeError: If the Ollama call fails or returns an unparseable response.
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(passage=passage.strip())
    raw = _call_ollama(prompt, model)
    claims = _parse_claims_from_response(raw)

    if not claims:
        raise RuntimeError(
            f"Claim extraction returned no parseable claims. Raw response:\n{raw}"
        )

    return claims[:max_claims]


def extract_claims_with_provenance(
    passage: str,
    model: str = MODEL,
    max_claims_per_sentence: int = None,
    skip_short_sentences: bool = True,
    min_sentence_length: int = 15,
    context_window: int = 1,
) -> ExtractionResult:
    """
    Extract claims with sentence-level provenance.

    Splits the passage into sentences and runs one LLM extraction call per
    sentence so each claim is automatically associated with its source sentence.

    Args:
        passage:                 The input text to extract claims from.
        model:                   Ollama model name.
        max_claims_per_sentence: Cap on claims extracted per sentence (None = no cap).
        skip_short_sentences:    If True, sentences shorter than min_sentence_length
                                 characters are skipped (avoids wasted LLM calls on
                                 headings, bylines, etc.).
        min_sentence_length:     Character threshold for skip_short_sentences.
        context_window:          Number of sentences on each side of the target
                                 sentence to provide as lightweight context for
                                 pronoun/reference resolution. Extraction still
                                 happens only for the target sentence.

    Returns:
        ExtractionResult with per-sentence claim lists.
        Call .to_er_payload() to get the dict for the ER module.
        Call .all_claims() to get a flat list of all claims.
    """
    sentences = split_into_sentences(passage)
    result = ExtractionResult(passage=passage)

    for idx, sentence in enumerate(sentences):
        if skip_short_sentences and len(sentence) < min_sentence_length:
            # Still include the sentence so sentence_ids stay contiguous,
            # but emit an empty claims list rather than calling the LLM.
            result.sentence_claims.append(
                SentenceClaims(sentence_id=idx, sentence=sentence, claims=[])
            )
            continue

        try:
            prev_context = " ".join(sentences[max(0, idx - context_window):idx]).strip()
            next_context = " ".join(
                sentences[idx + 1:min(len(sentences), idx + 1 + context_window)]
            ).strip()
            prompt_input = SINGLE_SENTENCE_WITH_CONTEXT_PROMPT_TEMPLATE.format(
                previous_sentence=prev_context or "(none)",
                target_sentence=sentence,
                next_sentence=next_context or "(none)",
            )
            claims = _extract_from_text(
                prompt_input, model, "{passage}"
            )
            if max_claims_per_sentence is not None:
                claims = claims[:max_claims_per_sentence]
            result.sentence_claims.append(
                SentenceClaims(sentence_id=idx, sentence=sentence, claims=claims)
            )
        except RuntimeError as exc:
            result.sentence_claims.append(
                SentenceClaims(
                    sentence_id=idx,
                    sentence=sentence,
                    claims=[],
                    error=str(exc),
                )
            )

    return result


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passage = (
        "Marie Curie was born in Warsaw in 1867 and became the first woman to win a Nobel Prize. "
        "She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911, "
        "making her the only person to win Nobel Prizes in two different sciences. "
        "Curie died in 1934 from aplastic anaemia, believed to have been caused by her long-term "
        "exposure to radiation."
    )

    print("=" * 60)
    print("--- extract_claims (original interface) ---")
    claims = extract_claims(passage, model="mistral")
    for c in claims:
        print(f"  • {c}")

    print()
    print("--- extract_claims_with_provenance ---")
    result = extract_claims_with_provenance(passage, model="mistral")

    for sc in result.sentence_claims:
        print(f"\n[S{sc.sentence_id}] {sc.sentence}")
        if sc.error:
            print(f"  ⚠ Error: {sc.error}")
        elif not sc.claims:
            print("  (no verifiable claims)")
        else:
            for c in sc.claims:
                print(f"  • {c}")

    print()
    print("--- ER payload (to_er_payload) ---")
    import json as _json
    print(_json.dumps(result.to_er_payload(), indent=2))

    print()
    print("--- Flat payload (to_flat_payload) ---")
    print(_json.dumps(result.to_flat_payload(), indent=2))
