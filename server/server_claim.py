"""
FastAPI server for CLAIM VERIFICATION demo.

Run from repo root:
    uvicorn server.claim_server:app --reload --port 8001

This server is independent of the bias-detection server.py
and only handles claim verification.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from api.claim_extraction.claim_extraction import (
    extract_claims,
    extract_claims_with_provenance,
)
from api.claim_verification.inference import baseline
from api.claim_verification.retrieval.colbert_rerank import ClaimEvidenceRetriever

# Load environment variables from local .env (if present).
load_dotenv()

# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------

app = FastAPI(title="Claim Verification Demo API")

# Allow extension / frontend to call this easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models / schemas
# -----------------------------------------------------------------------------

class PassageRequest(BaseModel):
    text: str


class ClaimEvidence(BaseModel):
    claim: str
    evidence: List[str]
    gold_label: Optional[str] = None


class ClaimVerificationResult(BaseModel):
    claim: str
    evidence: List[str]
    label: str
    probabilities: Dict[str, float]
    # Provenance fields — always present when using verify_claims_from_passage.
    # None only in verify_claims_direct (where no passage was split).
    sentence_id: Optional[int] = None
    source_sentence: Optional[str] = None


class ClaimExtractionResult(BaseModel):
    claims: List[str]
    model: str
    count: int


# Optional: direct endpoint where extension (or you) pass claims+evidence directly
class ClaimVerificationDirectRequest(BaseModel):
    items: List[ClaimEvidence]


class ClaimRequest(BaseModel):
    claim: str


class EvidenceRetrievalResult(BaseModel):
    claim: str
    evidence: List[str]
    confidence: bool
    force_nei: bool
    source: str


# -----------------------------------------------------------------------------
# Claim verification model
# -----------------------------------------------------------------------------

# Configure via env vars:
#   CLAIM_MODEL_DATASET   -> dataset folder under api/claim_verification/models (loads latest.pt)
#   CLAIM_MODEL_CHECKPOINT-> explicit path to a .pt state_dict
#   CLAIM_MODEL_ARCH      -> HF model id / local HF directory for the base architecture
MODEL_ARCH = os.getenv("CLAIM_MODEL_ARCH", baseline.MODEL)
MODEL_DATASET = os.getenv("CLAIM_MODEL_DATASET", "fever_averitec_er_r50_h4") 
MODEL_CHECKPOINT = os.getenv("CLAIM_MODEL_CHECKPOINT")


def _resolve_checkpoint():
    if MODEL_CHECKPOINT:
        ckpt_path = Path(MODEL_CHECKPOINT)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).resolve().parent / MODEL_CHECKPOINT
        if not ckpt_path.exists():
            raise FileNotFoundError(f"CLAIM_MODEL_CHECKPOINT not found at {ckpt_path}")
        return ckpt_path
    if MODEL_DATASET:
        try:
            return baseline.resolve_latest_checkpoint(MODEL_DATASET)
        except FileNotFoundError as exc:
            print(f"[ClaimServer] {exc}. Falling back to base HF weights.")


try:
    ckpt = _resolve_checkpoint()
except FileNotFoundError as exc:
    print(f"[ClaimServer] {exc}. Falling back to base HF weights.")
    ckpt = None

source_desc = str(ckpt) if ckpt else MODEL_ARCH
print(f"[ClaimServer] Loading claim verification model from: {source_desc}")
use_attention = False
if ckpt:
    try:
        claim_tokenizer, claim_model = baseline.load_attention_verifier(
            model_name=MODEL_ARCH,
            state_dict_path=ckpt,
        )
        use_attention = True
        print("[ClaimServer] Loaded attention-based verifier.")
    except Exception as exc:
        print(f"[ClaimServer] Attention load failed ({exc}). Falling back to baseline.")
        claim_tokenizer, claim_model = baseline.load_claim_verifier(
            model_name=MODEL_ARCH,
            state_dict_path=ckpt,
        )
else:
    claim_tokenizer, claim_model = baseline.load_claim_verifier(
        model_name=MODEL_ARCH,
        state_dict_path=ckpt,
    )

print("[ClaimServer] Claim verification model loaded!")

available = baseline.list_available_checkpoints()
if available:
    print("[ClaimServer] Available latest checkpoints:")
    for name, path in available.items():
        print(f"  - {name}: {path}")


def verify_single_claim(claim: str, evidence_list: List[str]):
    """
    Given a single claim and its evidence sentences, run the NLI/claim-verification model.
    """
    if use_attention:
        label, probs, _weights = baseline.verify_claim_attention(
            claim,
            evidence_list,
            tokenizer=claim_tokenizer,
            model=claim_model,
        )
    else:
        label, probs = baseline.verify_claim(
            claim,
            evidence_list,
            tokenizer=claim_tokenizer,
            model=claim_model,
        )
    return label, probs


# -----------------------------------------------------------------------------
# Claim extraction config (env vars)
# -----------------------------------------------------------------------------

# CLAIM_EXTRACTION_MODEL -> Ollama model to use (default: mistral)
# CLAIM_EXTRACTION_MAX   -> max claims to return per passage (default: 10)
EXTRACTION_MODEL = os.getenv("CLAIM_EXTRACTION_MODEL", "mistral")
EXTRACTION_MAX_CLAIMS = int(os.getenv("CLAIM_EXTRACTION_MAX", "10"))

# -----------------------------------------------------------------------------
# Evidence retrieval config (env vars)
# -----------------------------------------------------------------------------

RETRIEVER_BM25_INDEX = os.getenv(
    "CLAIM_RETRIEVAL_BM25_INDEX",
    "api/claim_verification/data/wiki_corpus/bm25_index",
)
RETRIEVER_COLBERT_CHECKPOINT = os.getenv(
    "CLAIM_RETRIEVAL_COLBERT_CHECKPOINT",
    "colbert-ir/colbertv2.0",
)
THENEWSAPI_TOKEN = os.getenv("THENEWSAPI_TOKEN")
ENABLE_LIVE_NEWS = bool(THENEWSAPI_TOKEN)
if not ENABLE_LIVE_NEWS:
    print("[ClaimServer] THENEWSAPI_TOKEN not set; disabling live-news fallback.")

print(f"[ClaimServer] Loading evidence retriever from BM25 index: {RETRIEVER_BM25_INDEX}")
claim_retriever = ClaimEvidenceRetriever(
    bm25_index=RETRIEVER_BM25_INDEX,
    checkpoint=RETRIEVER_COLBERT_CHECKPOINT,
    use_gpu=True,
    batch_size=16,
    bm25_k=200,
    top_k=10,
    evidence_k=5,
    enable_live_news=ENABLE_LIVE_NEWS,
    news_limit=8,
    news_hours_back=72,
    news_language="en",
    allow_untrusted_domains=False,
)
print("[ClaimServer] Evidence retriever loaded!")


URL_LINE_PATTERN = re.compile(r"^\s*URL:\s*\S+\s*$", re.IGNORECASE | re.MULTILINE)
RAW_URL_PATTERN = re.compile(r"https?://\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_evidence_text(text: str) -> str:
    """
    Remove URL-only lines and inline URLs from retrieval passages
    so UI shows readable evidence text.
    """
    if not text:
        return ""

    cleaned = URL_LINE_PATTERN.sub(" ", text)
    cleaned = RAW_URL_PATTERN.sub("", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def sanitize_evidence_list(evidence: List[str]) -> List[str]:
    seen = set()
    cleaned_evidence: List[str] = []

    for item in evidence or []:
        cleaned = clean_evidence_text(item)
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        cleaned_evidence.append(cleaned)

    return cleaned_evidence


def retrieve_single_claim_evidence(claim: str):
    """
    Directly callable ER pipeline function.
    Returns:
      {
        "claim": ...,
        "evidence": [...],
        "confidence": bool,
        "force_nei": bool,
        "source": ...
      }
    """
    result = claim_retriever.retrieve(claim)
    result["evidence"] = sanitize_evidence_list(result.get("evidence", []))
    return result


# -----------------------------------------------------------------------------
# DEMO: stub for claim extraction + evidence retrieval
# -----------------------------------------------------------------------------

def get_claims_and_evidence_for_demo(passage: str) -> List[ClaimEvidence]:

    demo_items = [

    # --- REFUTED (real-world contradiction) ---

    ClaimEvidence(
        claim="5G is a compartmentalized weapons deployment system that masquerades as a benign technological advancement for enhanced communications and faster downloads.",
        evidence=[
            "5G is the fifth generation of wireless communication technology designed to provide faster data speeds and improved connectivity."
        ],
        gold_label="REFUTED",
    ),

    ClaimEvidence(
        claim="5G has the capability to target, acquire and attack vaccinated individuals through the nano-metamaterial antenna in Wuhan coronavirus (COVID-19) vaccines.",
        evidence=[
            "5G is a wireless communication standard used for transmitting data and does not have the capability to target or attack individuals."
        ],
        gold_label="REFUTED",
    ),

    ClaimEvidence(
        claim="COVID-19 vaccines contain nano-metamaterial antennas.",
        evidence=[
            "COVID-19 vaccines do not contain microchips, antennas, or tracking devices and are designed to stimulate an immune response."
        ],
        gold_label="REFUTED",
    ),

    ClaimEvidence(
        claim="5G is a weapon system and a crime against humanity.",
        evidence=[
            "5G is a telecommunications technology used for wireless communication and internet connectivity."
        ],
        gold_label="REFUTED",
    ),

    ClaimEvidence(
        claim="the world is blindly following the plans of the technocratic elite and the military-industrial-pharma complex to terminate large numbers within populations across the world.",
        evidence=[
            "There is no credible evidence that global institutions are coordinating a plan to terminate large populations."
        ],
        gold_label="REFUTED",
    ),

    # --- SUPPORTED (direct factual statements from passage that are actually true) ---

    ClaimEvidence(
        claim="A weapon can be a device, tool or action fashioned to cause physical or psychological harm.",
        evidence=[
            "A weapon is commonly defined as an object or tool used to inflict physical or psychological harm."
        ],
        gold_label="SUPPORTED",
    ),

    ClaimEvidence(
        claim="The Lethal Autonomous Weapons Systems (LAWS) will need 5G networks to maintain geo-position and navigate their environment to the target because these weapons cannot rely simply on satellite communications, which can be affected by inclement weather events and signal latency.",
        evidence=[
            "Satellite communication signals can be disrupted by weather conditions and may experience latency due to long transmission distances."
        ],
        gold_label="SUPPORTED",
    ),

    # --- NOT ENOUGH INFO  ---

    ClaimEvidence(
        claim="The Lethal Autonomous Weapons Systems (LAWS) will need 5G networks to maintain geo-position and navigate their environment to the target.",
        evidence=[],
        gold_label="NOT ENOUGH INFO",
    ),

    ClaimEvidence(
        claim="The prima facie evidence of this globalist depopulation agenda is unequivocal.",
        evidence=[],
        gold_label="NOT ENOUGH INFO",
    ),
]

    return demo_items


# -----------------------------------------------------------------------------
# Mock data (used for claim extraction + optional evidence retrieval)
# -----------------------------------------------------------------------------

MOCK_ITEMS: List[ClaimEvidence] = get_claims_and_evidence_for_demo("demo")
MOCK_CLAIMS: List[str] = [item.claim for item in MOCK_ITEMS]
MOCK_EVIDENCE_MAP: Dict[str, List[str]] = {item.claim: item.evidence for item in MOCK_ITEMS}

DEFAULT_MOCK_EVIDENCE = [
    "This is a placeholder evidence sentence for UI testing.",
    "Another mock evidence sentence that is not sourced.",
]

MAX_CLAIMS = len(MOCK_CLAIMS)
MAX_EVIDENCE = 4


def get_mock_claims(max_claims: int = MAX_CLAIMS) -> List[str]:
    return MOCK_CLAIMS[:max_claims]


def get_mock_evidence(claim: str, max_evidence: int = MAX_EVIDENCE) -> List[str]:
    evidence = MOCK_EVIDENCE_MAP.get(claim, DEFAULT_MOCK_EVIDENCE)
    return evidence[:max_evidence]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "component": "claim_verification"}


@app.post("/extract-claims", response_model=ClaimExtractionResult)
async def extract_claims_endpoint(
    request: PassageRequest,
    model: Optional[str] = None,
    max_claims: Optional[int] = None,
):
    """
    Extract atomic, verifiable claims from a passage using a local Ollama LLM.

    Query params:
      - model:      Ollama model to use (default: env CLAIM_EXTRACTION_MODEL or "mistral").
                    Options: mistral, llama3, gemma:7b, phi3, mixtral
      - max_claims: Max number of claims to return (default: env CLAIM_EXTRACTION_MAX or 10).

    Body:
      { "text": "Your passage here..." }

    Returns:
      { "claims": [...], "model": "mistral", "count": 5 }
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    resolved_model = model or EXTRACTION_MODEL
    resolved_max = max_claims if max_claims is not None else EXTRACTION_MAX_CLAIMS

    try:
        claims = extract_claims(
            passage=text,
            model=resolved_model,
            max_claims=resolved_max,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return ClaimExtractionResult(
        claims=claims,
        model=resolved_model,
        count=len(claims),
    )


@app.post("/retrieve-evidence", response_model=EvidenceRetrievalResult)
async def retrieve_evidence_endpoint(request: ClaimRequest):
    claim = request.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    try:
        result = retrieve_single_claim_evidence(claim)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence retrieval failed: {exc}")

    return EvidenceRetrievalResult(
        claim=result["claim"],
        evidence=result["evidence"],
        confidence=result["confidence"],
        force_nei=result["force_nei"],
        source=result["source"],
    )


@app.post("/verify-claims-from-passage", response_model=List[ClaimVerificationResult])
async def verify_claims_from_passage(
    request: PassageRequest,
    mock_claims: bool = False,
    mock_retrieve: bool = False,
):
    """
    Entry point for the Chrome extension for claim verification.

    Pipeline:
      1. Claim extraction  — mock or real (Ollama, per-sentence with provenance)
      2. Evidence retrieval — mock or real (ColBERT + BM25)
      3. Claim verification — always real model

    Each result carries sentence_id and source_sentence so the frontend can
    group or highlight claims by the sentence they came from, without needing
    a separate join step.

    Query params:
      - mock_claims:   If false, uses extract_claims_with_provenance() for real extraction.
      - mock_retrieve: If true, uses hard-coded mock evidence instead of the retriever.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # ------------------------------------------------------------------
    # Step 1 — Claim extraction
    # Build a flat list of (claim, sentence_id, source_sentence) tuples
    # so provenance travels through the rest of the pipeline as plain data.
    # ------------------------------------------------------------------
    if mock_claims:
        # Mock path: no sentence provenance available, use sentinel None values
        claim_tuples = [
            (claim, None, None) for claim in get_mock_claims()
        ]
    else:
        try:
            extraction = extract_claims_with_provenance(
                passage=text,
                model=EXTRACTION_MODEL,
                max_claims_per_sentence=EXTRACTION_MAX_CLAIMS,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        claim_tuples = [
            (claim, sc.sentence_id, sc.sentence)
            for sc in extraction.sentence_claims
            for claim in sc.claims
        ]

    if not claim_tuples:
        raise HTTPException(status_code=400, detail="No claims extracted from passage.")

    # ------------------------------------------------------------------
    # Steps 2 + 3 — Evidence retrieval and verification
    # ------------------------------------------------------------------
    results: List[ClaimVerificationResult] = []

    for claim, sentence_id, source_sentence in claim_tuples:
        # Evidence retrieval
        if mock_retrieve:
            evidence = get_mock_evidence(claim)
            force_nei = False
        else:
            try:
                retrieval_result = retrieve_single_claim_evidence(claim)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Evidence retrieval failed for claim '{claim}': {exc}",
                )
            evidence = retrieval_result["evidence"]
            force_nei = retrieval_result["force_nei"]

        # Claim verification
        if force_nei:
            label = "NOT ENOUGH INFO"
            probs = {"REFUTED": 0.0, "NOT ENOUGH INFO": 1.0, "SUPPORTED": 0.0}
        else:
            label, probs = verify_single_claim(claim, evidence)

        results.append(
            ClaimVerificationResult(
                claim=claim,
                evidence=evidence,
                label=label,
                probabilities=probs,
                sentence_id=sentence_id,
                source_sentence=source_sentence,
            )
        )

    return results


@app.post("/verify-claims-direct", response_model=List[ClaimVerificationResult])
async def verify_claims_direct(request: ClaimVerificationDirectRequest):
    """
    Optional utility endpoint: pass claims + evidence directly.

    Body:
      {
        "items": [
          { "claim": "...", "evidence": ["...", "..."] },
          ...
        ]
      }

    Useful for testing the verifier independently of extraction/retrieval.
    sentence_id and source_sentence will be null in these responses.
    """
    results: List[ClaimVerificationResult] = []

    for item in request.items:
        if not item.claim.strip():
            continue
        label, probs = verify_single_claim(item.claim, item.evidence)
        results.append(
            ClaimVerificationResult(
                claim=item.claim,
                evidence=item.evidence,
                label=label,
                probabilities=probs,
                # No passage context available in the direct endpoint
                sentence_id=None,
                source_sentence=None,
            )
        )

    return results