"""
FastAPI server for CLAIM VERIFICATION demo.

Run from repo root:
    uvicorn server.claim_server:app --reload --port 8001

This server is independent of the bias-detection server.py
and only handles claim verification.
"""

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.claim_verification.inference import baseline

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


class ClaimVerificationResult(BaseModel):
    claim: str
    evidence: List[str]
    label: str
    probabilities: Dict[str, float]


# Optional: direct endpoint where extension (or you) pass claims+evidence directly
class ClaimVerificationDirectRequest(BaseModel):
    items: List[ClaimEvidence]


# -----------------------------------------------------------------------------
# Claim verification model
# -----------------------------------------------------------------------------

# Configure via env vars:
#   CLAIM_MODEL_DATASET   -> dataset folder under api/claim_verification/models (loads latest.pt)
#   CLAIM_MODEL_CHECKPOINT-> explicit path to a .pt state_dict
#   CLAIM_MODEL_ARCH      -> HF model id / local HF directory for the base architecture
MODEL_ARCH = os.getenv("CLAIM_MODEL_ARCH", baseline.MODEL)
MODEL_DATASET = os.getenv("CLAIM_MODEL_DATASET", "fever_train_claims_80")
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

    Strategy (simple, demo-friendly):
      - Build "Claim: ...\nEvidence: ..." pairs
      - Run all pairs through the model
      - Aggregate probabilities across evidence (see baseline.aggregate)
      - Return (label_string, {label: prob})
    """
    label, probs = baseline.verify_claim(
        claim,
        evidence_list,
        tokenizer=claim_tokenizer,
        model=claim_model,
    )
    return label, probs


# -----------------------------------------------------------------------------
# DEMO: stub for claim extraction + evidence retrieval
# -----------------------------------------------------------------------------

def get_claims_and_evidence_for_demo(passage: str) -> List[ClaimEvidence]:
    """
    DEMO VERSION:

    Right now we ignore the `passage` content and return a hand-picked
    list of claims + evidence that you define for your demo article.

    In the future, replace this with:
      - claim_extractor(passage)
      - retriever(passage, claims)
      returning the same List[ClaimEvidence] structure.
    """

    demo_items = [
        # Example – replace these with your actual demo claims/evidence:
        ClaimEvidence(
            claim="The Eiffel Tower is located in Paris, France.",
            evidence=[
                "The Eiffel Tower is located in Paris, France."
            ],
        ),
        ClaimEvidence(
            claim="Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            evidence=[
                "He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam."
            ],
        ),
    ]

    return demo_items


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "component": "claim_verification"}


@app.post("/verify-claims-from-passage", response_model=List[ClaimVerificationResult])
async def verify_claims_from_passage(request: PassageRequest):
    """
    Entry point for the Chrome extension for claim verification.

    CURRENT (demo):
      - Receives `text` (highlighted passage) from the extension.
      - Uses get_claims_and_evidence_for_demo() to get claim+evidence pairs.
      - Runs claim verification on each pair.
      - Returns labels + probabilities.

    FUTURE:
      - get_claims_and_evidence_for_demo(passage) will be replaced with your real
        claim-extraction + evidence-retrieval pipeline that *uses* `request.text`.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    items = get_claims_and_evidence_for_demo(text)

    results: List[ClaimVerificationResult] = []
    for item in items:
        label, probs = verify_single_claim(item.claim, item.evidence)
        results.append(
            ClaimVerificationResult(
                claim=item.claim,
                evidence=item.evidence,
                label=label,
                probabilities=probs,
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
          {
            "claim": "...",
            "evidence": ["...", "..."]
          },
          ...
        ]
      }

    Useful for:
      - testing the verifier independently of extraction/retrieval
      - quick local experiments in a notebook / Postman.
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
            )
        )

    return results
