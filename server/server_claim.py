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
MODEL_DATASET = os.getenv("CLAIM_MODEL_DATASET", "fever_averitec_mix")
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
        claim="satellite communications can be affected by inclement weather events and signal latency.",
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


@app.post("/verify-claims-from-passage", response_model=List[ClaimVerificationResult])
async def verify_claims_from_passage(
    request: PassageRequest,
    mock_claims: bool = True,
    mock_retrieve: bool = True,
):
    """
    Entry point for the Chrome extension for claim verification.

    Pipeline:
      - Claim extraction (currently mock only)
      - Evidence retrieval (mock via ?mock_retrieve=true, or empty if false)
      - Claim verification (always real model)
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if mock_claims:
        claims = get_mock_claims()
    else:
        claims = []  # TODO: implement claim extraction and call here

    if not claims:
        raise HTTPException(status_code=400, detail="No claims extracted from passage.")

    results: List[ClaimVerificationResult] = []
    for claim in claims:
        if mock_retrieve:
            evidence = get_mock_evidence(claim)
        else:
            evidence = []
        label, probs = verify_single_claim(claim, evidence)
        results.append(
            ClaimVerificationResult(
                claim=claim,
                evidence=evidence,
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
