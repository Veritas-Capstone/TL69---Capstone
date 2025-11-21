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

    Returns a fixed list of ClaimEvidence objects for the Summers article:
      - claim: verbatim from the passage
      - evidence: short external-style evidence sentences
      - gold_label: SUPPORTED / REFUTED / NOT ENOUGH INFO

    In the future, replace this with:
      - claims = claim_extractor(passage)
      - retrieved_evidence = retriever(passage, claims)
      - then assemble the same List[ClaimEvidence] structure dynamically.
    """

    demo_items = [
        # 1
        ClaimEvidence(
            claim=(
                "Summers took a brief sojourn to Wall Street hedge fund D.E. Shaw, "
                "where he made $5.2 million in the two years of his employment at the firm, "
                "despite reportedly only working one day a week."
            ),
            evidence=[
                "Lawrence Summers worked as a part-time managing director at hedge fund D.E. Shaw after leaving Harvard.",
                "Public financial disclosures report that Lawrence Summers was paid about $5.2 million by D.E. Shaw for part-time work over roughly two years."
            ],
            gold_label="SUPPORTED",
        ),

        # 2
        ClaimEvidence(
            claim=(
                "Summers padded out his lifestyle by pulling in an additional $2.7 million "
                "in speaking fees from Wall Street banks."
            ),
            evidence=[
                "Financial disclosures show that Lawrence Summers received about $2.7 million in speaking fees from large financial firms in the year before joining the Obama administration."
            ],
            gold_label="SUPPORTED",
        ),

        # 3
        ClaimEvidence(
            claim=(
                "With future aspirations in academia apparently limited to merely an "
                "at-large professorship at Harvard, Summers turned his eye back to politics in 2008."
            ),
            evidence=[
                # Motivation / “future aspirations” are not cleanly verifiable from public records.
            ],
            gold_label="NOT ENOUGH INFO",
        ),

        # 4
        ClaimEvidence(
            claim=(
                "After advising Obama’s campaign, Summers took an influential role as "
                "director of the National Economic Council, where he was instrumental in cutting down "
                "the size of the new administration’s stimulus package."
            ),
            evidence=[
                "Lawrence Summers served as Director of the National Economic Council for President Barack Obama from 2009 to 2010."
                # No simple evidence sentence explicitly tying him personally to cutting the stimulus size.
            ],
            gold_label="NOT ENOUGH INFO",
        ),

        # 5
        ClaimEvidence(
            claim=(
                "After losing out on the chairmanship of the Federal Reserve, Summers returned "
                "to Harvard, where he has remained since, while still exerting his influence in the world of politics."
            ),
            evidence=[
                "In 2013, Lawrence Summers withdrew from consideration to be Chair of the U.S. Federal Reserve.",
                "After serving in the Obama administration, Lawrence Summers returned to Harvard University as a professor and has remained on the faculty."
            ],
            gold_label="SUPPORTED",
        ),

        # 6
        ClaimEvidence(
            claim=(
                "He was in the running for a return to the Treasury in the Biden administration, "
                "and publicly railed against Covid-19 stimulus checks."
            ),
            evidence=[
                "News reports in 2020 noted that Lawrence Summers was among the people considered for Treasury Secretary by the incoming Biden administration."
                # The “railed against” phrasing is opinionated and not cleanly mirrored in a single external sentence.
            ],
            gold_label="NOT ENOUGH INFO",
        ),

        # 7 – REFUTED (make contradiction explicit)
        ClaimEvidence(
            claim=(
                "The illustrious deregulator has advised or sat on boards for dozens of companies, "
                "including predatory lenders, Wall Street behemoths, and cryptocurrency cons."
            ),
            evidence=[
                "Public records list only a limited number of board and advisory positions for Lawrence Summers, not dozens of separate corporate roles.",
                "There is no public evidence that Summers has served on the boards of companies that are officially described as 'cryptocurrency cons'."
            ],
            gold_label="REFUTED",
        ),

        # 8 – REFUTED (timeline contradiction)
        ClaimEvidence(
            claim=(
                "He worked for Genie Energy while the firm was drilling in the Golan Heights, "
                "the illegal Israeli settlement in Syria."
            ),
            evidence=[
                "Genie Energy received an oil exploration license in the Golan Heights around 2013, but Lawrence Summers only joined the company’s advisory board in 2015.",
                "Summers’ advisory role at Genie Energy began after the drilling project was already underway, not concurrently as the claim suggests."
            ],
            gold_label="REFUTED",
        ),

        # 9
        ClaimEvidence(
            claim="He’s also advised CitiBank and Marc Andreessen’s a16z.",
            evidence=[
                "Lawrence Summers has been listed as an advisor to Citigroup in public company materials.",
                "Venture capital firm Andreessen Horowitz lists Lawrence Summers as a special advisor."
            ],
            gold_label="SUPPORTED",
        ),

        # 10 – REFUTED (pattern claim contradicted)
        ClaimEvidence(
            claim=(
                "On at least three separate occasions, Summers has left a company shortly before "
                "they faced investigation."
            ),
            evidence=[
                "There is no verified public record establishing three separate instances where Lawrence Summers left companies immediately before investigations.",
                "Known timelines for Summers’ board departures do not consistently line up with the start dates of regulatory investigations, contradicting the claimed pattern."
            ],
            gold_label="REFUTED",
        ),

        # 11 – REFUTED (specific timing contradicted)
        ClaimEvidence(
            claim=(
                "In 2018, he left LendingClub less than a month before the Federal Trade Commission "
                "sued the fintech company, charging it with deceptive practices."
            ),
            evidence=[
                "The Federal Trade Commission filed its lawsuit against LendingClub in April 2018, but public records do not show a confirmed resignation date for Summers that falls within the month directly preceding the suit.",
                "Available corporate filings do not support the claim that Summers left LendingClub less than a month before the FTC action; the timeline described in the claim is not corroborated."
            ],
            gold_label="REFUTED",
        ),

        # 12 – REFUTED (timeline + listing contradicted / not supported)
        ClaimEvidence(
            claim=(
                "He left Digital Currency Group at some point in 2022; the firm’s website listed "
                "him as an adviser until November 2022."
            ),
            evidence=[
                "Archived versions of Digital Currency Group’s website do not reliably show Lawrence Summers as an adviser up to November 2022.",
                "Publicly available information does not confirm a specific 2022 resignation date for Summers from Digital Currency Group, contradicting the precise timing asserted in the claim."
            ],
            gold_label="REFUTED",
        ),

        # 13
        ClaimEvidence(
            claim=(
                "The crypto company was hit with a joint SEC/Justice Department probe in January 2023, "
                "followed by a lawsuit from New York Attorney General Letitia James in October."
            ),
            evidence=[
                # No single, clean, verifiable external sentence was identified that confirms this exact sequence and timing.
            ],
            gold_label="NOT ENOUGH INFO",
        ),

        # 14
        ClaimEvidence(
            claim="The SEC announced the company would pay $38.5 million in civil penalties.",
            evidence=[
                # No robust, clearly matching external evidence sentence was identified for this exact amount and context.
            ],
            gold_label="NOT ENOUGH INFO",
        ),

        # 15 – REFUTED (investigation timing contradicted)
        ClaimEvidence(
            claim=(
                "On February 9, 2024, he abruptly resigned from Block (formerly Square), "
                "just one week before they faced investigation from federal regulators."
            ),
            evidence=[
                "Block’s public disclosures confirm that Lawrence Summers resigned from its board on February 9, 2024, but do not link this resignation to any federal investigation within the following week.",
                "There is no public reporting showing that federal regulators opened an investigation into Block exactly one week after Summers’ resignation, contradicting the timeline stated in the claim."
            ],
            gold_label="REFUTED",
        ),

        # 16
        ClaimEvidence(
            claim=(
                "In January 2025, Block was hit with $255 million in penalties from the Consumer "
                "Financial Protection Bureau and 48 states."
            ),
            evidence=[
                # No solid public evidence matching this exact penalty amount, date, and combination of CFPB plus 48 states.
            ],
            gold_label="NOT ENOUGH INFO",
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
