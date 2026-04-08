import os

import pytest

from api.claim_verification.inference import baseline


def _load_sample_rows():
    # One example per label (Supported / Refuted / Not Enough Info).
    return [
        {
            "claim": "Barack Obama was born in Hawaii.",
            "evidence": ["Barack Obama was born in Honolulu, Hawaii in 1961."],
            "label": "SUPPORTED",
        },
        {
            "claim": "Barack Obama was born in Kenya.",
            "evidence": ["Barack Obama was born in Honolulu, Hawaii in 1961."],
            "label": "REFUTED",
        },
        {
            "claim": "Barack Obama was born in Kenya.",
            "evidence": [],
            "label": "NOT ENOUGH INFO",
        },
    ]


def _load_model():
    dataset = os.getenv("CLAIM_MODEL_DATASET", "fever_averitec_er_r50_h4")
    try:
        ckpt = baseline.resolve_latest_checkpoint(dataset)
    except FileNotFoundError:
        ckpt = None

    if ckpt is None:
        raise RuntimeError(
            "Claim model checkpoint not found. "
            "Set CLAIM_MODEL_DATASET or CLAIM_MODEL_CHECKPOINT."
        )

    tokenizer, model = baseline.load_claim_verifier(
        model_name=os.getenv("CLAIM_MODEL_ARCH", baseline.MODEL),
        state_dict_path=ckpt,
    )
    model.eval()
    return tokenizer, model


@pytest.mark.parametrize("row", _load_sample_rows())
def test_claim_veracity_regression(row):
    claim = row["claim"]
    evidence = row["evidence"]
    expected_label = row["label"]

    tokenizer, model = _load_model()
    label, probs = baseline.verify_claim(
        claim,
        evidence,
        tokenizer=tokenizer,
        model=model,
    )

    assert label in baseline.LABELS
    assert set(probs.keys()) == set(baseline.LABELS)
    assert label == expected_label
