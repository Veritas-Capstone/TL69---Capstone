"""
Training wrapper for FEVER claim verification.
"""

from pathlib import Path

from api.claim_verification.training.train_core import train_claim_model

MODEL = "FacebookAI/roberta-large-mnli"
LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}

PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_ROOT / "data" / "processed"
METRICS_DIR = PKG_ROOT / "training_metrics"
MODELS_DIR = PKG_ROOT / "models"


def train_fever(
    init_weights="hf",
    data_set=None,
    data_path=None,
    batch_size=2,
    max_length=256,
    threshold=0.05,
    num_epochs=4,
    patience=2,
    lr=2e-5,
    use_attention_model=True,
    max_evidence=5,
    accum_steps=8,
    num_heads=4,
    mix_config=None,
):
    if data_path is None:
        data_path = DATA_DIR / f"{data_set}.csv"

    return train_claim_model(
        data_set=data_set,
        data_path=data_path,
        label_map=LABEL_MAP,
        model_name=MODEL,
        metrics_dir=METRICS_DIR,
        models_dir=MODELS_DIR,
        init_weights=init_weights,
        batch_size=batch_size,
        max_length=max_length,
        threshold=threshold,
        num_epochs=num_epochs,
        patience=patience,
        lr=lr,
        use_attention_model=use_attention_model,
        max_evidence=max_evidence,
        accum_steps=accum_steps,
        num_heads=num_heads,
        mix_config=mix_config,
    )


if __name__ == "__main__":
    train_fever()
