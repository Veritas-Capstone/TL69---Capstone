# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
# training_order.py

from pathlib import Path

# Use absolute package imports so this works as:
#   python -m api.claim_verification.training.training_order
# from repo root. Avoid sys.path hacks.
from api.claim_verification.training.training_fever import train_fever
from api.claim_verification.training.training_averitec import train_averitec

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PKG_ROOT / "models"

TRAIN_PLAN = [
    # {
    #     "dataset": "fever_er_r70_h4",
    #     "trainer": "fever",
    #     "data_path": PKG_ROOT / "data" / "processed" / "fever_train_claims_80_er.csv",
    #     "mode": "retrain",
    #     "init_from": "hf",
    #     "use_attention_model": True,
    #     "num_heads": 4,
    #     "num_epochs": 4,
    #     # "mix_config": {
    #     #     "aux_data_path": PKG_ROOT / "data" / "processed" / "averitec_80_er.csv",
    #     #     "ratio_main": 0.7,
    #     #     "ratio_aux": 0.3,
    #     #     "balance_labels_main": False,
    #     #     "balance_labels_aux": False,
    #     #     "label_weights_aux": None,
    #     #     "epoch_size": 15000,
    #     #     "val_size": 0.2,
    #     # },
    # },
    # {
    #     "dataset": "fever_averitec_er_r50_BL_h4",
    #     "trainer": "fever",
    #     "data_path": PKG_ROOT / "data" / "processed" / "fever_train_claims_80_er.csv",
    #     "mode": "retrain",
    #     "init_from": "hf",
    #     "num_epochs": 4,
    #     "use_attention_model": True,
    #     "num_heads": 4,
    #     "mix_config": {
    #         "aux_data_path": PKG_ROOT / "data" / "processed" / "averitec_80_er.csv",
    #         "ratio_main": 0.5,
    #         "ratio_aux": 0.5,
    #         "balance_labels_main": False,
    #         "balance_labels_aux": True,
    #         "label_weights_aux": None,
    #         "epoch_size": 12000,
    #         "val_size": 0.2,
    #     },
    # },
    {
    "dataset": "fever_averitec_er_r50_h4",
    "trainer": "fever",
    "data_path": PKG_ROOT / "data" / "processed" / "fever_train_claims_80_er.csv",
    "mode": "retrain",
    "init_from": "hf",
    "num_epochs": 4,
    "use_attention_model": True,
    "num_heads": 4,
    "mix_config": {
        "aux_data_path": PKG_ROOT / "data" / "processed" / "averitec_80_er.csv",
        "ratio_main": 0.5,
        "ratio_aux": 0.5,
        "balance_labels_main": False,
        "balance_labels_aux": False,
        "label_weights_aux": {
            "SUPPORTED": 1.3,
            "NOT ENOUGH INFO": 1.1,
            "REFUTED": 1.0},
        "epoch_size": 12000,
        "val_size": 0.2,
    },
    },
#     {
#     "dataset": "fever_averitec_new_er_r50_h4_k1",
#     "trainer": "fever",
#     "data_path": PKG_ROOT / "data" / "processed" / "fever_train_claims_80_er.csv",
#     "mode": "retrain",
#     "init_from": "hf",
#     "num_epochs": 4,
#     "use_attention_model": True,
#     "num_heads": 4,
#     "max_evidence": 1,
#     "mix_config": {
#         "aux_data_path": PKG_ROOT / "data" / "processed" / "averitec_80_er.csv",
#         "ratio_main": 0.5,
#         "ratio_aux": 0.5,
#         "balance_labels_main": False,
#         "balance_labels_aux": False,
#         "label_weights_aux": {
#             "SUPPORTED": 1.3,
#             "NOT ENOUGH INFO": 1.1,
#             "REFUTED": 1.0},
#         "epoch_size": 12000,
#         "val_size": 0.2,
#     },
# },
]


def get_ckpt_path(dataset):
    return MODELS_DIR / dataset / "latest.pt"


def resolve_init_weights(step_cfg):
    init_from = step_cfg.get("init_from", "hf")

    # Base HF model
    if init_from == "hf":
        return "hf"

    ckpt = get_ckpt_path(init_from)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Expected checkpoint for dataset '{init_from}' at {ckpt}, but it does not exist."
        )
    return ckpt


def main():
    for step in TRAIN_PLAN:
        dataset = step["dataset"]
        mode = step.get("mode", "train")  # default: 'train'
        dataset_ckpt = get_ckpt_path(dataset)

        init_weights = resolve_init_weights(step)

        print(f"\n=== Dataset: {dataset} ===")
        print(f"  mode:       {mode}")
        print(f"  init_from:  {step.get('init_from', 'hf')} -> {init_weights}")
        print(f"  dataset ckpt path: {dataset_ckpt}")

        if mode == "skip":
            print(f"  Mode=skip: skipping training for '{dataset}'.")
            continue

        elif mode == "train":
            if dataset_ckpt.exists():
                print(f"  '{dataset}' already has {dataset_ckpt}. Mode='train' -> skipping.")
                continue
            else:
                print(f"  No existing checkpoint for '{dataset}'. Training from {init_weights}.")

        elif mode == "retrain":
            # Always train, ignoring existing checkpoint
            if dataset_ckpt.exists():
                print(f"  Retraining '{dataset}' from {init_weights}, ignoring {dataset_ckpt}.")
            else:
                print(f"  Retraining '{dataset}' from {init_weights} (no prior ckpt).")

        else:
            raise ValueError(f"Unknown mode={mode!r} for dataset={dataset!r}")

        train_kwargs = {
            "data_set": dataset,
            "init_weights": init_weights,
        }
        for key in [
            "data_path",
            "batch_size",
            "max_length",
            "threshold",
            "num_epochs",
            "patience",
            "lr",
            "use_attention_model",
            "max_evidence",
            "accum_steps",
            "num_heads",
            "mix_config",
        ]:
            if key in step:
                train_kwargs[key] = step[key]

        trainer = step.get("trainer")
        if trainer is None:
            if "fever" in dataset:
                trainer = "fever"
            elif "averitec" in dataset:
                trainer = "averitec"
            else:
                raise ValueError(f"Unknown dataset {dataset!r}; add 'trainer' to TRAIN_PLAN.")

        if trainer == "fever":
            train_fever(**train_kwargs)
        elif trainer == "averitec":
            train_averitec(**train_kwargs)
        else:
            raise ValueError(f"Unknown trainer {trainer!r}")


if __name__ == "__main__":
    main()
