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
from training_fever import train_fever
from training_averitec import train_averitec

MODELS_DIR = Path("../models")

TRAIN_PLAN = [
    {
        "dataset": "fever_train_claims_80",
        "mode": "retrain",       
        "init_from": "hf",  
    },
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

        if "fever_train_claims" in dataset:
            train_fever(init_weights=init_weights, data_set=dataset)
        elif dataset == "averitec":
            train_averitec(init_weights=init_weights, data_set=dataset)
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")


if __name__ == "__main__":
    main()
