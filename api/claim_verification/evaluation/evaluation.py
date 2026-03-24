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
# evaluation.py

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
# from training_fever import train_fever
from .evaluating_averitec import eval_averitec
import timeit

PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_ROOT / "data" 
EVAL_OUT_DIR = PKG_ROOT / "eval_metrics"
MODELS_DIR = PKG_ROOT / "models"

EVAL_PLAN = [
    # Example 1: HF baseline on AveriTeC
    # {
    #     "dataset": "fever_train_claims_20",
    #     "dataset_path": DATA_DIR / "processed" / f"fever_train_claims_20.csv",   # this needs to be baked invariant to cwd
    #     "mode": "eval",
    #     "init_from": "hf",          # <- baseline
    # },
    {
        "dataset": "fever",
        "dataset_path": DATA_DIR / "processed" / f"fever_train_claims_20.csv", 
        "mode": "eval",
        "init_from": "fever_averitec_mix",         # <- use ../models/fever_train_claims_80/latest.pt         
    },
    # {
    #     "dataset": "fever",
    #     "dataset_path": DATA_DIR / "processed" / f"fever_train_claims_sample.csv",  
    #     "mode": "eval",
    #     "init_from": "hf",         
    # },
    # Example 2 (optional): evaluate AveriTeC model trained on AveriTeC
    {
        "dataset": "averitec",
        "dataset_path": DATA_DIR / "processed" / f"averitec_20.csv",
        "mode": "eval",
        "init_from": "fever_averitec_mix",   # <- use ../models/fever_train_claims_80/latest.pt
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
    for step in EVAL_PLAN:
            dataset = step["dataset"]
            mode = step.get("mode", "eval")  # default: 'eval'
            dataset_ckpt = get_ckpt_path(dataset)
            init_weights = resolve_init_weights(step)
            data_path = step.get("dataset_path") or (DATA_DIR / "processed" / f"{dataset}.csv")

            print(f"\n=== EVAL Dataset: {dataset} ===")
            print(f"  mode:       {mode}")
            print(f"  init_from:  {step.get('init_from', 'hf')} -> {init_weights}")
            print(f"  dataset ckpt path (for this dataset): {dataset_ckpt}")
            print(f"  data path:  {data_path}")

            if mode != "eval":
                raise ValueError(f"Unknown mode={mode!r} for dataset={dataset!r}; expected 'eval'.")

            # Both AveriTeC and FEVER-family splits use the same evaluator today
            if dataset != "averitec" and "fever_train_claims" not in dataset and dataset != "fever":
                raise ValueError(f"Unknown dataset {dataset!r} for evaluation")

            eval_averitec(
                init_weights=init_weights,
                data_set=dataset,
                data_path=data_path,
                output_root=EVAL_OUT_DIR,
                num_heads=step.get("num_heads", 4),
            )


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print("Evaluation took", end - start, "seconds")
