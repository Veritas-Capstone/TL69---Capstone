#!/usr/bin/env python3
"""
Load Fine-Tuned LoRA Model for Inference
==========================================
Shows how to integrate the LoRA-adapted model back into your
existing pipeline_v2 / BiasDetector workflow.

Usage:
  # Quick test
  python load_finetuned.py

  # Use in your pipeline
  from load_finetuned import load_finetuned_model
  model, tokenizer = load_finetuned_model()
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

LABELS = ["Left", "Center", "Right"]


def load_finetuned_model(
    base_model_path="models/bias_detector",
    adapter_path="models/bias_detector_finetuned",
    device=None,
):
    """
    Load the base model + LoRA adapter for inference.

    Returns (model, tokenizer) ready for prediction.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=3,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge LoRA weights into the base model for faster inference
    # (no adapter overhead at runtime)
    print("Merging adapter weights...")
    model = model.merge_and_unload()

    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print(f"✓ Fine-tuned model loaded on {device}")
    return model, tokenizer


def save_merged_model(
    base_model_path="models/bias_detector",
    adapter_path="models/bias_detector_finetuned",
    output_path="models/bias_detector_merged",
):
    """
    Merge LoRA weights and save as a standalone model.
    This creates a normal HuggingFace model (no peft dependency needed).

    After running this, you can update your pipeline to just point
    at the merged model path and everything works as before.
    """
    model, tokenizer = load_finetuned_model(base_model_path, adapter_path, device="cpu")

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Merged model saved!")
    print(f"\n  To use in your pipeline, change:")
    print(f"    bias_path='models/bias_detector'")
    print(f"  to:")
    print(f"    bias_path='{output_path}'")


def quick_test(model, tokenizer, device="cuda"):
    """Quick sanity check on hand-picked examples."""
    examples = [
        ("Left-leaning",
         "Progressive Democrats championed Medicare for All and free college tuition, "
         "arguing that universal healthcare is a fundamental right that the richest "
         "nation on Earth can and should guarantee to every citizen."),
        ("Center",
         "The Congressional Budget Office released its annual economic outlook on "
         "Wednesday, projecting GDP growth of 2.1% for the coming fiscal year and "
         "noting that the federal deficit is expected to widen modestly."),
        ("Right-leaning",
         "Conservative lawmakers defended free market principles and called for "
         "reduced government regulation, arguing that individual liberty and "
         "personal responsibility are the cornerstones of American prosperity."),
    ]

    print(f"\n{'═' * 60}")
    print(f"  QUICK TEST")
    print(f"{'═' * 60}")

    for expected, text in examples:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        pred = LABELS[pred_idx]

        probs_str = "  ".join(f"{l}: {probs[i]:.3f}" for i, l in enumerate(LABELS))
        match = "✓" if pred.lower() in expected.lower() else "✗"
        print(f"\n  [{expected}] {match}")
        print(f"  Predicted: {pred}  |  {probs_str}")
        print(f"  Text: {text[:80]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/bias_detector")
    parser.add_argument("--adapter", default="models/bias_detector_finetuned")
    parser.add_argument("--merge", action="store_true",
                        help="Merge and save as standalone model")
    parser.add_argument("--output", default="models/bias_detector_merged")
    args = parser.parse_args()

    if args.merge:
        save_merged_model(args.base_model, args.adapter, args.output)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_finetuned_model(args.base_model, args.adapter, device)
        quick_test(model, tokenizer, device)