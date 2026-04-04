"""
Load a LoRA fine-tuned model and merge the adapter weights back into the
base DeBERTa model for inference. Can also save the merged model as a
standalone checkpoint.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

LABELS = ["Left", "Center", "Right"]

# paths
BASE_MODEL_PATH = "models/bias_detector"
ADAPTER_PATH = "models/bias_detector_finetuned"
MERGED_OUTPUT_PATH = "models/bias_detector_merged"


def load_finetuned_model(base_path=BASE_MODEL_PATH, adapter_path=ADAPTER_PATH, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model from {base_path}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(base_path, num_labels=3)

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # merge adapter weights into base model (removes adapter overhead at runtime)
    print("Merging adapter weights..")
    model = model.merge_and_unload()
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    print(f"Fine-tuned model loaded on {device}")
    return model, tokenizer


def save_merged_model(base_path=BASE_MODEL_PATH, adapter_path=ADAPTER_PATH,
                      output_path=MERGED_OUTPUT_PATH):
    model, tokenizer = load_finetuned_model(base_path, adapter_path, device="cpu")
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved. Use bias_path='{output_path}' in the pipeline.")


def quick_test(model, tokenizer, device="cuda"):
    examples = [
        ("Left-leaning",
         "Progressive Democrats championed Medicare for All and free college tuition, "
         "arguing that universal healthcare is a fundamental right."),
        ("Center",
         "The Congressional Budget Office released its annual economic outlook on "
         "Wednesday, projecting GDP growth of 2.1% for the coming fiscal year."),
        ("Right-leaning",
         "Conservative lawmakers defended free market principles and called for "
         "reduced government regulation, arguing that individual liberty is paramount."),
    ]

    for expected, text in examples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        pred = LABELS[torch.argmax(probs).item()]
        match = "✓" if pred.lower() in expected.lower() else "✗"
        print(f"  {match} Expected: {expected}, Got: {pred} (L={probs[0]:.3f} C={probs[1]:.3f} R={probs[2]:.3f})")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_finetuned_model(device=device)
    quick_test(model, tokenizer, device)

    # uncomment to save merged model:
    # save_merged_model()