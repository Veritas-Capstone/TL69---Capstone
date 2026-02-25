from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

os.makedirs("models", exist_ok=True)

print("Downloading models...")

# 1. Bias Detection Model
print("\n1/2 Downloading bias detector...")
print("  Model: matous-volf/political-leaning-deberta-large")

bias_model = AutoModelForSequenceClassification.from_pretrained(
    "matous-volf/political-leaning-deberta-large",
    output_attentions=True
)

print("  Tokenizer: microsoft/deberta-v3-large")
bias_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

bias_model.save_pretrained("models/bias_detector")
bias_tokenizer.save_pretrained("models/bias_detector")
print("✓ Saved to models/bias_detector/")

# 2. Politicalness Filter - CORRECTED
print("\n2/2 Downloading politicalness filter...")
print("  Model: mlburnham/Political_DEBATE_large_v1.0")

pol_model = AutoModelForSequenceClassification.from_pretrained(
    "mlburnham/Political_DEBATE_large_v1.0"
)
pol_tokenizer = AutoTokenizer.from_pretrained(
    "mlburnham/Political_DEBATE_large_v1.0"
)

pol_model.save_pretrained("models/politicalness_filter")
pol_tokenizer.save_pretrained("models/politicalness_filter")
print("✓ Saved to models/politicalness_filter/")

print("\n" + "="*60)
print("✓ Both models downloaded successfully!")
print("="*60)
print(f"\nBias detector: {bias_model.num_parameters():,} parameters")
print(f"Politicalness filter: {pol_model.num_parameters():,} parameters")