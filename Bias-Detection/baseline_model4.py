from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-deberta-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)

text = "The Department of Health and Human Services's Office for Civil Rights has released guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies against refusing to dispense abortion-inducing medication, stipulating that doing so is pregnancy discrimination. That includes discrimination based on current pregnancy, past pregnancy, potential or intended pregnancy, and medical conditions related to pregnancy or childbirth. HHS is committed to ensuring that everyone can access healthcare, free of discrimination,"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    predicted = torch.argmax(logits, dim=1).item()

label_map = {0: "Left", 1: "Center", 2: "Right"}
print("Predicted Bias:", label_map[predicted])

# Show confidence scores
probs = torch.nn.functional.softmax(logits, dim=1)
for label_id, label_name in label_map.items():
    print(f"{label_name}: {probs[0, label_id].item():.4f}")