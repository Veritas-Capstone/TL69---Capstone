from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from torch import argmax

model = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-politics")
tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")

text = "The Department of Health and Human Services's Office for Civil Rights has released guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies against refusing to dispense abortion-inducing medication, stipulating that doing so is pregnancy discrimination. That includes discrimination based on current pregnancy, past pregnancy, potential or intended pregnancy, and medical conditions related to pregnancy or childbirth. HHS is committed to ensuring that everyone can access healthcare, free of discrimination,"
# inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# with torch.no_grad():
#     logits = model(**inputs).logits
#     predicted = torch.argmax(logits, dim=1).item()

label_map = {0: "Left", 1: "Center", 2: "Right"}
# print("Predicted Bias:", label_map[predicted])

tokens = tokenizer(text, return_tensors="pt")
output = model(**tokens)
logits = output.logits

political_leaning = argmax(logits, dim=1).item()
probabilities = softmax(logits, dim=1)
score = probabilities[0, political_leaning].item()
print(label_map[int(political_leaning)], score)