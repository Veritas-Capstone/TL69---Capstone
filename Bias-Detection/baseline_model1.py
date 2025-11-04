from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

text = "The action comes a few weeks after the Supreme Court overturned Roe v. Wade and gave more than a dozen states a green light to ban abortion, and aims to respond to a wave of reports that pharmacies in those states are refusing to not only fill prescriptions for abortion medication, but also for common medications such as antibiotics and blood pressure treatments."

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")


inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([0])
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# [0] -> left 
# [1] -> center
# [2] -> right
print(logits.softmax(dim=-1)[0].tolist()) 