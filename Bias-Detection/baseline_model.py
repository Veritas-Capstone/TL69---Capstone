import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import json
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, classification_report



# Step 1 ------------ Load Dataset ------------
df = pd.read_csv("datasets/allsides_balanced_news_headlines-texts.csv")  # Replace with your dataset path
texts = df["text"].astype(str).tolist()
labels = df["bias_rating"].astype(str).str.lower().tolist()

# Map label text to integers
label_map = {"left": 0, "center": 1, "right": 2}
true_labels = [label_map[label] for label in labels]



# Step 2 ------------ Load Pretrained Model ------------
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512)
result = classifier("A spate of shootings throughout the US left more than 150 people wounded and nearly two-dozen dead so far this weekend, including 67 gunshot victims in Chicago over a blood-splattered weekend, according to reports. Police said 13 peopl were killed in separate shootings in the Second City, including a 7-year-old girl who was at a Fourth of July party in the city’s Austin neighborhood and a 14-year-old boy, the Chicago Sun-Times reported Sunday. Tonight, a 7-year-old...")
print(result)
