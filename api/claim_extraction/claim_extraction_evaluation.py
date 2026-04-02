import ollama
import json
import time
import timeit
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Visuzlization
import matplotlib.pyplot as plt
import numpy as np

def run_model(model, prompt):
    # TODO: Switch this over to taking a claim extractor instance as param and running it, but for prelim, we duping the code
    start = time.time()
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options={"temperature": 0},
        format='json'
    )
    
    latency = time.time() - start
    
    try:
        claims = json.loads(response['message']['content'])
        valid_json = True
    except:
        claims = []
        valid_json = False
    
    return claims, latency, valid_json


def normalize(text):
    return text.lower().strip()

def evaluate(predicted, ground_truth):
    predicted_norm = [normalize(p) for p in predicted]
    gt_norm = [normalize(g) for g in ground_truth]
    
    correct = sum(1 for p in predicted_norm if p in gt_norm)
    
    precision = correct / len(predicted_norm) if predicted_norm else 0
    recall = correct / len(gt_norm) if gt_norm else 0
    
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0)
    
    return precision, recall, f1

def return_Prompt_gt(idx):
    return 


################ Main
models = ["mistral", "mixtral", "gemma:7b", "llama3", "phi3"]
NUM_TRIALS = 5
results = {}

# TODO: Eventually I'll have to offload these to a seperate txt or json file once I have more than 2
passage = ["The Earth revolves around the Sun. Mars has two moons called Phobos and Deimos. Jupiter is the largest planet in the Solar System. Saturn is known for its prominent ring system."
           , """In 1969, NASA successfully landed the Apollo 11 spacecraft on the Moon. 
            Neil Armstrong became the first human to walk on the lunar surface, and Buzz Aldrin joined him shortly after. 
            The mission was launched from Kennedy Space Center in Florida on July 16, 1969. 
            The Saturn V rocket used for the launch remains one of the most powerful rockets ever built. 
            The Moon landing was broadcast live to an estimated 600 million people worldwide. 

            Some critics argue that the Moon landing was staged, but no credible evidence has been found to support this claim. 
            The United States spent approximately $25.4 billion on the Apollo program between 1961 and 1973. 
            The Apollo program resulted in significant technological advancements, including improvements in computing and materials science. 
            The Soviet Union never successfully landed a cosmonaut on the Moon. 
            Today, NASA is developing the Artemis program, which aims to return humans to the Moon by the late 2020s."""
    ]

ground_truths = [
    ["The Earth revolves around the Sun.", 
     "Mars has two moons called Phobos and Deimos.", 
     "Jupiter is the largest planet in the Solar System.", 
     "Saturn is known for its prominent ring system."],
     [
        "NASA successfully landed the Apollo 11 spacecraft on the Moon in 1969.",
        "Neil Armstrong was the first human to walk on the Moon.",
        "Buzz Aldrin walked on the Moon shortly after Neil Armstrong.",
        "Apollo 11 was launched from Kennedy Space Center in Florida on July 16, 1969.",
        "The Saturn V rocket is one of the most powerful rockets ever built.",
        "The Moon landing was broadcast live to approximately 600 million people worldwide.",
        "No credible evidence has been found supporting the claim that the Moon landing was staged.",
        "The United States spent approximately $25.4 billion on the Apollo program between 1961 and 1973.",
        "The Apollo program led to technological advancements in computing.",
        "The Apollo program led to technological advancements in materials science.",
        "The Soviet Union did not successfully land a cosmonaut on the Moon.",
        "NASA is developing the Artemis program.",
        "The Artemis program aims to return humans to the Moon by the late 2020s."
    ]
]

EXTRACTION_PROMPT = f"""Extract atomic, self-contained, independently verifiable factual claims.
Split compound statements into separate claims.
Do not include opinions.
Return a JSON list of strings.

Passage:
{passage[0]}
"""

ground_truth = ground_truths[0]



for model in models:
    precisions = []
    recalls = []
    f1s = []
    latencies = []
    valid_json_count = 0
    
    for _ in range(NUM_TRIALS):
        claims, latency, valid = run_model(model, EXTRACTION_PROMPT)
        
        precision, recall, f1 = evaluate(claims, ground_truth)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        latencies.append(latency)
        
        if valid:
            valid_json_count += 1
    
    results[model] = {
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "latency_mean": np.mean(latencies),
        "json_valid_rate": valid_json_count / NUM_TRIALS
    }

print(results)