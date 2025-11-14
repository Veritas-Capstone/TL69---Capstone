"""
FastAPI server for bias detection model
Run with: uvicorn server:app --reload --port 8000

This script uses Model 4 as the baseline and takes in as input an article, 
splits it into sentences, feeds each sentence to the model, and aggregates 
the predicted bias for each sentence.

Todo-> 
1) Replace Naive Sentence splitting with a better algorithm, perhaps one from 
   a real sentence segmentation library to prevent sentences real context being 
   cut off midway 
2) Merge small sentences (connected to part 1) 
3) Introduce a weighted aggregate system instead of majority voting (essentially 
   give less weight to low confidence results), maybe even ignore them completely outright 
4) ADVANCED : Weight quotes and counterpoints less as the model might think they 
   are part of the problem when actually they aren't 
5) Smoothing across consecutive sentences (CHAT RECOMMENDATION) 
6) 
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from typing import List, Dict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# -----------------------
# Sentence Splitting
# -----------------------
def SplitIntoSentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# -----------------------
# Load Model Once
# -----------------------
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "matous-volf/political-leaning-deberta-large"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        use_fast=False
    )
    model.eval()
    return model, tokenizer


# -----------------------
# Sentence-Level Prediction
# -----------------------
def baseline_bias_detection(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    label_map = {0: "Left", 1: "Center", 2: "Right"}

    print(f"Sentence: {sentence}")
    print("Predicted Bias:", label_map[predicted])
    print("Probabilities:", probs[0].tolist(), "\n")

    return label_map[predicted], probs[0]


# -----------------------
# Weighted Aggregate
# -----------------------
def AggregateBiasScores(probabilities_list):
    """
    probabilities_list = [tensor([0.8, 0.1, 0.1]), tensor([0.2,0.3,0.5]), ...]
    We sum all probability vectors and choose the highest total.
    """
    total = torch.stack(probabilities_list).sum(dim=0)
    idx = torch.argmax(total).item()

    label_map = {0: "Left", 1: "Center", 2: "Right"}
    return label_map[idx], total


# ======================
# FastAPI Application
# ======================

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your extension's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
print("Loading models...")
model, tokenizer = load_model()
print("Models loaded successfully!")


class AnalysisRequest(BaseModel):
    text: str


class SentenceBias(BaseModel):
    text: str
    category: str
    description: str
    bias: str
    confidence: float
    probabilities: Dict[str, float]


class AnalysisResponse(BaseModel):
    overall_bias: str
    overall_probabilities: Dict[str, float]
    sentences: List[SentenceBias]
    checks: int
    issues: int


def categorize_bias(bias: str, confidence: float) -> tuple:
    """Helper function to create UI-friendly labels"""
    if confidence < 0.5:
        category = "Neutral/Balanced"
        description = "No strong political bias detected in this statement."
    elif bias == "Left":
        category = "Left-leaning"
        description = "This statement shows left-wing political perspective or framing."
    elif bias == "Right":
        category = "Right-leaning"
        description = "This statement shows right-wing political perspective or framing."
    else:
        category = "Centrist"
        description = "This statement presents a balanced or centrist perspective."
    
    return category, description


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text for political bias"""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use your existing functions
        sentences = SplitIntoSentences(text)
        
        if not sentences:
            raise HTTPException(status_code=400, detail="Could not split text into sentences")
        
        # Analyze each sentence
        probability_vectors = []
        sentence_results = []
        
        for sentence in sentences:
            predicted_bias, probs = baseline_bias_detection(model, tokenizer, sentence)
            probability_vectors.append(probs)
            
            confidence = float(probs.max())
            category, description = categorize_bias(predicted_bias, confidence)
            
            sentence_results.append(SentenceBias(
                text=sentence,
                category=category,
                description=description,
                bias=predicted_bias,
                confidence=confidence,
                probabilities={
                    "Left": float(probs[0]),
                    "Center": float(probs[1]),
                    "Right": float(probs[2])
                }
            ))
        
        # Use your aggregation function
        overall_bias, totals = AggregateBiasScores(probability_vectors)
        
        # Normalize to get average probabilities
        avg_probs = totals / len(probability_vectors)
        
        # Count checks and issues
        checks = len(sentences)
        issues = sum(1 for s in sentence_results if s.confidence > 0.6 and s.bias != "Center")
        
        return AnalysisResponse(
            overall_bias=overall_bias,
            overall_probabilities={
                "Left": float(avg_probs[0]),
                "Center": float(avg_probs[1]),
                "Right": float(avg_probs[2])
            },
            sentences=sentence_results,
            checks=checks,
            issues=issues
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


# -----------------------
# Main Driver (for standalone testing)
# -----------------------
def main():
    """Original test function - can still be used for testing"""
    text = (
        "The Department of Health and Human Services's Office for Civil Rights has released "
        "guidelines reinforcing the Obamacare law that warns more than 60,000 U.S. pharmacies "
        "against refusing to dispense abortion-inducing medication, stipulating that doing so "
        "is pregnancy discrimination. That includes discrimination based on current pregnancy, "
        "past pregnancy, potential or intended pregnancy, and medical conditions related to "
        "pregnancy or childbirth. HHS is committed to ensuring that everyone can access "
        "healthcare, free of discrimination."
    )

    model_test, tokenizer_test = load_model()

    sentences = SplitIntoSentences(text)

    probability_vectors = []

    for sentence in sentences:
        _, probs = baseline_bias_detection(model_test, tokenizer_test, sentence)
        probability_vectors.append(probs)

    overall_bias, totals = AggregateBiasScores(probability_vectors)

    print("======== FINAL RESULT ========")
    print("Overall Predicted Bias:", overall_bias)
    print("Aggregated Probabilities:", totals.tolist())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)