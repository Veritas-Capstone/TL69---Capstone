"""
FastAPI server for Veritas Bias Detection

Integrates the two-pass BiasDetector pipeline:
    Pass 1: Politicalness filter (Political DEBATE NLI)
    Pass 2: Sentence-level sliding window bias prediction
        + Gradient×Input explainability

Sliding Window Approach:
Instead of feeding isolated sentences to the model (too short,
model trained on full articles), we feed a window of surrounding
sentences for each target sentence. E.g. for sentence 3:
    Window = [sent1, sent2, SENT3, sent4, sent5]
The model gets ~100-150 tokens of context, and the prediction
is attributed to the center sentence.

Run with: uvicorn server:app --reload --port 8000

Models loaded locally from:
- models/bias_detector          (DeBERTa political leaning)
- models/politicalness_filter   (Political DEBATE NLI)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
import re
import time
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
LABELS = ["Left", "Center", "Right"]

STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "can", "that", "this", "these", "those", "it", "its", "not",
    "no", "so", "if", "then", "than", "too", "very", "just",
    "about", "up", "out", "also", "as", "from", "into", "he",
    "she", "they", "we", "you", "his", "her", "their", "our",
    "my", "me", "him", "them", "us", "who", "which", "what",
    "when", "where", "how", "all", "each", "every", "both",
    "more", "most", "other", "some", "such", "only", "own",
    "same", "any", "there", "here", "said", "says", "new",
})

PUNCTUATION = frozenset({
    ".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]",
    "-", "\u2013", "\u2014", "\u2581.", "\u2581,", "\u2581!", "\u2581?", "\u2581;", "\u2581:", "\u2581-",
})

POLITICALNESS_HYPOTHESES = [
    "This text is about politics or government policy.",
    "This text discusses a politically controversial or partisan topic.",
    "This text is about legislation, regulation, or political debate.",
    "This text discusses government officials or political parties.",
]

# Sliding window: how many sentences before/after the target
WINDOW_RADIUS = 2


# Bias detection
class BiasDetector:
    def __init__(self, bias_path="models/bias_detector",
        pol_path="models/politicalness_filter",
        device=None, use_fp16=True,
        pol_threshold=0.5, window_radius=WINDOW_RADIUS):

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pol_threshold = pol_threshold
        self.window_radius = window_radius
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        dtype_str = "FP16" if self.use_fp16 else "FP32"
        print(f"Device: {self.device}  |  Precision: {dtype_str}")

        # Load bias model locally
        print(f"Loading bias model from {bias_path} ...")
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(bias_path)
        self.bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
        self.bias_model.eval().to(self.device)
        if self.use_fp16:
            self.bias_model.half()

        # Load politicalness model locally
        print(f"Loading politicalness model from {pol_path} ...")
        self.pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
        self.pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
        self.pol_model.eval().to(self.device)
        if self.use_fp16:
            self.pol_model.half()

        label2id = self.pol_model.config.label2id
        self.entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))

        self._embedding_layer = self._find_embedding_layer(self.bias_model)
        if self._embedding_layer is None:
            print("Warning: Could not locate embedding layer - explainability disabled")

        print("Models loaded successfully\n")

    @staticmethod
    def _find_embedding_layer(model):
        for attr in ("deberta", "bert", "roberta", "distilbert"):
            backbone = getattr(model, attr, None)
            if backbone is not None:
                return backbone.embeddings.word_embeddings
        for name, module in model.named_modules():
            if "word_embedding" in name.lower() and hasattr(module, "weight"):
                return module
        return None

    # Politicalness Filter

    def check_politicalness(self, text):
        pairs = [(text[:1024], h) for h in POLITICALNESS_HYPOTHESES]
        inputs = self.pol_tokenizer(
            pairs, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.pol_model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1)
        best_conf = float(probs[:, self.entail_idx].max())
        return best_conf > self.pol_threshold, best_conf

    # Sentence-Level Sliding Window Prediction

    def analyze_sentences(self, sentences, explain=True):
        """
        For each target sentence i, build a context window:
            window = sentences[max(0, i-R) : min(n, i+R+1)]
        Feed the window to the model, attribute prediction to sentence i.

        Example with R=2:
            Sent 0: [0][1][2]           -> prediction for 0
            Sent 1: [0][1][2][3]        -> prediction for 1
            Sent 2: [0][1][2][3][4]     -> prediction for 2
            Sent 3: [1][2][3][4][5]     -> prediction for 3
            Sent 4: [2][3][4][5][6]     -> prediction for 4
        """
        if not sentences:
            return []

        n = len(sentences)
        results = []

        for i in range(n):
            win_start = max(0, i - self.window_radius)
            win_end = min(n, i + self.window_radius + 1)
            window_text = " ".join(sentences[win_start:win_end])

            inputs = self.bias_tokenizer(
                window_text, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            ).to(self.device)

            with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
                logits = self.bias_model(**inputs).logits

            probs = torch.softmax(logits.float(), dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx])
            label = LABELS[pred_idx]

            # explainability on the TARGET sentence only
            top_tokens = []
            if explain and self._embedding_layer is not None:
                is_pol, _ = self.check_politicalness(sentences[i])
                if is_pol and confidence > 0.4:
                    top_tokens = self.gradient_x_input(sentences[i])[:5]

            results.append({
                "text": sentences[i],
                "bias": label,
                "confidence": confidence,
                "probabilities": {l: float(probs[j]) for j, l in enumerate(LABELS)},
                "window": f"[{win_start}:{win_end-1}] -> sent {i}",
                "window_size": win_end - win_start,
                "top_tokens": top_tokens,
            })

        return results

    # Gradient x Input explainability

    def gradient_x_input(self, text, target_class=None):
        if self._embedding_layer is None:
            return []

        was_fp16 = next(self.bias_model.parameters()).dtype == torch.float16
        if was_fp16:
            self.bias_model.float()

        inputs = self.bias_tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tokens = self.bias_tokenizer.convert_ids_to_tokens(input_ids[0])

        embeddings = self._embedding_layer(input_ids)
        embeddings = embeddings.detach().requires_grad_(True)

        outputs = self.bias_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        logits[0, target_class].backward()

        attr = (embeddings.grad * embeddings).sum(dim=-1).abs().squeeze(0)
        if attr.max() > 0:
            attr = attr / attr.max()
        scores = attr.detach().cpu().numpy()

        if was_fp16:
            self.bias_model.half()

        return self._build_token_list(tokens, scores)

    @staticmethod
    def _build_token_list(tokens, scores):
        result = []
        for i, (token, score) in enumerate(zip(tokens, scores)):
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
                continue
            clean = token.replace("##", "").replace("\u2581", "").strip()
            if not clean or clean in PUNCTUATION:
                continue
            if all(c in '.,!?;:\'"()[]<>-\u2013\u2014/' for c in clean):
                continue
            if clean.lower() in STOP_WORDS:
                continue
            result.append({
                "token": clean,
                "score": round(float(score), 3),
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    # weighted aggregation

    def aggregate_bias(self, sentence_results):
        if not sentence_results:
            return {
                "predicted_label": "Center",
                "all_probs": {"Left": 0.0, "Center": 1.0, "Right": 0.0},
            }

        weighted_probs = []
        total_weight = 0.0

        for s in sentence_results:
            weight = 1.0
            if s["confidence"] < 0.45:
                weight *= 0.3
            text = s["text"].strip()
            if text.startswith('"') or text.startswith("\u201c"):
                weight *= 0.5

            probs_tensor = torch.tensor([
                s["probabilities"]["Left"],
                s["probabilities"]["Center"],
                s["probabilities"]["Right"],
            ])
            weighted_probs.append(probs_tensor * weight)
            total_weight += weight

        if total_weight == 0:
            return {
                "predicted_label": "Center",
                "all_probs": {"Left": 0.0, "Center": 1.0, "Right": 0.0},
            }

        avg_probs = torch.stack(weighted_probs).sum(dim=0) / total_weight
        pred_idx = torch.argmax(avg_probs).item()

        return {
            "predicted_label": LABELS[pred_idx],
            "all_probs": {l: float(avg_probs[i]) for i, l in enumerate(LABELS)},
        }

    def analyze(self, text, explain=True):
        start = time.time()

        is_political, pol_conf = self.check_politicalness(text)

        if not is_political:
            return {
                "is_political": False,
                "political_confidence": pol_conf,
                "prediction": None,
                "confidence": None,
                "probs": None,
                "sentences": [],
                "elapsed": time.time() - start,
            }

        sentences = self._split_sentences(text)

        if not sentences:
            return {
                "is_political": True,
                "political_confidence": pol_conf,
                "prediction": "Center",
                "confidence": 0.0,
                "probs": {"Left": 0.0, "Center": 1.0, "Right": 0.0},
                "sentences": [],
                "elapsed": time.time() - start,
            }

        sentence_results = self.analyze_sentences(sentences, explain=explain)
        overall = self.aggregate_bias(sentence_results)

        return {
            "is_political": True,
            "political_confidence": pol_conf,
            "prediction": overall["predicted_label"],
            "probs": overall["all_probs"],
            "sentences": sentence_results,
            "elapsed": time.time() - start,
        }

    # sentence splitter 
    @staticmethod
    def _split_sentences(text):
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        expanded = []
        for chunk in raw:
            parts = chunk.split("\n\n")
            expanded.extend(p.strip() for p in parts if p.strip())

        merged = []
        for s in expanded:
            if merged and len(s.split()) < 8:
                merged[-1] += " " + s
            else:
                merged.append(s)

        return [s for s in merged if len(s.split()) >= 5]

# FastAPI

app = FastAPI(title="Veritas Bias Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
detector = BiasDetector(
    bias_path="../models/bias_detector",
    pol_path="../models/politicalness_filter",
)
print("Server ready!")


# Request/Response Models
class AnalysisRequest(BaseModel):
    text: str

class TokenAttribution(BaseModel):
    token: str
    score: float

class SentenceBias(BaseModel):
    text: str
    category: str
    description: str
    bias: str
    confidence: float
    probabilities: Dict[str, float]
    top_tokens: List[TokenAttribution]

class AnalysisResponse(BaseModel):
    overall_bias: str
    overall_probabilities: Dict[str, float]
    sentences: List[SentenceBias]
    checks: int
    issues: int


def categorize_bias(bias: str, confidence: float) -> tuple:
    if confidence < 0.5:
        return "Neutral/Balanced", "No strong political bias detected in this statement."
    elif bias == "Left":
        return "Left-leaning", "This statement shows left-wing political perspective or framing."
    elif bias == "Right":
        return "Right-leaning", "This statement shows right-wing political perspective or framing."
    else:
        return "Centrist", "This statement presents a balanced or centrist perspective."


@app.post("/bias/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        result = detector.analyze(text, explain=True)

        if not result["is_political"]:
            return AnalysisResponse(
                overall_bias="Center",
                overall_probabilities={"Left": 0.0, "Center": 1.0, "Right": 0.0},
                sentences=[],
                checks=0,
                issues=0,
            )

        sentence_results = []
        for s in result["sentences"]:
            category, description = categorize_bias(s["bias"], s["confidence"])
            sentence_results.append(SentenceBias(
                text=s["text"],
                category=category,
                description=description,
                bias=s["bias"],
                confidence=s["confidence"],
                probabilities=s["probabilities"],
                top_tokens=[
                    TokenAttribution(token=t["token"], score=t["score"])
                    for t in s.get("top_tokens", [])
                ],
            ))

        checks = len(sentence_results)
        issues = sum(
            1 for s in sentence_results
            if s.confidence > 0.6 and s.bias != "Center"
        )

        return AnalysisResponse(
            overall_bias=result["prediction"],
            overall_probabilities=result["probs"],
            sentences=sentence_results,
            checks=checks,
            issues=issues,
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text_legacy(request: AnalysisRequest):
    return await analyze_text(request)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "device": str(detector.device),
        "window_radius": detector.window_radius,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)