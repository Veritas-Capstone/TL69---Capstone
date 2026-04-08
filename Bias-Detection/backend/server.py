"""
Veritas bias detection server
Auto-detects GPU/CPU. Endpoints: /bias/analyze, /bias/explain
Run with: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np
import os, json, time, warnings
import spacy
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# model paths (override with env vars on server)
AA_MODEL_PATH = os.getenv("BIAS_AA_MODEL_PATH", "/u50/shared/bias_detection/models/leace")
POLITICALNESS_MODEL_PATH = os.getenv("BIAS_POLITICALNESS_MODEL_PATH", "/u50/shared/bias_detection/models/politicalness_filter/")

# inference config
LABELS = ["Left", "Center", "Right"]
BIAS_CONFIDENCE_THRESHOLD = 0.55
OVERALL_HIGH_CONFIDENCE = 0.55
OVERALL_LOW_CONFIDENCE = 0.45
POLITICALNESS_THRESHOLD = 0.4
WINDOW_RADIUS = 2

POLITICALNESS_HYPOTHESES = [
    "This text is about politics or government policy.",
    "This text discusses a politically controversial or partisan topic.",
    "This text is about legislation, regulation, or political debate.",
    "This text discusses government officials or political parties.",
    "This text discusses social policy like healthcare, immigration, or civil rights.",
    "This text discusses government regulation, public health policy, or government overreach.",
    "This text discusses military operations, weapons systems, or national security.",
    "This text discusses economic policy, taxes, trade, or government spending.",
    "This text discusses elections, voting, or political campaigns.",
    "This text discusses foreign policy or international relations.",
]

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
    "-", "\u2013", "\u2014", "\u2581.", "\u2581,", "\u2581!", "\u2581?",
    "\u2581;", "\u2581:", "\u2581-",
})


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LeaceBiasModel(nn.Module):
    def __init__(self, model_name, leace_projection, num_labels=3,
                    num_domains=8, lambda_adv=0.7, hidden_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.register_buffer("leace_proj",
                                torch.tensor(leace_projection, dtype=torch.float32))
        self.classifier = MLPClassifier(hidden_size, num_labels, hidden_dim)
        self.bias_classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hidden_size, num_labels))
        self.gradient_reversal = nn.Identity()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_domains))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled @ self.leace_proj.T)

    def forward_with_embeds(self, inputs_embeds, attention_mask):
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled @ self.leace_proj.T)


class BiasDetector:
    def __init__(self, aa_path, pol_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(8)
            torch.set_num_interop_threads(4)

        self.use_fp16 = self.device.type == "cuda"
        self.max_batch_size = 8 if self.device.type == "cuda" else 4
        print(f"  Device: {self.device},  FP16: {self.use_fp16},  Batch: {self.max_batch_size}")

        self.nlp = spacy.load("en_core_web_sm")
        print("  spaCy en_core_web_sm loaded")

        self._load_bias_model(aa_path)
        self._load_pol_model(pol_path)
        print("  Ready\n")

    def _load_bias_model(self, aa_path):
        config_path = os.path.join(aa_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        base_model = config.get("base_model", "microsoft/deberta-v3-large")
        lambda_adv = config.get("lambda_adv", 0.7)

        leace_proj = np.load(os.path.join(aa_path, "leace_projection.npy"))
        print(f"  LEACE projection: {leace_proj.shape}")

        self.bias_tokenizer = AutoTokenizer.from_pretrained(aa_path)
        self.bias_model = LeaceBiasModel(base_model, leace_proj, lambda_adv=lambda_adv)

        aa_state = torch.load(os.path.join(aa_path, "model.pt"), map_location=self.device)
        self.bias_model.load_state_dict(aa_state, strict=False)

        mlp_state = torch.load(os.path.join(aa_path, "classifier.pt"), map_location=self.device)
        self.bias_model.classifier.load_state_dict(mlp_state)

        self.bias_model.eval().to(self.device)
        if self.use_fp16:
            self.bias_model.half()
        self._embedding_layer = self.bias_model.encoder.embeddings.word_embeddings
        print("  Encoder + LEACE + MLP loaded")

    def _load_pol_model(self, pol_path):
        self.pol_model = AutoModelForSequenceClassification.from_pretrained(pol_path)
        self.pol_tokenizer = AutoTokenizer.from_pretrained(pol_path)
        self.pol_model.eval().to(self.device)
        if self.use_fp16:
            self.pol_model.half()
        label2id = self.pol_model.config.label2id
        self.entail_idx = label2id.get("ENTAILMENT", label2id.get("entailment", 0))
        print("  Politicalness filter loaded")

    def check_politicalness(self, text):
        pairs = [(text[:1024], h) for h in POLITICALNESS_HYPOTHESES]
        inputs = self.pol_tokenizer(
            pairs, return_tensors="pt", truncation=True,
            max_length=512, padding=True).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.pol_model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1)
        best_conf = float(probs[:, self.entail_idx].max())
        return best_conf > POLITICALNESS_THRESHOLD, best_conf

    def _predict_bias_batch(self, texts):
        inputs = self.bias_tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True).to(self.device)

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.use_fp16):
            logits = self.bias_model(inputs["input_ids"], inputs["attention_mask"])

        probs = torch.softmax(logits.float(), dim=-1)
        return [probs[i] for i in range(len(texts))]

    def analyze_sentences(self, sentences):
        if not sentences:
            return []

        n = len(sentences)
        window_texts, window_meta = [], []
        for i in range(n):
            ws = max(0, i - WINDOW_RADIUS)
            we = min(n, i + WINDOW_RADIUS + 1)
            window_texts.append(" ".join(sentences[ws:we]))
            window_meta.append((i, ws, we))

        all_probs = []
        for bs in range(0, len(window_texts), self.max_batch_size):
            be = min(bs + self.max_batch_size, len(window_texts))
            all_probs.extend(self._predict_bias_batch(window_texts[bs:be]))

        results = []
        for probs, (i, ws, we) in zip(all_probs, window_meta):
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx])
            entropy = -sum(float(p) * np.log(float(p) + 1e-10) for p in probs) / np.log(3)
            is_uncertain = confidence < BIAS_CONFIDENCE_THRESHOLD or entropy > 0.90
            label = "Uncertain" if is_uncertain else LABELS[pred_idx]

            is_quote = (sentences[i].strip().startswith('"') or
                        sentences[i].strip().startswith("\u201c") or
                        sentences[i].strip().startswith("\u2018"))

            results.append({
                "text": sentences[i], "bias": label, "confidence": confidence,
                "is_quote": is_quote,
                "probabilities": {l: float(probs[j]) for j, l in enumerate(LABELS)},
            })

        return results

    def explain_sentence(self, text):
        was_fp16 = next(self.bias_model.parameters()).dtype == torch.float16
        if was_fp16:
            self.bias_model.float()

        inputs = self.bias_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=256, padding=True).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tokens = self.bias_tokenizer.convert_ids_to_tokens(input_ids[0])

        embeddings = self._embedding_layer(input_ids)
        embeddings = embeddings.detach().requires_grad_(True)
        logits = self.bias_model.forward_with_embeds(embeddings, attention_mask)

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
        merged = []
        for token, score in zip(tokens, scores):
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
                continue
            clean = token.replace("##", "").replace("\u2581", "").strip()
            if not clean:
                continue
            if not token.startswith("\u2581") and merged and token not in ("<s>", "</s>"):
                merged[-1] = (merged[-1][0] + clean, max(merged[-1][1], float(score)))
            else:
                merged.append((clean, float(score)))

        result = []
        for clean, score in merged:
            if clean in PUNCTUATION:
                continue
            if all(c in '.,!?;:\'"()[]<>-\u2013\u2014/' for c in clean):
                continue
            if clean.lower() in STOP_WORDS:
                continue
            if len(clean) < 2:
                continue
            result.append({"token": clean, "score": round(score, 3)})

        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    def aggregate_bias(self, sentence_results):
        if not sentence_results:
            return {"predicted_label": "Uncertain",
                    "all_probs": {"Left": 0.0, "Center": 0.0, "Right": 0.0},
                    "confidence_level": "none", "overall_confidence": 0.0}

        weighted_probs = []
        total_weight = 0.0

        for s in sentence_results:
            if s["bias"] == "Uncertain":
                continue

            weight = 1.0
            if s["confidence"] < 0.50:
                weight *= 0.3
            elif s["confidence"] < 0.60:
                weight *= 0.6
            if s.get("is_quote", False):
                weight *= 0.3

            probs_tensor = torch.tensor([
                s["probabilities"]["Left"],
                s["probabilities"]["Center"],
                s["probabilities"]["Right"],
            ])
            weighted_probs.append(probs_tensor * weight)
            total_weight += weight

        if total_weight == 0:
            return {"predicted_label": "Uncertain",
                    "all_probs": {"Left": 0.33, "Center": 0.34, "Right": 0.33},
                    "confidence_level": "insufficient", "overall_confidence": 0.0}

        avg_probs = torch.stack(weighted_probs).sum(dim=0) / total_weight
        pred_idx = torch.argmax(avg_probs).item()
        overall_conf = float(avg_probs[pred_idx])
        agreement = sum(1 for s in sentence_results
                        if s["bias"] == LABELS[pred_idx]) / len(sentence_results)

        if overall_conf >= OVERALL_HIGH_CONFIDENCE and agreement >= 0.4:
            confidence_level = "high"
        elif overall_conf >= OVERALL_LOW_CONFIDENCE:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        return {"predicted_label": LABELS[pred_idx],
                "all_probs": {l: float(avg_probs[i]) for i, l in enumerate(LABELS)},
                "confidence_level": confidence_level,
                "overall_confidence": overall_conf}

    def analyze(self, text):
        start = time.time()
        is_political, pol_conf = self.check_politicalness(text)

        if not is_political:
            return {"is_political": False, "political_confidence": pol_conf,
                    "prediction": "Not Political", "probs": None,
                    "overall_confidence": 0.0, "confidence_level": "none",
                    "sentences": [], "elapsed": time.time() - start}

        sentences = self._split_sentences(text)
        if not sentences:
            return {"is_political": True, "political_confidence": pol_conf,
                    "prediction": "Uncertain",
                    "probs": {"Left": 0.0, "Center": 0.0, "Right": 0.0},
                    "overall_confidence": 0.0, "confidence_level": "insufficient",
                    "sentences": [], "elapsed": time.time() - start}

        sentence_results = self.analyze_sentences(sentences)
        overall = self.aggregate_bias(sentence_results)

        return {"is_political": True, "political_confidence": pol_conf,
                "prediction": overall["predicted_label"],
                "probs": overall["all_probs"],
                "overall_confidence": overall["overall_confidence"],
                "confidence_level": overall["confidence_level"],
                "sentences": sentence_results,
                "elapsed": time.time() - start}

    def _split_sentences(self, text):
        doc = self.nlp(text)
        raw = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        merged = []
        for s in raw:
            if merged and len(s.split()) < 8:
                merged[-1] += " " + s
            else:
                merged.append(s)

        return [s for s in merged if len(s.split()) >= 5]


# fastapi

app = FastAPI(title="Veritas Bias Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

print("Loading models...")
detector = BiasDetector(aa_path=AA_MODEL_PATH, pol_path=POLITICALNESS_MODEL_PATH)


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
    overall_confidence: float
    confidence_level: str
    sentences: List[SentenceBias]
    checks: int
    issues: int

class ExplainRequest(BaseModel):
    text: str

class TokenAttribution(BaseModel):
    token: str
    score: float

class ExplainResponse(BaseModel):
    text: str
    top_tokens: List[TokenAttribution]


def categorize_bias(bias, confidence):
    if bias == "Uncertain" or confidence < BIAS_CONFIDENCE_THRESHOLD:
        return "Uncertain", "The model does not have enough confidence to determine bias for this statement."
    if bias == "Left":
        if confidence >= 0.70:
            return "Left-leaning", "This statement shows clear left-wing political perspective or framing."
        return "Possibly Left-leaning", "This statement may show left-wing framing, but confidence is moderate."
    if bias == "Right":
        if confidence >= 0.70:
            return "Right-leaning", "This statement shows clear right-wing political perspective or framing."
        return "Possibly Right-leaning", "This statement may show right-wing framing, but confidence is moderate."
    return "Centrist", "This statement presents a balanced or centrist perspective."


@app.post("/bias/analyze", response_model=AnalysisResponse)
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        result = detector.analyze(text)

        if not result["is_political"]:
            return AnalysisResponse(
                overall_bias="Not Political",
                overall_probabilities={"Left": 0.0, "Center": 0.0, "Right": 0.0},
                overall_confidence=0.0, confidence_level="none",
                sentences=[], checks=0, issues=0)

        sentence_results = []
        for s in result["sentences"]:
            category, description = categorize_bias(s["bias"], s["confidence"])
            sentence_results.append(SentenceBias(
                text=s["text"], category=category, description=description,
                bias=s["bias"], confidence=s["confidence"],
                probabilities=s["probabilities"]))

        checks = len(sentence_results)
        issues = sum(1 for s in sentence_results
                        if s.confidence >= BIAS_CONFIDENCE_THRESHOLD
                        and s.bias not in ("Center", "Uncertain"))

        return AnalysisResponse(
            overall_bias=result["prediction"] or "Uncertain",
            overall_probabilities=result["probs"] or {"Left": 0.0, "Center": 0.0, "Right": 0.0},
            overall_confidence=result.get("overall_confidence", 0.0),
            confidence_level=result.get("confidence_level", "low"),
            sentences=sentence_results, checks=checks, issues=issues)

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bias/explain", response_model=ExplainResponse)
@app.post("/explain", response_model=ExplainResponse)
async def explain_sentence(request: ExplainRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        top_tokens = detector.explain_sentence(text)[:5]
        return ExplainResponse(
            text=text,
            top_tokens=[TokenAttribution(token=t["token"], score=t["score"])
                        for t in top_tokens])

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(detector.device),
            "pipeline": "AA + LEACE", "fp16": detector.use_fp16}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
