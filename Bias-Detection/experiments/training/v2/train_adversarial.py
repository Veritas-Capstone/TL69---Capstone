"""
Adversarial media adaptation for political bias detection
Gradient reversal forces encoder to learn source-invariant representations
"""

import json, os, random, warnings
from collections import Counter
from pathlib import Path
from torch.autograd import Function

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", category=FutureWarning)

PARQUET_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets"
ALLSIDES_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets/other_data"
BASE_OUTPUT = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/adversarial"
BASE_MODEL = "microsoft/deberta-v3-large"

LABELS = ["left", "center", "right"]
LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {0: "left", 1: "center", 2: "right"}
ALLSIDES_MAP = {"Left": "left", "Lean Left": "left", "Center": "center",
                "Lean Right": "right", "Right": "right"}

DATASETS = [
    "article_bias_prediction", "dem_rep_party_platform_topics",
    "gpt4_political_bias", "gpt4_political_ideologies",
    "political_tweets", "qbias", "webis_bias_flipper_18", "webis_news_bias_20",
]
DATASET2ID = {name: i for i, name in enumerate(DATASETS)}
NUM_DOMAINS = len(DATASETS)

CENTER_MULTIPLIERS = {
    "article_bias_prediction": 5.0, "gpt4_political_bias": 3.25,
    "qbias": 3.25, "webis_bias_flipper_18": 2.6, "webis_news_bias_20": 2.6,
}

# config settings
EPOCHS = 4
BATCH_SIZE = 10
GRAD_ACCUM = 4
LR = 3e-5
LAMBDA_ADV = 0.7
CENTER_WEIGHT = 1.0 
MAX_LENGTH = 256
SEED = 42


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# model
class AdversarialBiasModel(nn.Module):
    def __init__(self, model_name, num_labels=3, num_domains=NUM_DOMAINS, lambda_adv=0.7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.bias_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, num_labels))
        self.gradient_reversal = GradientReversalLayer(alpha=lambda_adv)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_domains))
        self.num_labels = num_labels
        self.lambda_adv = lambda_adv

    def forward(self, input_ids, attention_mask, bias_labels=None, domain_labels=None,
                class_weights=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        bias_logits = self.bias_classifier(pooled)
        reversed_pooled = self.gradient_reversal(pooled)
        domain_logits = self.domain_classifier(reversed_pooled)

        loss = None
        if bias_labels is not None:
            if class_weights is not None:
                bias_loss = nn.CrossEntropyLoss(weight=class_weights)(bias_logits, bias_labels)
            else:
                bias_loss = nn.CrossEntropyLoss()(bias_logits, bias_labels)
            loss = bias_loss
            if domain_labels is not None:
                domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_labels)
                loss = loss + self.lambda_adv * domain_loss

        return {"loss": loss, "bias_logits": bias_logits, "domain_logits": domain_logits}

# dataset
class AdversarialDataset(Dataset):
    def __init__(self, texts, bias_labels, domain_labels, tokenizer, max_length=256):
        self.texts, self.bias_labels, self.domain_labels = texts, bias_labels, domain_labels
        self.tokenizer, self.max_length = tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True,
                            max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "bias_labels": torch.tensor(self.bias_labels[idx], dtype=torch.long),
                "domain_labels": torch.tensor(self.domain_labels[idx], dtype=torch.long)}

# data loading
def load_datasets(sample_size=10000, test_frac=0.15, seed=42):
    np.random.seed(seed); random.seed(seed)
    train_texts, train_bias, train_domain = [], [], []
    val_texts, val_bias, val_domain = [], [], []

    for name in DATASETS:
        path = os.path.join(PARQUET_DIR, f"{name}.parquet")
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        if "title" in df.columns:
            df["text"] = df.apply(lambda r: (str(r["title"]) + "\n\n" + str(r["body"]))
                if pd.notna(r.get("title")) else str(r["body"]), axis=1)
        else:
            df["text"] = df["body"].astype(str)
        df["label"] = df["leaning"].map(LABEL2ID)
        df = df.dropna(subset=["label"]); df["label"] = df["label"].astype(int)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        test_n = int(len(df) * test_frac)
        remaining = df.iloc[test_n:]
        val_n = min(100, len(remaining))
        val_df = remaining.iloc[:val_n]
        pool = remaining.iloc[val_n:]

        classes = pool["label"].unique()
        per_class = sample_size // len(classes)
        sampled = []
        for cls in classes:
            cls_df = pool[pool["label"] == cls]
            n = per_class
            if cls == LABEL2ID["center"] and name in CENTER_MULTIPLIERS:
                n = int(per_class * CENTER_MULTIPLIERS[name])
            sampled.append(cls_df.sample(n=min(n, len(cls_df)), random_state=seed))
        train_df = pd.concat(sampled)

        domain_id = DATASET2ID[name]
        train_texts.extend(train_df["text"].tolist()); train_bias.extend(train_df["label"].tolist())
        train_domain.extend([domain_id] * len(train_df))
        val_texts.extend(val_df["text"].tolist()); val_bias.extend(val_df["label"].tolist())
        val_domain.extend([domain_id] * len(val_df))
        print(f"  {name}: train={len(train_df)}, val={len(val_df)}")

    combined = list(zip(train_texts, train_bias, train_domain))
    random.shuffle(combined)
    train_texts, train_bias, train_domain = zip(*combined)
    print(f"\n  Total: train={len(train_texts)}, val={len(val_texts)}")
    print(f"  Train dist: {Counter(train_bias)}")
    return (list(train_texts), list(train_bias), list(train_domain),
            list(val_texts), list(val_bias), list(val_domain))


def load_allsides():
    articles, seen = [], set()
    for fp in sorted(Path(ALLSIDES_DIR).glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for story in data:
            for side in story.get("sides", []):
                bd = side.get("bias_detail")
                body = (side.get("body") or "").strip()
                url = side.get("original_url", "")
                if not body or not bd or bd not in ALLSIDES_MAP: continue
                if url and url in seen: continue
                if url: seen.add(url)
                hl = side.get("headline", "")
                articles.append({"text": (hl + ". " + body) if hl else body,
                    "source": side.get("source", "Unknown"), "bias_detail": bd,
                    "true_label": ALLSIDES_MAP[bd]})
    return articles

# training 
def train(epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM, lr=LR,
          lambda_adv=LAMBDA_ADV, center_weight=CENTER_WEIGHT, max_length=MAX_LENGTH, seed=SEED):
    
    # config-specific output directory
    config_name = f"lambda{lambda_adv}"

    if center_weight != 1.0: config_name += f"_cw{center_weight}"
    if max_length != 256: config_name += f"_ml{max_length}"

    output_dir = os.path.join(BASE_OUTPUT, config_name, "model")
    results_dir = os.path.join(BASE_OUTPUT, config_name, "results")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Config: {config_name}, lambda={lambda_adv}, cw={center_weight}, "
          f"ml={max_length}, bs={batch_size*grad_accum}, device={device}")

    (train_t, train_b, train_d, val_t, val_b, val_d) = load_datasets(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AdversarialBiasModel(BASE_MODEL, lambda_adv=lambda_adv).to(device)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_ds = AdversarialDataset(train_t, train_b, train_d, tokenizer, max_length)
    val_ds = AdversarialDataset(val_t, val_b, val_d, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    # class weights for bias loss
    class_weights = None
    if center_weight != 1.0:
        class_weights = torch.tensor([1.0, center_weight, 1.0], dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, class_weights=class_weights)
            loss = out["loss"] / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += out["loss"].item(); steps += 1
            if (i + 1) % log_interval == 0:
                print(f"    Epoch {epoch+1}, step {i+1}/{len(train_loader)}, loss={total_loss/steps:.4f}")

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_preds.extend(torch.argmax(out["bias_logits"], dim=-1).cpu().tolist())
                val_true.extend(batch["bias_labels"].cpu().tolist())

        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/steps:.4f}, val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump({"lambda_adv": lambda_adv, "max_length": max_length,
                           "center_weight": center_weight, "base_model": BASE_MODEL}, f)
            print(f"    Saved (F1={best_f1:.3f})")

    print(f"\n  Best val F1: {best_f1:.3f}")

    # evaluate on AllSides
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))
    model.eval()
    articles = load_allsides()
    print(f"\n  Evaluating on {len(articles)} AllSides articles...")

    preds_list, truths_list, all_probs = [], [], []
    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=max_length,
                          padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(out["bias_logits"], dim=-1)[0]
        preds_list.append(ID2LABEL[torch.argmax(probs).item()])
        truths_list.append(article["true_label"])
        all_probs.append(probs.cpu().numpy())

    acc = accuracy_score(truths_list, preds_list)
    f1 = f1_score(truths_list, preds_list, average="macro", zero_division=0, labels=LABELS)

    print(f"\n  AllSides OOD [{config_name}]")
    print(f"  Accuracy: {acc:.1%}  Macro F1: {f1:.3f}")
    print(f"\n{classification_report(truths_list, preds_list, labels=LABELS, zero_division=0)}")

    cm = confusion_matrix(truths_list, preds_list, labels=LABELS)
    print(f"  Confusion Matrix:")
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    # confidence abstention
    all_probs = np.array(all_probs)
    print(f"\n  Confidence abstention:")
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = all_probs.max(axis=1) >= threshold
        n = mask.sum()
        if n == 0: continue
        cp = [p for p, m in zip(preds_list, mask) if m]
        ct = [t for t, m in zip(truths_list, mask) if m]
        print(f"    t={threshold:.2f}: {n}/{len(articles)} ({n/len(articles):.0%}), "
              f"acc={accuracy_score(ct, cp):.1%}, F1={f1_score(ct, cp, average='macro', zero_division=0, labels=LABELS):.3f}")

    # breakdown by detail label
    detail_groups = {}
    for article, pred in zip(articles, preds_list):
        d = article["bias_detail"]
        if d not in detail_groups: detail_groups[d] = {"preds": [], "truths": []}
        detail_groups[d]["preds"].append(pred)
        detail_groups[d]["truths"].append(article["true_label"])

    print(f"\n  Breakdown:")
    for detail in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
        if detail in detail_groups:
            g = detail_groups[detail]
            da = accuracy_score(g["truths"], g["preds"])
            nc = sum(p == t for p, t in zip(g["preds"], g["truths"]))
            print(f"    {detail:<12}: {da:.1%} ({nc}/{len(g['preds'])})")

    results = {"config": config_name, "lambda_adv": lambda_adv, "center_weight": center_weight,
               "max_length": max_length, "accuracy": float(acc), "macro_f1": float(f1),
               "best_val_f1": float(best_f1)}
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()