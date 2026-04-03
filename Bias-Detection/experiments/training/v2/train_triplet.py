"""
Triplet loss pre-training for political bias detection

Implements Baly et al. (2020) triplet loss approach:
  Phase 1: Pre-train encoder with triplet loss
    - Anchor + Positive: same ideology, DIFFERENT dataset
    - Anchor + Negative: different ideology, SAME dataset encoder clusters by ideology, not by source
  Phase 2: Fine-tune on classification with pre-trained encoder
"""

import json, os, random, warnings
from collections import Counter, defaultdict
from pathlib import Path

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
OUTPUT_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/triplet/model"
RESULTS_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/triplet/results"
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
CENTER_MULTIPLIERS = {
    "article_bias_prediction": 5.0, "gpt4_political_bias": 3.25,
    "qbias": 3.25, "webis_bias_flipper_18": 2.6, "webis_news_bias_20": 2.6,
}
MAX_LENGTH = 256

# config settings
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 4
BATCH_SIZE = 10
TRIPLET_BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 3e-5
TRIPLET_LR = 2e-5
N_TRIPLETS = 35000
SEED = 42

# model
class BiasModelWithTriplet(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, num_labels))

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.classifier(self.encode(input_ids, attention_mask))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# triplet dataset
class TripletDataset(Dataset):
    def __init__(self, articles_by_dataset_and_label, tokenizer, max_length=256, n_triplets=35000, seed=42):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.data = articles_by_dataset_and_label
        self.triplets = self._generate_triplets(n_triplets, seed)
        print(f"  Generated {len(self.triplets)} triplets")

    def _generate_triplets(self, n, seed):
        rng = random.Random(seed)
        triplets = []
        datasets = list(self.data.keys())
        attempts = 0
        while len(triplets) < n and attempts < n * 10:
            attempts += 1
            anchor_ds = rng.choice(datasets)
            anchor_label = rng.choice(list(self.data[anchor_ds].keys()))
            if not self.data[anchor_ds][anchor_label]: continue
            anchor_text = rng.choice(self.data[anchor_ds][anchor_label])

            # positive: same label, different dataset
            other_ds = [d for d in datasets if d != anchor_ds and anchor_label in self.data[d] and self.data[d][anchor_label]]
            if not other_ds: continue
            pos_text = rng.choice(self.data[rng.choice(other_ds)][anchor_label])

            # negative: different label, same dataset
            other_labels = [l for l in self.data[anchor_ds].keys() if l != anchor_label and self.data[anchor_ds][l]]
            if not other_labels: continue
            neg_text = rng.choice(self.data[anchor_ds][rng.choice(other_labels)])
            triplets.append((anchor_text, pos_text, neg_text))
        return triplets

    def __len__(self): return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]
        def enc(text):
            e = self.tokenizer(str(text), truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
            return e["input_ids"].squeeze(0), e["attention_mask"].squeeze(0)
        a_ids, a_mask = enc(anchor); p_ids, p_mask = enc(pos); n_ids, n_mask = enc(neg)
        return {"anchor_ids": a_ids, "anchor_mask": a_mask,
                "pos_ids": p_ids, "pos_mask": p_mask, "neg_ids": n_ids, "neg_mask": n_mask}


class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    
    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_length,
                            padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

# data loading
def load_all_data(sample_size=10000, seed=42):
    np.random.seed(seed); random.seed(seed)
    by_ds_label = defaultdict(lambda: defaultdict(list))
    train_texts, train_labels, val_texts, val_labels = [], [], [], []

    for name in DATASETS:
        path = os.path.join(PARQUET_DIR, f"{name}.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        if "title" in df.columns:
            df["text"] = df.apply(
                lambda r: (str(r["title"]) + "\n\n" + str(r["body"]))
                if pd.notna(r.get("title")) else str(r["body"]), axis=1)
        else:
            df["text"] = df["body"].astype(str)

        df["label"] = df["leaning"].map(LABEL2ID)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Hold out 15%, take val, then train
        test_n = int(len(df) * 0.15)
        remaining = df.iloc[test_n:]
        val_n = min(100, len(remaining))
        val_df = remaining.iloc[:val_n]
        pool = remaining.iloc[val_n:]

        # Train sample
        classes = pool["label"].unique()
        per_class = sample_size // len(classes)
        sampled = []
        for cls in classes:
            cls_df = pool[pool["label"] == cls]
            n = per_class
            if cls == LABEL2ID["center"] and name in CENTER_MULTIPLIERS:
                n = int(per_class * CENTER_MULTIPLIERS[name])
            n = min(n, len(cls_df))
            sampled.append(cls_df.sample(n=n, random_state=seed))
        train_df = pd.concat(sampled)

        # Populate triplet structure
        for _, row in train_df.iterrows():
            by_ds_label[name][row["label"]].append(row["text"])

        train_texts.extend(train_df["text"].tolist())
        train_labels.extend(train_df["label"].tolist())
        val_texts.extend(val_df["text"].tolist())
        val_labels.extend(val_df["label"].tolist())

        print(f"  {name}: {len(train_df)} train, {len(val_df)} val")

    # Shuffle
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    print(f"  Total: {len(train_texts)} train, {len(val_texts)} val")
    return dict(by_ds_label), list(train_texts), list(train_labels), list(val_texts), list(val_labels)


def load_allsides():
    articles, seen = [], set()
    for fp in sorted(Path(ALLSIDES_DIR).glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f: data = json.load(f)
        for story in data:
            for side in story.get("sides", []):
                bd = side.get("bias_detail"); body = (side.get("body") or "").strip()
                url = side.get("original_url", "")
                if not body or not bd or bd not in ALLSIDES_MAP: continue
                if url and url in seen: continue
                if url: seen.add(url)
                hl = side.get("headline", "")
                articles.append({"text": (hl + ". " + body) if hl else body,
                    "source": side.get("source", "Unknown"), "bias_detail": bd, "true_label": ALLSIDES_MAP[bd]})
    return articles

# triplet loss pre-training
def pretrain_triplet(model, triplet_ds, device, epochs=PRETRAIN_EPOCHS,
                     batch_size=TRIPLET_BATCH_SIZE, lr=TRIPLET_LR):
    print(f"\n  Phase 1: Triplet pre-training ({epochs} epochs)")
    loader = DataLoader(triplet_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=lr, weight_decay=0.01)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    for epoch in range(epochs):
        model.train(); total_loss, steps = 0, 0
        for batch in loader:
            a_enc = model.encode(batch["anchor_ids"].to(device), batch["anchor_mask"].to(device))
            p_enc = model.encode(batch["pos_ids"].to(device), batch["pos_mask"].to(device))
            n_enc = model.encode(batch["neg_ids"].to(device), batch["neg_mask"].to(device))
            loss = triplet_loss_fn(a_enc, p_enc, n_enc); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item(); steps += 1
        print(f"  Epoch {epoch+1}/{epochs}: triplet_loss={total_loss/steps:.4f}")
    return model

# fintune on classification
def finetune_classify(model, train_texts, train_labels, val_texts, val_labels,
                      tokenizer, device, epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
                      grad_accum=GRAD_ACCUM, lr=LR):
    print(f"\n  Phase 2: Classification fine-tuning ({epochs} epochs)")
    train_loader = DataLoader(ClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH),
                            batch_size=batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(epochs):
        model.train(); total_loss, steps = 0, 0; optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch); loss = out["loss"] / grad_accum; loss.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += out["loss"].item(); steps += 1
            if (i + 1) % log_interval == 0:
                print(f"    Epoch {epoch+1}, step {i+1}/{len(train_loader)}, loss={total_loss/steps:.4f}")

        model.eval(); val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_preds.extend(torch.argmax(model(**batch)["logits"], dim=-1).cpu().tolist())
                val_true.extend(batch["labels"].cpu().tolist())
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/steps:.4f}, "
              f"val_acc={accuracy_score(val_true, val_preds):.3f}, val_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"    Saved (F1={best_f1:.3f})")

    print(f"\n  Best val F1: {best_f1:.3f}")
    return best_f1


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Triplet pipeline, device={device}")

    by_ds_label, train_t, train_l, val_t, val_l = load_all_data(seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = BiasModelWithTriplet(BASE_MODEL).to(device)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # phase 1
    triplet_ds = TripletDataset(by_ds_label, tokenizer, MAX_LENGTH, n_triplets=N_TRIPLETS, seed=SEED)
    model = pretrain_triplet(model, triplet_ds, device)
    del triplet_ds; torch.cuda.empty_cache()

    # phase 2
    best_f1 = finetune_classify(model, train_t, train_l, val_t, val_l, tokenizer, device)

    # AllSides eval
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "model.pt"))); model.eval()
    articles = load_allsides()
    print(f"\n  Evaluating on {len(articles)} AllSides articles...")

    preds, truths = [], []
    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=MAX_LENGTH,
                          padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            pred = torch.argmax(model(inputs["input_ids"], inputs["attention_mask"])["logits"], dim=-1).item()
        preds.append(ID2LABEL[pred]); truths.append(article["true_label"])

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro", zero_division=0, labels=LABELS)
    print(f"\n  Triplet AllSides OOD: acc={acc:.1%}, F1={f1:.3f}")
    print(f"\n{classification_report(truths, preds, labels=LABELS, zero_division=0)}")
    cm = confusion_matrix(truths, preds, labels=LABELS)
    print(f"  Confusion Matrix:")
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(f1),
                   "best_val_f1": float(best_f1), "n_triplets": N_TRIPLETS}, f, indent=2)