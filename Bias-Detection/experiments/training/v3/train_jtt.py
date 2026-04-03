"""
Just Train Twice (JTT) — Liu et al., ICML 2021
Stage 1: Train AA for a few epochs (early-stopped)
Stage 2: Retrain from scratch, upweighting misclassified examples 3-5x
"""

import json, os, random, warnings
from collections import Counter
from pathlib import Path
from torch.autograd import Function

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", category=FutureWarning)

PARQUET_DIR  = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets"
ALLSIDES_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets/other_data"
BASE_OUTPUT  = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments"

LABELS   = ["left", "center", "right"]
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
BASE_MODEL = "microsoft/deberta-v3-large"
EPOCHS = 6 
STAGE1_EPOCHS = 1
UPWEIGHT = 3 
BATCH_SIZE = 10
GRAD_ACCUM = 4
LR = 3e-5
LAMBDA_ADV = 0.7
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


class AdversarialBiasModel(nn.Module):
    def __init__(self, model_name, num_labels=3, num_domains=NUM_DOMAINS, lambda_adv=0.7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.bias_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, num_labels))
        self.gradient_reversal = GradientReversalLayer(alpha=lambda_adv)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_domains))
        self.lambda_adv = lambda_adv

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.bias_classifier(pooled), self.domain_classifier(self.gradient_reversal(pooled))


class BiasDataset(Dataset):
    def __init__(self, texts, bias_labels, domain_labels, tokenizer, max_length=256,
                 sample_weights=None):
        self.texts, self.bias_labels, self.domain_labels = texts, bias_labels, domain_labels
        self.tokenizer, self.max_length = tokenizer, max_length
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True,
                             max_length=self.max_length, padding="max_length", return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bias_labels": torch.tensor(self.bias_labels[idx], dtype=torch.long),
            "domain_labels": torch.tensor(self.domain_labels[idx], dtype=torch.long),
        }
        if self.sample_weights is not None:
            item["weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return item


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
        val_df = remaining.iloc[:min(100, len(remaining))]
        pool = remaining.iloc[len(val_df):]

        per_class = sample_size // len(pool["label"].unique())
        sampled = []
        for cls in pool["label"].unique():
            cls_df = pool[pool["label"] == cls]
            n = int(per_class * CENTER_MULTIPLIERS[name]) if cls == LABEL2ID["center"] and name in CENTER_MULTIPLIERS else per_class
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
    return (list(train_texts), list(train_bias), list(train_domain),
            list(val_texts), list(val_bias), list(val_domain))


def load_allsides():
    articles, seen = [], set()
    for fp in sorted(Path(ALLSIDES_DIR).glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for story in data:
            for side in story.get("sides", []):
                bd = side.get("bias_detail"); body = (side.get("body") or "").strip()
                url = side.get("original_url", "")
                if not body or not bd or bd not in ALLSIDES_MAP: continue
                if url and url in seen: continue
                if url: seen.add(url)
                hl = side.get("headline", "")
                articles.append({"text": (hl + ". " + body) if hl else body,
                    "source": side.get("source", "Unknown"), "bias_detail": bd,
                    "true_label": ALLSIDES_MAP[bd]})
    return articles


def evaluate_ood(model, tokenizer, device, max_length, tag):
    articles = load_allsides()
    model.eval()
    preds_list, truths_list = [], []

    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            logits, _ = model(inputs["input_ids"], inputs["attention_mask"])
            preds_list.append(ID2LABEL[torch.argmax(logits, dim=-1).item()])
        truths_list.append(article["true_label"])

    acc = accuracy_score(truths_list, preds_list)
    f1 = f1_score(truths_list, preds_list, average="macro", zero_division=0, labels=LABELS)
    print(f"\n  [{tag}] OOD: acc={acc:.1%}, F1={f1:.3f}")
    print(classification_report(truths_list, preds_list, labels=LABELS, zero_division=0))

    cm = confusion_matrix(truths_list, preds_list, labels=LABELS)
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")
    return acc, f1


def run_loop(model, train_loader, val_loader, device, epochs, tag, save_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    ce_fn = nn.CrossEntropyLoss(reduction='none')
    domain_ce = nn.CrossEntropyLoss()
    best_f1 = 0
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0, 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            bias_logits, domain_logits = model(batch["input_ids"], batch["attention_mask"])

            per_sample = ce_fn(bias_logits, batch["bias_labels"])
            if "weight" in batch:
                per_sample = per_sample * batch["weight"]
            bias_loss = per_sample.mean()

            domain_loss = domain_ce(domain_logits, batch["domain_labels"])
            loss = bias_loss + LAMBDA_ADV * domain_loss

            (loss / GRAD_ACCUM).backward()
            if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss += loss.item(); steps += 1
            if (i + 1) % log_interval == 0:
                print(f"    [{tag}] Ep {epoch+1}, step {i+1}/{len(train_loader)}, loss={total_loss/steps:.4f}")

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _ = model(batch["input_ids"], batch["attention_mask"])
                vp.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                vt.extend(batch["bias_labels"].cpu().tolist())

        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        print(f"  [{tag}] Epoch {epoch+1}/{epochs}: val_f1={vf1:.3f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            print(f"    Saved (F1={best_f1:.3f})")

    return best_f1


def train():
    mtag = "bert" if "bert-base" in BASE_MODEL else "deberta"
    config_name = f"jtt_{mtag}_up{UPWEIGHT}_s1e{STAGE1_EPOCHS}_ep{EPOCHS}"

    output_dir = os.path.join(BASE_OUTPUT, "jtt_v2", config_name, "model")
    results_dir = os.path.join(BASE_OUTPUT, "jtt_v2", config_name, "results")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  JTT: {config_name}, upweight={UPWEIGHT}x, device={device}")

    (train_t, train_b, train_d, val_t, val_b, val_d) = load_datasets(seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # stage 1: short training to identify hard examples
    print(f"\n  Stage 1: {STAGE1_EPOCHS} epoch(s)")
    model_s1 = AdversarialBiasModel(BASE_MODEL, lambda_adv=LAMBDA_ADV).to(device)
    s1_dir = os.path.join(output_dir, "stage1"); os.makedirs(s1_dir, exist_ok=True)

    train_ds = BiasDataset(train_t, train_b, train_d, tokenizer, MAX_LENGTH)
    val_ds = BiasDataset(val_t, val_b, val_d, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

    run_loop(model_s1, train_loader, val_loader, device, STAGE1_EPOCHS, "S1", s1_dir)

    # identify error set
    model_s1.load_state_dict(torch.load(os.path.join(s1_dir, "model.pt"), map_location=device))
    model_s1.eval()

    error_indices = []
    eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    offset = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            logits, _ = model_s1(batch_dev["input_ids"], batch_dev["attention_mask"])
            wrong = (torch.argmax(logits, dim=-1).cpu() != batch["bias_labels"]).nonzero(as_tuple=True)[0]
            for w in wrong:
                error_indices.append(offset + w.item())
            offset += len(batch["bias_labels"])

    error_set = set(error_indices)
    n_error = len(error_set)
    n_normal = len(train_t) - n_error
    error_share = (n_error * UPWEIGHT) / (n_error * UPWEIGHT + n_normal)

    print(f"\n  Error set: {n_error}/{len(train_t)} ({n_error/len(train_t):.1%})")
    print(f"  Effective gradient share: {error_share:.1%} (vs {n_error/len(train_t):.1%} of data)")

    del model_s1; torch.cuda.empty_cache()

    # stage 2: retrain with upweighted error set
    print(f"\n  Stage 2: {EPOCHS} epochs with {UPWEIGHT}x upweight")
    sample_weights = [UPWEIGHT if i in error_set else 1.0 for i in range(len(train_t))]
    train_ds_s2 = BiasDataset(train_t, train_b, train_d, tokenizer, MAX_LENGTH, sample_weights)
    train_loader_s2 = DataLoader(train_ds_s2, batch_size=BATCH_SIZE, shuffle=True)

    model_s2 = AdversarialBiasModel(BASE_MODEL, lambda_adv=LAMBDA_ADV).to(device)
    best_f1 = run_loop(model_s2, train_loader_s2, val_loader, device, EPOCHS, "S2", output_dir)
    tokenizer.save_pretrained(output_dir)

    # OOD eval
    model_s2.load_state_dict(torch.load(os.path.join(output_dir, "model.pt"), map_location=device))
    acc, f1_ood = evaluate_ood(model_s2, tokenizer, device, MAX_LENGTH, config_name)

    results = {"config": config_name, "upweight": UPWEIGHT, "stage1_epochs": STAGE1_EPOCHS,
               "error_set_size": n_error, "error_gradient_share": float(error_share),
               "ood_accuracy": float(acc), "ood_macro_f1": float(f1_ood),
               "best_val_f1": float(best_f1)}
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()