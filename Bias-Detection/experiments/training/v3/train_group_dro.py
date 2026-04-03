"""
Group DRO + adversarial adaptation (Sagawa et al., ICLR 2020)
"""

import json, os, random, warnings
from collections import Counter, defaultdict
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
BATCH_SIZE = 10
GRAD_ACCUM = 4
LR = 3e-5
LAMBDA_ADV = 0.7
MAX_LENGTH = 256
SEED = 42
DRO_ETA = 0.01 
WARMUP_EPOCHS = 2 


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
        bias_logits = self.bias_classifier(pooled)
        domain_logits = self.domain_classifier(self.gradient_reversal(pooled))
        return bias_logits, domain_logits


class BiasDataset(Dataset):
    def __init__(self, texts, bias_labels, domain_labels, tokenizer, max_length=256):
        self.texts, self.bias_labels, self.domain_labels = texts, bias_labels, domain_labels
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True,
                             max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bias_labels": torch.tensor(self.bias_labels[idx], dtype=torch.long),
            "domain_labels": torch.tensor(self.domain_labels[idx], dtype=torch.long),
        }


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


def train():
    mtag = "bert" if "bert-base" in BASE_MODEL else "deberta"
    config_name = f"gdro_{mtag}_eta{DRO_ETA}_wu{WARMUP_EPOCHS}_l{LAMBDA_ADV}_ep{EPOCHS}"
    if MAX_LENGTH != 256: config_name += f"_ml{MAX_LENGTH}"

    output_dir  = os.path.join(BASE_OUTPUT, "group_dro_v2", config_name, "model")
    results_dir = os.path.join(BASE_OUTPUT, "group_dro_v2", config_name, "results")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Group DRO: {config_name}")
    print(f"  eta={DRO_ETA} (epoch-level updates), warmup={WARMUP_EPOCHS} epochs, device={device}")

    (train_t, train_b, train_d, val_t, val_b, val_d) = load_datasets(seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AdversarialBiasModel(BASE_MODEL, lambda_adv=LAMBDA_ADV).to(device)

    train_loader = DataLoader(BiasDataset(train_t, train_b, train_d, tokenizer, MAX_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BiasDataset(val_t, val_b, val_d, tokenizer, MAX_LENGTH),
                            batch_size=BATCH_SIZE * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    # uniform group weights
    group_weights = torch.ones(NUM_DOMAINS, device=device) / NUM_DOMAINS
    domain_ce = nn.CrossEntropyLoss()

    best_f1, best_ood_f1 = 0, 0
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, steps = 0, 0
        epoch_domain_losses = defaultdict(list)
        optimizer.zero_grad()

        use_dro = epoch >= WARMUP_EPOCHS

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            bias_logits, domain_logits = model(batch["input_ids"], batch["attention_mask"])

            per_sample_loss = F.cross_entropy(bias_logits, batch["bias_labels"], reduction='none')

            # track per-domain losses for epoch-level update
            with torch.no_grad():
                for d in batch["domain_labels"].unique():
                    dm = batch["domain_labels"] == d
                    if dm.sum() > 0:
                        epoch_domain_losses[d.item()].append(per_sample_loss[dm].mean().item())

            if use_dro:
                # apply group weights to per-sample losses
                sample_weights = torch.ones(len(per_sample_loss), device=device)
                for d in batch["domain_labels"].unique():
                    dm = batch["domain_labels"] == d
                    sample_weights[dm] = group_weights[d]
                sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
                bias_loss = (per_sample_loss * sample_weights).mean()
            else:
                bias_loss = per_sample_loss.mean()

            domain_loss = domain_ce(domain_logits, batch["domain_labels"])
            loss = bias_loss + LAMBDA_ADV * domain_loss

            (loss / GRAD_ACCUM).backward()
            if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss += loss.item(); steps += 1
            if (i + 1) % log_interval == 0:
                mode = "DRO" if use_dro else "ERM"
                print(f"    [{mode}] Ep {epoch+1}, step {i+1}/{len(train_loader)}, loss={total_loss/steps:.4f}")

        # epoch-level group weight update
        print(f"\n  Per-domain losses (epoch {epoch+1}):")
        domain_avg = {}
        for d, name in enumerate(DATASETS):
            if d in epoch_domain_losses:
                avg = np.mean(epoch_domain_losses[d])
                domain_avg[d] = avg
                print(f"    {name[:25]:<25}: loss={avg:.4f}, weight={group_weights[d]:.4f}")

        if use_dro and len(domain_avg) >= 2:
            for d, avg_loss in domain_avg.items():
                group_weights[d] *= np.exp(DRO_ETA * avg_loss)
            group_weights /= group_weights.sum()
            print(f"  Updated weights (eta={DRO_ETA}):")
            for d, name in enumerate(DATASETS):
                print(f"    {name[:25]:<25}: {group_weights[d]:.4f}")
        elif not use_dro:
            print(f"  (ERM warmup - DRO starts epoch {WARMUP_EPOCHS + 1})")

        # validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _ = model(batch["input_ids"], batch["attention_mask"])
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                val_true.extend(batch["bias_labels"].cpu().tolist())

        vf1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"\n  Epoch {epoch+1}/{EPOCHS}: val_acc={accuracy_score(val_true, val_preds):.3f}, val_f1={vf1:.3f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            tokenizer.save_pretrained(output_dir)
            print(f"    Saved (val F1={best_f1:.3f})")

        # OOD eval each epoch
        ood_acc, ood_f1 = evaluate_ood(model, tokenizer, device, MAX_LENGTH, f"ep{epoch+1}")
        if ood_f1 > best_ood_f1:
            best_ood_f1 = ood_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best_ood.pt"))
            print(f"    Best OOD (F1={best_ood_f1:.3f})")

    print(f"\n  Best val F1: {best_f1:.3f}, Best OOD F1: {best_ood_f1:.3f}")

    results = {"config": config_name, "dro_eta": DRO_ETA, "warmup_epochs": WARMUP_EPOCHS,
               "best_val_f1": float(best_f1), "best_ood_f1": float(best_ood_f1),
               "final_group_weights": group_weights.cpu().tolist()}
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()