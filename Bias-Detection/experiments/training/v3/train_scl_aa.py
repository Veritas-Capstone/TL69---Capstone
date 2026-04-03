"""
Supervised contrastive learning + AA (Gunel et al., ICLR 2021)
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
BASE_MODEL   = "microsoft/deberta-v3-large"

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
EPOCHS = 4
BATCH_SIZE = 10
GRAD_ACCUM = 4
LR = 3e-5
LAMBDA_ADV = 0.7
MAX_LENGTH = 256
SEED = 42
SCL_ALPHA = 0.3 
SCL_TEMP = 0.1 
PROJ_DIM = 128


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


class SCLAdversarialModel(nn.Module):
    def __init__(self, model_name, num_labels=3, num_domains=NUM_DOMAINS,
                 lambda_adv=0.7, proj_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.bias_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, num_labels))

        self.gradient_reversal = GradientReversalLayer(alpha=lambda_adv)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_domains))

        # 2-layer projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, proj_dim))

        self.lambda_adv = lambda_adv

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        bias_logits = self.bias_classifier(pooled)
        domain_logits = self.domain_classifier(self.gradient_reversal(pooled))
        proj = F.normalize(self.projection_head(pooled), dim=-1)

        return bias_logits, domain_logits, proj


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # positive mask: same label, excluding self
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        mask = mask - self_mask

        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # log-softmax over negatives
        logits_mask = 1.0 - self_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # mean of log-prob over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return -mean_log_prob.mean()


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


def train():
    config_name = f"scl_a{SCL_ALPHA}_t{SCL_TEMP}_l{LAMBDA_ADV}"
    if MAX_LENGTH != 256:
        config_name += f"_ml{MAX_LENGTH}"

    output_dir  = os.path.join(BASE_OUTPUT, "scl_aa", config_name, "model")
    results_dir = os.path.join(BASE_OUTPUT, "scl_aa", config_name, "results")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  SCL+AA: {config_name}")
    print(f"  scl_alpha={SCL_ALPHA}, scl_temp={SCL_TEMP}, lambda={LAMBDA_ADV}, "
          f"bs={BATCH_SIZE*GRAD_ACCUM}, device={device}")

    (train_t, train_b, train_d, val_t, val_b, val_d) = load_datasets(seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = SCLAdversarialModel(BASE_MODEL, num_domains=NUM_DOMAINS,
                                 lambda_adv=LAMBDA_ADV, proj_dim=PROJ_DIM).to(device)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_ds = BiasDataset(train_t, train_b, train_d, tokenizer, MAX_LENGTH)
    val_ds = BiasDataset(val_t, val_b, val_d, tokenizer, MAX_LENGTH)
    # drop_last for SCL (needs full batches for positive/negative pairs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_ce = nn.CrossEntropyLoss()
    scl_fn = SupConLoss(temperature=SCL_TEMP)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, scl_total, steps = 0, 0, 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            bias_logits, domain_logits, proj = model(batch["input_ids"], batch["attention_mask"])

            bias_loss = ce_fn(bias_logits, batch["bias_labels"])
            domain_loss = domain_ce(domain_logits, batch["domain_labels"])
            scl_loss = scl_fn(proj, batch["bias_labels"])

            # L_CE + alpha * L_SCL + lambda * L_adv
            loss = bias_loss + SCL_ALPHA * scl_loss + LAMBDA_ADV * domain_loss

            (loss / GRAD_ACCUM).backward()
            if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss += loss.item()
            scl_total += scl_loss.item()
            steps += 1

            if (i + 1) % log_interval == 0:
                print(f"    Ep {epoch+1}, step {i+1}/{len(train_loader)}, "
                      f"loss={total_loss/steps:.4f}, scl={scl_total/steps:.4f}")

        # validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _, _ = model(batch["input_ids"], batch["attention_mask"])
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                val_true.extend(batch["bias_labels"].cpu().tolist())

        vf1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/steps:.4f}, "
              f"val_acc={accuracy_score(val_true, val_preds):.3f}, val_f1={vf1:.3f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump({"scl_alpha": SCL_ALPHA, "scl_temp": SCL_TEMP,
                           "lambda_adv": LAMBDA_ADV, "proj_dim": PROJ_DIM}, f, indent=2)
            print(f"    Saved (F1={best_f1:.3f})")

    # AllSides OOD eval
    print(f"\n  Best val F1: {best_f1:.3f}")
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt"), map_location=device))
    model.eval()

    articles = load_allsides()
    print(f"  Evaluating on {len(articles)} AllSides articles...")

    preds_list, truths_list = [], []
    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=MAX_LENGTH,
                           padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            logits, _, _ = model(inputs["input_ids"], inputs["attention_mask"])
        preds_list.append(ID2LABEL[torch.argmax(logits, dim=-1).item()])
        truths_list.append(article["true_label"])

    acc = accuracy_score(truths_list, preds_list)
    f1 = f1_score(truths_list, preds_list, average="macro", zero_division=0, labels=LABELS)

    print(f"\n  [{config_name}] AllSides OOD: acc={acc:.1%}, F1={f1:.3f}")
    print(classification_report(truths_list, preds_list, labels=LABELS, zero_division=0))

    cm = confusion_matrix(truths_list, preds_list, labels=LABELS)
    print(f"  Confusion Matrix:")
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    results = {
        "config": config_name, "scl_alpha": SCL_ALPHA, "scl_temp": SCL_TEMP,
        "lambda_adv": LAMBDA_ADV, "ood_accuracy": float(acc),
        "ood_macro_f1": float(f1), "best_val_f1": float(best_f1),
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()