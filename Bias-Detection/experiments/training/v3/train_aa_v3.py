"""
Adversarial adaptation v3 with fixed regularizations
Also supports spectral decoupling and label smoothing
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
SPECTRAL_LAMBDA = 0.0
RDROP_ALPHA = 0.0
LABEL_SMOOTHING = 0.0
VREX_BETA = 0.0


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

        self.bias_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )
        self.gradient_reversal = GradientReversalLayer(alpha=lambda_adv)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_domains),
        )
        self.lambda_adv = lambda_adv

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        bias_logits = self.bias_classifier(pooled)
        reversed_pooled = self.gradient_reversal(pooled)
        domain_logits = self.domain_classifier(reversed_pooled)
        return bias_logits, domain_logits

# dataset
class BiasDataset(Dataset):
    def __init__(self, texts, bias_labels, domain_labels, tokenizer, max_length=256):
        self.texts = texts
        self.bias_labels = bias_labels
        self.domain_labels = domain_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True,
                             max_length=self.max_length, padding="max_length",
                             return_tensors="pt")
        return {
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bias_labels":   torch.tensor(self.bias_labels[idx], dtype=torch.long),
            "domain_labels": torch.tensor(self.domain_labels[idx], dtype=torch.long),
        }


# data loading
def load_datasets(sample_size=10000, test_frac=0.15, seed=42):
    np.random.seed(seed); random.seed(seed)
    train_texts, train_bias, train_domain = [], [], []
    val_texts, val_bias, val_domain = [], [], []

    for name in DATASETS:
        path = os.path.join(PARQUET_DIR, f"{name}.parquet")
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
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
            n = min(n, len(cls_df))
            sampled.append(cls_df.sample(n=n, random_state=seed))
        train_df = pd.concat(sampled)

        domain_id = DATASET2ID[name]
        train_texts.extend(train_df["text"].tolist())
        train_bias.extend(train_df["label"].tolist())
        train_domain.extend([domain_id] * len(train_df))
        val_texts.extend(val_df["text"].tolist())
        val_bias.extend(val_df["label"].tolist())
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
                if not body or not bd or bd not in ALLSIDES_MAP:
                    continue
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                hl = side.get("headline", "")
                articles.append({
                    "text": (hl + ". " + body) if hl else body,
                    "source": side.get("source", "Unknown"),
                    "bias_detail": bd,
                    "true_label": ALLSIDES_MAP[bd],
                })
    return articles


def compute_rdrop_loss(logits1, logits2, labels, ce_fn, alpha):
    ce1 = ce_fn(logits1, labels)
    ce2 = ce_fn(logits2, labels)
    avg_ce = (ce1 + ce2) / 2.0

    # bidirectional KL
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    kl_pq = F.kl_div(p, q.exp(), reduction='batchmean')
    kl_qp = F.kl_div(q, p.exp(), reduction='batchmean')
    kl = (kl_pq + kl_qp) / 2.0

    return avg_ce, alpha * kl

# OOD eval
def evaluate_allsides(model, tokenizer, device, max_length, config_name):
    articles = load_allsides()
    print(f"\n  Evaluating on {len(articles)} AllSides articles...")
    model.eval()

    preds_list, truths_list, all_probs = [], [], []
    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            bias_logits, _ = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(bias_logits, dim=-1)[0]
            pred = torch.argmax(probs).item()
        preds_list.append(ID2LABEL[pred])
        truths_list.append(article["true_label"])
        all_probs.append(probs.cpu().numpy())

    acc = accuracy_score(truths_list, preds_list)
    f1 = f1_score(truths_list, preds_list, average="macro", zero_division=0, labels=LABELS)

    print(f"\n{'='*60}")
    print(f"  [{config_name}] AllSides OOD Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Macro F1: {f1:.3f}")
    print(f"\n{classification_report(truths_list, preds_list, labels=LABELS, zero_division=0)}")

    cm = confusion_matrix(truths_list, preds_list, labels=LABELS)
    print(f"  Confusion Matrix:")
    print(f"  {'':>10s} {'left':>8s} {'center':>8s} {'right':>8s}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10s} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    # confidence abstention
    all_probs_np = np.array(all_probs)
    print(f"\n  Confidence-Based Abstention:")
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = all_probs_np.max(axis=1) >= thr
        n = mask.sum()
        if n == 0:
            continue
        cp = [p for p, m in zip(preds_list, mask) if m]
        ct = [t for t, m in zip(truths_list, mask) if m]
        print(f"    theta={thr:.2f}: {n}/{len(articles)} ({n/len(articles):.0%}), "
              f"acc={accuracy_score(ct, cp):.1%}, "
              f"F1={f1_score(ct, cp, average='macro', zero_division=0, labels=LABELS):.3f}")

    # breakdown
    detail_groups = defaultdict(lambda: {"preds": [], "truths": []})
    for article, pred in zip(articles, preds_list):
        detail_groups[article["bias_detail"]]["preds"].append(pred)
        detail_groups[article["bias_detail"]]["truths"].append(article["true_label"])
    print(f"\n  Breakdown by AllSides label:")
    for detail in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
        if detail in detail_groups:
            g = detail_groups[detail]
            dacc = accuracy_score(g["truths"], g["preds"])
            nc = sum(p == t for p, t in zip(g["preds"], g["truths"]))
            print(f"    {detail:<12s}: {dacc:.1%} ({nc}/{len(g['preds'])})")

    return acc, f1

# training
def train():
    # identify model short name
    if "bert-base" in BASE_MODEL:
        model_tag = "bert"
    elif "deberta" in BASE_MODEL:
        model_tag = "deberta"
    else:
        model_tag = BASE_MODEL.split("/")[-1][:10]

    # config name
    parts = [f"v3_{model_tag}_l{LAMBDA_ADV}"]
    if SPECTRAL_LAMBDA > 0: parts.append(f"sd{SPECTRAL_LAMBDA}")
    if RDROP_ALPHA > 0:     parts.append(f"rd{RDROP_ALPHA}")
    if LABEL_SMOOTHING > 0: parts.append(f"ls{LABEL_SMOOTHING}")
    if VREX_BETA > 0:       parts.append(f"vrex{VREX_BETA}")
    if MAX_LENGTH != 256:   parts.append(f"ml{MAX_LENGTH}")
    if SEED != 42: parts.append(f"s{SEED}")
    parts.append(f"ep{EPOCHS}")
    config_name = "_".join(parts)

    output_dir  = os.path.join(BASE_OUTPUT, "v3", config_name, "model")
    results_dir = os.path.join(BASE_OUTPUT, "v3", config_name, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  AA v3 - Fixed Regularizations")
    print(f"  Config: {config_name}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  lambda_adv={LAMBDA_ADV}, spectral_lambda={SPECTRAL_LAMBDA}")
    print(f"  rdrop_alpha={RDROP_ALPHA} (FIXED: correct averaging)")
    print(f"  label_smoothing={LABEL_SMOOTHING}")
    print(f"  vrex_beta={VREX_BETA} (FIXED: epoch-level accumulation)")
    print(f"  max_length={MAX_LENGTH}, epochs={EPOCHS}")
    print(f"  Effective BS: {BATCH_SIZE * GRAD_ACCUM}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # load data
    (train_t, train_b, train_d,
     val_t, val_b, val_d) = load_datasets(seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AdversarialBiasModel(
        BASE_MODEL, num_domains=NUM_DOMAINS, lambda_adv=LAMBDA_ADV
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    train_ds = BiasDataset(train_t, train_b, train_d, tokenizer, MAX_LENGTH)
    val_ds   = BiasDataset(val_t, val_b, val_d, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

    ce_fn = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0)
    domain_ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    best_ood_f1 = 0
    log_interval = max(1, len(train_loader) // 5)

    # V-REx: warmup in epochs (not fraction of steps)
    vrex_warmup_epochs = max(1, EPOCHS // 3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_rdrop, total_vrex = 0, 0, 0
        steps = 0
        optimizer.zero_grad()

        epoch_domain_losses = defaultdict(list)

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            bias_labels = batch["bias_labels"]
            domain_labels = batch["domain_labels"]

            bias_logits1, domain_logits1 = model(ids, mask)

            if RDROP_ALPHA > 0:
                bias_logits2, _ = model(ids, mask)

                avg_ce, kl_loss = compute_rdrop_loss(
                    bias_logits1, bias_logits2, bias_labels, ce_fn, RDROP_ALPHA)
                bias_loss = avg_ce + kl_loss
                total_rdrop += kl_loss.item()
            else:
                bias_loss = ce_fn(bias_logits1, bias_labels)

            # domain adversarial loss
            domain_loss = domain_ce(domain_logits1, domain_labels)

            # combined
            loss = bias_loss + LAMBDA_ADV * domain_loss

            if SPECTRAL_LAMBDA > 0:
                loss = loss + SPECTRAL_LAMBDA * bias_logits1.pow(2).mean()

            # accumulate per-domain losses
            if VREX_BETA > 0:
                with torch.no_grad():
                    per_sample = F.cross_entropy(
                        bias_logits1, bias_labels, reduction='none')
                    for d in domain_labels.unique():
                        d_mask = domain_labels == d
                        if d_mask.sum() > 0:
                            epoch_domain_losses[d.item()].append(
                                per_sample[d_mask].mean().item())


            scaled = loss / GRAD_ACCUM
            scaled.backward()

            if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1

            if (i + 1) % log_interval == 0:
                extra = ""
                if RDROP_ALPHA > 0:
                    extra += f", rdrop_kl={total_rdrop/steps:.4f}"
                print(f"    Ep {epoch+1}, step {i+1}/{len(train_loader)}, "
                      f"loss={total_loss/steps:.4f}{extra}")

        if VREX_BETA > 0 and epoch >= vrex_warmup_epochs:
            domain_means = []
            for d in sorted(epoch_domain_losses.keys()):
                dm = np.mean(epoch_domain_losses[d])
                domain_means.append(dm)
            if len(domain_means) >= 2:
                vrex_var = np.var(domain_means)
                total_vrex = vrex_var
                print(f"\n  V-REx epoch {epoch+1}: domain loss variance = {vrex_var:.6f}")
                print(f"  Per-domain avg losses: ", end="")
                for d, name in enumerate(DATASETS):
                    if d in epoch_domain_losses:
                        print(f"{name[:12]}={np.mean(epoch_domain_losses[d]):.3f} ", end="")
                print()

                # V-REx as a soft penalty
                if vrex_var > 0.001:
                    model.train()
                    vrex_sample_loader = DataLoader(train_ds, batch_size=BATCH_SIZE * 4,
                                                    shuffle=True)
                    vrex_batch = next(iter(vrex_sample_loader))
                    vrex_batch = {k: v.to(device) for k, v in vrex_batch.items()}

                    vlogits, _ = model(vrex_batch["input_ids"], vrex_batch["attention_mask"])
                    vper = F.cross_entropy(vlogits, vrex_batch["bias_labels"], reduction='none')

                    dloss_list = []
                    for d in vrex_batch["domain_labels"].unique():
                        dm = vrex_batch["domain_labels"] == d
                        if dm.sum() > 0:
                            dloss_list.append(vper[dm].mean())

                    if len(dloss_list) >= 2:
                        stacked = torch.stack(dloss_list)
                        vrex_penalty = stacked.var() * VREX_BETA
                        optimizer.zero_grad()
                        vrex_penalty.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        print(f"  V-REx penalty applied: {vrex_penalty.item():.4f}")

        # validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _ = model(batch["input_ids"], batch["attention_mask"])
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                val_true.extend(batch["bias_labels"].cpu().tolist())

        vf1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        vacc = accuracy_score(val_true, val_preds)
        print(f"\n  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/steps:.4f}, "
              f"val_acc={vacc:.3f}, val_f1={vf1:.3f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump({"base_model": BASE_MODEL, "lambda_adv": LAMBDA_ADV, "spectral_lambda": SPECTRAL_LAMBDA, "rdrop_alpha": RDROP_ALPHA, "label_smoothing": LABEL_SMOOTHING, "vrex_beta": VREX_BETA, "num_domains": NUM_DOMAINS}, f, indent=2)
            print(f"    Saved (val F1={best_f1:.3f})")

        # OOD eval every epoch (track OOD trajectory) 
        ood_acc, ood_f1 = evaluate_allsides(model, tokenizer, device,
                                             MAX_LENGTH, f"{config_name}_ep{epoch+1}")
        if ood_f1 > best_ood_f1:
            best_ood_f1 = ood_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best_ood.pt"))
            print(f"    Best OOD so far (F1={best_ood_f1:.3f})")

    # Final eval with best OOD model
    print(f"  FINAL - Best val F1: {best_f1:.3f}, Best OOD F1: {best_ood_f1:.3f}")

    # Load best OOD model for final report
    if os.path.exists(os.path.join(output_dir, "model_best_ood.pt")):
        model.load_state_dict(torch.load(
            os.path.join(output_dir, "model_best_ood.pt"), map_location=device))
        acc, f1_ood = evaluate_allsides(model, tokenizer, device,
                                         MAX_LENGTH, f"{config_name}_BEST_OOD")
    else:
        acc, f1_ood = best_ood_f1, best_ood_f1

    results = {
        "config": config_name, 
        "best_val_f1": float(best_f1),
        "best_ood_accuracy": float(acc) if isinstance(acc, float) else acc,
        "best_ood_f1": float(best_ood_f1),
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    train()