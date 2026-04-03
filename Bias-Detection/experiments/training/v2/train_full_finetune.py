"""
Full fine-tune DeBERTa-v3-large for political leaning.
Replicates Volf et al. (2025) methodology with 8/12 available datasets.

Paper methodology (Section 3.5.2 + 3.6):
  - Full fine-tune (all weights updated, not adapters/LoRA)
  - 10,000 samples per dataset with center class balancing
  - DeBERTa V3 large, max_length=256
  - lr=3e-5, warmup_ratio=0.1, batch_size=40 (grad_accum=4)
  - 4 epochs, F1 for best checkpoint
  - 15% test holdout, 100 per dataset for balanced validation
"""

import json, os, random, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", category=FutureWarning)

PARQUET_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets"
ALLSIDES_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets/other_data"
OUTPUT_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/replication/model"
RESULTS_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/replication/results"
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
EPOCHS = 4
BATCH_SIZE = 10
GRAD_ACCUM = 4
LR = 3e-5
WARMUP_RATIO = 0.1
TRAIN_SAMPLE_SIZE = 10000
SEED = 42
EVAL_ONLY = False

# dataset
class PoliticalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_length,
                            padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}


# data loading
def load_and_prepare_datasets(train_sample_size=10000, val_sample_size=100, test_fraction=0.15, seed=42):
    np.random.seed(seed); random.seed(seed)
    all_train_texts, all_train_labels = [], []
    all_val_texts, all_val_labels = [], []
    all_test_texts, all_test_labels = [], []

    for name in DATASETS:
        path = os.path.join(PARQUET_DIR, f"{name}.parquet")
        if not os.path.exists(path): print(f"  SKIP: {name}"); continue
        df = pd.read_parquet(path)
        if "title" in df.columns:
            df["text"] = df.apply(lambda r: (str(r["title"]) + "\n\n" + str(r["body"]))
                if pd.notna(r.get("title")) else str(r["body"]), axis=1)
        else: df["text"] = df["body"].astype(str)
        df["label"] = df["leaning"].map(LABEL2ID)
        df = df.dropna(subset=["label"]); df["label"] = df["label"].astype(int)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # stratified test holdout
        test_dfs = []
        for cls in df["label"].unique():
            cls_df = df[df["label"] == cls]
            test_dfs.append(cls_df.iloc[:max(1, int(len(cls_df) * test_fraction))])
        test_df = pd.concat(test_dfs)
        remaining = df.loc[~df.index.isin(test_df.index)]

        # val: balanced per dataset
        val_dfs = []
        for cls in remaining["label"].unique():
            cls_df = remaining[remaining["label"] == cls]
            val_dfs.append(cls_df.iloc[:min(val_sample_size // len(remaining["label"].unique()), len(cls_df))])
        val_df = pd.concat(val_dfs)
        train_pool = remaining.loc[~remaining.index.isin(val_df.index)]

        # train: sample with center balancing
        per_class = train_sample_size // len(train_pool["label"].unique())
        train_dfs = []
        for cls in train_pool["label"].unique():
            cls_df = train_pool[train_pool["label"] == cls]
            n = per_class

            # center multiplier
            if cls == LABEL2ID["center"] and name in CENTER_MULTIPLIERS:
                n = int(per_class * CENTER_MULTIPLIERS[name])

            n = min(n, len(cls_df))

            indices = np.linspace(0, len(cls_df) - 1, n, dtype=int)
            train_dfs.append(cls_df.iloc[indices])

        train_df = pd.concat(train_dfs)

        all_train_texts.extend(train_df["text"].tolist()); all_train_labels.extend(train_df["label"].tolist())
        all_val_texts.extend(val_df["text"].tolist()); all_val_labels.extend(val_df["label"].tolist())
        all_test_texts.extend(test_df["text"].tolist()); all_test_labels.extend(test_df["label"].tolist())
        print(f"  {name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    combined = list(zip(all_train_texts, all_train_labels))
    random.shuffle(combined)
    all_train_texts, all_train_labels = zip(*combined)
    print(f"\n  Total: train={len(all_train_texts)}, val={len(all_val_texts)}, test={len(all_test_texts)}")
    return (list(all_train_texts), list(all_train_labels), list(all_val_texts), list(all_val_labels),
            list(all_test_texts), list(all_test_labels))


def load_allsides_data(allsides_dir):
    articles, seen = [], set()
    for fp in sorted(Path(allsides_dir).glob("*.json")):
        with open(fp, "r") as f: data = json.load(f)
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
                    "true_label": ALLSIDES_MAP[bd], "label": LABEL2ID[ALLSIDES_MAP[bd]]})
    return articles

# training
def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM, lr=LR,
                warmup_ratio=WARMUP_RATIO, train_sample_size=TRAIN_SAMPLE_SIZE, seed=SEED):
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Full fine-tune: epochs={epochs}, bs={batch_size*grad_accum}, lr={lr}, device={device}")

    (train_texts, train_labels, val_texts, val_labels,
     test_texts, test_labels) = load_and_prepare_datasets(train_sample_size=train_sample_size, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        problem_type="single_label_classification").to(device)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader = DataLoader(PoliticalTextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PoliticalTextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH),
                            batch_size=batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    best_f1, best_epoch = 0, 0
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(epochs):
        model.train(); total_loss, steps = 0, 0; optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch); loss = outputs.loss / grad_accum; loss.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += outputs.loss.item(); steps += 1
            if (i + 1) % log_interval == 0:
                print(f"    Epoch {epoch+1}, step {i+1}/{len(train_loader)}, loss={total_loss/steps:.4f}")

        model.eval(); val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_preds.extend(torch.argmax(model(**batch).logits, dim=-1).cpu().tolist())
                val_true.extend(batch["labels"].cpu().tolist())
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/steps:.4f}, "
              f"val_acc={accuracy_score(val_true, val_preds):.3f}, val_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1; best_epoch = epoch + 1
            model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"    Saved (F1={best_f1:.3f})")

    print(f"\n  Best: epoch {best_epoch}, val F1={best_f1:.3f}")

    # in-distribution test
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device); model.eval()
    test_loader = DataLoader(PoliticalTextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH), batch_size=batch_size*2)
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            test_preds.extend(torch.argmax(model(**batch).logits, dim=-1).cpu().tolist())
            test_true.extend(batch["labels"].cpu().tolist())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    print(f"\n  In-distribution test: acc={test_acc:.1%}, F1={test_f1:.3f}")
    print(f"\n{classification_report(test_true, test_preds, target_names=LABELS, zero_division=0)}")
    cm = confusion_matrix(test_true, test_preds, labels=[0, 1, 2])
    print(f"  Confusion Matrix:")
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    with open(os.path.join(RESULTS_DIR, "training_results.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_f1": float(best_f1),
                   "test_accuracy": float(test_acc), "test_macro_f1": float(test_f1),
                   "train_size": len(train_texts), "test_size": len(test_texts)}, f, indent=2)
    return model, tokenizer

# ood test
def evaluate_allsides(model_path=None):
    if model_path is None: model_path = OUTPUT_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device); model.eval()

    articles = load_allsides_data(ALLSIDES_DIR)
    print(f"\n  AllSides eval: {len(articles)} articles")
    preds, truths = [], []
    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=MAX_LENGTH,
                          padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            pred = ID2LABEL[torch.argmax(model(**inputs).logits, dim=-1).item()]
        preds.append(pred); truths.append(article["true_label"])

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro", zero_division=0, labels=LABELS)
    print(f"  Accuracy: {acc:.1%}  Macro F1: {f1:.3f}")
    print(f"\n{classification_report(truths, preds, labels=LABELS, zero_division=0)}")
    cm = confusion_matrix(truths, preds, labels=LABELS)
    print(f"  Confusion Matrix:")
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    # breakdown
    detail_groups = {}
    for article, pred in zip(articles, preds):
        d = article["bias_detail"]
        if d not in detail_groups: detail_groups[d] = {"preds": [], "truths": []}
        detail_groups[d]["preds"].append(pred); detail_groups[d]["truths"].append(article["true_label"])
    print(f"\n  Breakdown:")
    for detail in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
        if detail in detail_groups:
            g = detail_groups[detail]
            nc = sum(p == t for p, t in zip(g["preds"], g["truths"]))
            print(f"    {detail:<12}: {accuracy_score(g['truths'], g['preds']):.1%} ({nc}/{len(g['preds'])})")

    with open(os.path.join(RESULTS_DIR, "allsides_eval_results.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(f1)}, f, indent=2)


if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not EVAL_ONLY:
        train_model()
    evaluate_allsides()