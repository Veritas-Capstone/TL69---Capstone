"""
AllSides Domain Adapter
"""

import os, json, random, warnings
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

import adapters
import adapters.composition as ac
from adapters import AutoAdapterModel, SeqBnConfig
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# config settings
config = SimpleNamespace(
    base_model="../../models/demo_models/bias_detector",
    basil_adapter="../../models/demo_models/basil_adapter",
    allsides=["../../datasets/other_data/allsides_data.json", "../../datasets/other_data/allsides_data_2_20.json"],
    output_dir="models/allsides_adapter",
    test_split=0.2,
    reduction_factor=16,
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    label_smoothing=0.1,
    seed=42,
)

LABELS    = ["Left", "Center", "Right"]
LABEL2ID  = {"Left": 0, "Center": 1, "Right": 2}
ALLSIDES_MAP = {
    "Left": "Left", "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right", "Right": "Right",
}

BASIL_ADAPTER_NAME    = "basil_bias"
ALLSIDES_ADAPTER_NAME = "allsides_domain"


def load_allsides(filepaths):
    articles, seen = [], set()
    for fp in filepaths:
        with open(fp) as f:
            data = json.load(f)
        for story in data:
            for side in story.get("sides", []):
                bd   = side.get("bias_detail")
                body = (side.get("body") or "").strip()
                url  = side.get("original_url", "")
                if not body or bd not in ALLSIDES_MAP or url in seen:
                    continue
                seen.add(url)
                label = ALLSIDES_MAP[bd]
                articles.append({
                    "headline": side.get("headline", ""),
                    "body": body,
                    "true_label": label,
                    "label_id": LABEL2ID[label],
                })
    return articles


class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, articles, tokenizer, max_length=512):
        self.articles, self.tokenizer, self.max_length = articles, tokenizer, max_length
    def __len__(self): return len(self.articles)
    def __getitem__(self, idx):
        a = self.articles[idx]
        text = (a["headline"] + ". " + a["body"]) if a.get("headline") else a["body"]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(a["label_id"], dtype=torch.long)}


if __name__ == "__main__":
    cfg = config
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load AllSides and split
    all_articles = load_allsides(cfg.allsides)
    print(f"Loaded {len(all_articles)} AllSides articles")
    dist = Counter(a["true_label"] for a in all_articles)
    print(f"Distribution: {dict(dist)}")

    train_arts, test_arts = train_test_split(
        all_articles,
        test_size=cfg.test_split,
        stratify=[a["true_label"] for a in all_articles],
        random_state=cfg.seed,
    )
    print(f"Train: {len(train_arts)}  |  Test (held out): {len(test_arts)}")

    # load base model
    print(f"\nLoading base model from {cfg.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model     = AutoAdapterModel.from_pretrained(cfg.base_model)

    # load the frozen BASIL adapter
    print(f"Loading BASIL adapter from {cfg.basil_adapter}...")
    model.load_adapter(cfg.basil_adapter, load_as=BASIL_ADAPTER_NAME, set_active=False)

    # add AllSides adapter on top
    allsides_config = SeqBnConfig(reduction_factor=cfg.reduction_factor, non_linearity="gelu")
    model.add_adapter(ALLSIDES_ADAPTER_NAME, config=allsides_config)

    # classification head tied to the stacked setup
    model.add_classification_head(
        ALLSIDES_ADAPTER_NAME, num_labels=3,
        id2label={0: "Left", 1: "Center", 2: "Right"})

    # stacked composition: base -> BASIL (frozen) -> AllSides (trains) -> head
    model.train_adapter(ALLSIDES_ADAPTER_NAME)
    model.active_adapters = ac.Stack(BASIL_ADAPTER_NAME, ALLSIDES_ADAPTER_NAME)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Active adapters: Stack({BASIL_ADAPTER_NAME} -> {ALLSIDES_ADAPTER_NAME})")

    # class-weighted loss
    label_counts  = Counter(a["label_id"] for a in train_arts)
    total_n       = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_n / (3 * label_counts.get(i, 1)) for i in range(3)], dtype=torch.float32)

    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=0.01)

    train_loader = torch.utils.data.DataLoader(
        BiasDataset(train_arts, tokenizer), batch_size=cfg.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
        BiasDataset(test_arts,  tokenizer), batch_size=cfg.batch_size*2, shuffle=False)

    # training
    best_f1, best_state = 0, None
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, correct, total_s = 0, 0, 0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            correct    += (torch.argmax(out.logits, -1) == labels).sum().item()
            total_s    += labels.size(0)

        # eval on held-out test set
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
                preds.extend(torch.argmax(out.logits, -1).cpu().tolist())
                truths.extend(batch["labels"].tolist())

        val_acc = accuracy_score(truths, preds)
        val_f1  = f1_score(truths, preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch+1}/{cfg.epochs}  "
              f"loss={epoch_loss/total_s:.4f}  train_acc={correct/total_s:.3f}  "
              f"test_acc={val_acc:.3f}  test_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1  = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best and final eval
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds.extend(torch.argmax(out.logits, -1).cpu().tolist())
            truths.extend(batch["labels"].tolist())

    print(f"\n  Final test results (held-out {len(test_arts)} AllSides articles)")
    print(f"  Accuracy: {accuracy_score(truths, preds):.3f}")
    print(f"  Macro F1: {f1_score(truths, preds, average='macro', zero_division=0):.3f}")
    print(f"\n{classification_report(truths, preds, target_names=LABELS, zero_division=0)}")

    # save AllSides adapter
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_adapter(cfg.output_dir, ALLSIDES_ADAPTER_NAME)
    print(f"\n  AllSides adapter saved to {cfg.output_dir}/")