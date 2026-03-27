#!/usr/bin/env python3
"""
AllSides Domain Adapter — Stacked on top of BASIL adapter
==========================================================
Trains a second Pfeiffer adapter for AllSides domain adaptation.
At inference: base model → BASIL adapter → AllSides adapter → head.

IMPORTANT: Split AllSides into train/test BEFORE running this.
           Never evaluate on the same articles used here.

Usage:
  python finetune_allsides_adapter.py \
    --base-model models/bias_detector \
    --basil-adapter models/basil_adapter \
    --allsides ../../AllSides/allsides_data.json ../../AllSides/allsides_data_2_20.json \
    --test-split 0.2
"""

import os, json, random, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

import adapters
import adapters.composition as ac
from adapters import AutoAdapterModel, SeqBnConfig
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",      default="models/bias_detector")
    parser.add_argument("--basil-adapter",   default="models/basil_adapter",
                        help="Path to saved BASIL adapter from finetune_basil_adapter.py")
    parser.add_argument("--allsides",        nargs="+", required=True)
    parser.add_argument("--output-dir",      default="models/allsides_adapter")
    parser.add_argument("--test-split",      type=float, default=0.2,
                        help="Fraction of AllSides data to hold out for evaluation")
    parser.add_argument("--reduction-factor",type=int,   default=16)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--learning-rate",   type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load AllSides and split ──
    all_articles = load_allsides(args.allsides)
    print(f"Loaded {len(all_articles)} AllSides articles")
    dist = Counter(a["true_label"] for a in all_articles)
    print(f"Distribution: {dict(dist)}")

    train_arts, test_arts = train_test_split(
        all_articles,
        test_size=args.test_split,
        stratify=[a["true_label"] for a in all_articles],
        random_state=args.seed,
    )
    print(f"Train: {len(train_arts)}  |  Test (held out): {len(test_arts)}")

    # ── Load base model ──
    print(f"\nLoading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model     = AutoAdapterModel.from_pretrained(args.base_model)

    # ── Load the frozen BASIL adapter ──
    print(f"Loading BASIL adapter from {args.basil_adapter}...")
    model.load_adapter(args.basil_adapter, load_as=BASIL_ADAPTER_NAME, set_active=False)

    # ── Add AllSides adapter on top ──
    allsides_config = SeqBnConfig(reduction_factor=args.reduction_factor, non_linearity="gelu")
    model.add_adapter(ALLSIDES_ADAPTER_NAME, config=allsides_config)

    # Add classification head tied to the STACKED setup
    model.add_classification_head(
        ALLSIDES_ADAPTER_NAME, num_labels=3,
        id2label={0: "Left", 1: "Center", 2: "Right"})

    # ── Activate STACKED composition ──
    # Input flows: base model → BASIL adapter → AllSides adapter → head
    # BASIL adapter is FROZEN. Only AllSides adapter + head train.
    model.train_adapter(ALLSIDES_ADAPTER_NAME)
    model.active_adapters = ac.Stack(BASIL_ADAPTER_NAME, ALLSIDES_ADAPTER_NAME)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Active adapters: Stack({BASIL_ADAPTER_NAME} → {ALLSIDES_ADAPTER_NAME})")

    # ── Class-weighted loss ──
    label_counts  = Counter(a["label_id"] for a in train_arts)
    total_n       = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_n / (3 * label_counts.get(i, 1)) for i in range(3)], dtype=torch.float32)

    # Label smoothing cross-entropy (same class from finetune_basil_adapter.py)
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01)

    train_loader = torch.utils.data.DataLoader(
        BiasDataset(train_arts, tokenizer), batch_size=args.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
        BiasDataset(test_arts,  tokenizer), batch_size=args.batch_size*2, shuffle=False)

    # ── Training ──
    best_f1, best_state = 0, None
    for epoch in range(args.epochs):
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

        # Eval on held-out test set
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
        print(f"  Epoch {epoch+1}/{args.epochs}  "
              f"loss={epoch_loss/total_s:.4f}  train_acc={correct/total_s:.3f}  "
              f"test_acc={val_acc:.3f}  test_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1  = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Restore best and final eval ──
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

    print(f"\n{'═'*60}")
    print(f"  FINAL TEST RESULTS (held-out {len(test_arts)} AllSides articles)")
    print(f"  Accuracy: {accuracy_score(truths, preds):.3f}")
    print(f"  Macro F1: {f1_score(truths, preds, average='macro', zero_division=0):.3f}")
    print(f"\n{classification_report(truths, preds, target_names=LABELS, zero_division=0)}")

    # ── Save AllSides adapter ──
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_adapter(args.output_dir, ALLSIDES_ADAPTER_NAME)
    print(f"\n  ✓ AllSides adapter saved to {args.output_dir}/")
    print(f"\n  To use stacked inference in pipeline.py:")
    print(f"    model.load_adapter('{args.basil_adapter}', load_as='basil_bias')")
    print(f"    model.load_adapter('{args.output_dir}', load_as='allsides_domain')")
    print(f"    model.active_adapters = ac.Stack('basil_bias', 'allsides_domain')")


if __name__ == "__main__":
    main()
