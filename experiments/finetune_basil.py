#!/usr/bin/env python3
"""
LoRA Fine-Tuning on BASIL for Political Bias Detection
========================================================
Fine-tunes matous-volf/political-leaning-deberta-large on the BASIL dataset
using LoRA (Low-Rank Adaptation) for parameter-efficient training.

Key design decisions:
  - Event-level cross-validation (no data leakage between same-event articles)
  - LoRA targets attention layers only (~0.3% trainable params)
  - Class-weighted loss to handle any imbalance
  - Label smoothing for better generalization
  - Optional data augmentation via back-translation / paraphrase

Requirements:
  pip install peft transformers datasets accelerate scikit-learn --break-system-packages

Usage:
  # Standard 5-fold CV training
  python finetune_basil.py --basil-dir BASIL

  # Quick test (1 fold only)
  python finetune_basil.py --basil-dir BASIL --folds 1

  # Train final model on all data (after CV gives good results)
  python finetune_basil.py --basil-dir BASIL --train-final

  # Use AllSides data for augmentation
  python finetune_basil.py --basil-dir BASIL --allsides allsides_data.json

  # Adjust LoRA rank
  python finetune_basil.py --basil-dir BASIL --lora-r 32 --lora-alpha 64
"""

import os
import sys
import json
import argparse
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

LABELS = ["Left", "Center", "Right"]
LABEL2ID = {"Left": 0, "Center": 1, "Right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# BASIL source → bias label mapping
SOURCE_LABEL_MAP = {
    "hpo": "Left",     # HuffPost
    "nyt": "Center",   # New York Times
    "fox": "Right",    # Fox News
}


# ═══════════════════════════════════════════════════════════════════════
# BASIL Loading
# ═══════════════════════════════════════════════════════════════════════

def load_basil_data(basil_dir, label_source="source"):
    """
    Load BASIL dataset. Tries your existing basil_loader.py first,
    falls back to a built-in parser if not available.

    Returns list of dicts with: event_id, source, headline, body,
    true_label, label_id
    """
    # ── Try existing basil_loader.py (which already works) ──
    try:
        from basil_loader import load_basil as _load_basil
        print("  Using existing basil_loader.py")
        raw_articles = _load_basil(basil_dir, label_source=label_source)

        # Ensure label_id is present
        articles = []
        for a in raw_articles:
            a["label_id"] = LABEL2ID.get(a["true_label"], 1)
            if "headline" not in a:
                a["headline"] = ""
            articles.append(a)
        return articles

    except ImportError:
        print("  basil_loader.py not found, using built-in parser")

    # ── Built-in fallback parser ──
    basil_path = Path(basil_dir)
    articles = []
    skipped = {"no_source": 0, "short_body": 0, "parse_error": 0}

    # BASIL repo structure: BASIL/articles/0/fox.json etc.
    # Try multiple possible paths
    candidates = [
        basil_path / "articles",   # BASIL/articles/
        basil_path,                 # if user points directly at articles/
    ]

    data_root = None
    for candidate in candidates:
        if candidate.is_dir():
            # Check if it has numbered subdirectories
            subdirs = [d for d in candidate.iterdir()
                       if d.is_dir() and d.name.isdigit()]
            if subdirs:
                data_root = candidate
                break

    if data_root is None:
        print(f"  ERROR: Could not find event directories in {basil_dir}")
        print(f"  Expected structure: {basil_dir}/articles/0/fox.json")
        print(f"  Or: {basil_dir}/0/fox.json")
        return []

    print(f"  Found articles in: {data_root}")

    for event_dir in sorted(data_root.iterdir()):
        if not event_dir.is_dir() or not event_dir.name.isdigit():
            continue

        event_id = event_dir.name

        for json_file in sorted(event_dir.glob("*.json")):
            source_key = json_file.stem.lower()  # "fox", "hpo", "nyt"

            if source_key not in SOURCE_LABEL_MAP:
                skipped["no_source"] += 1
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                skipped["parse_error"] += 1
                continue

            # Extract body text — handle multiple JSON formats
            body_parts = []

            if isinstance(data, list):
                # Format A: list of sentence dicts
                for item in data:
                    if isinstance(item, dict):
                        text = (item.get("sentence") or item.get("text")
                                or item.get("body") or "")
                        if text.strip():
                            body_parts.append(text.strip())
                    elif isinstance(item, str) and item.strip():
                        body_parts.append(item.strip())

            elif isinstance(data, dict):
                # Format B: dict with body/sentences key
                if "body" in data:
                    body_content = data["body"]
                    if isinstance(body_content, str):
                        body_parts = [body_content]
                    elif isinstance(body_content, list):
                        for item in body_content:
                            if isinstance(item, dict):
                                text = item.get("sentence", item.get("text", ""))
                            else:
                                text = str(item)
                            if text.strip():
                                body_parts.append(text.strip())
                elif "sentences" in data:
                    for item in data["sentences"]:
                        if isinstance(item, dict):
                            text = item.get("sentence", item.get("text", ""))
                        else:
                            text = str(item)
                        if text.strip():
                            body_parts.append(text.strip())
                elif "text" in data:
                    body_parts = [data["text"]]

            body = " ".join(body_parts)

            if len(body.split()) < 20:
                skipped["short_body"] += 1
                continue

            true_label = SOURCE_LABEL_MAP[source_key]
            headline = body_parts[0] if body_parts else ""

            articles.append({
                "event_id": event_id,
                "source": source_key,
                "file_suffix": source_key,
                "headline": headline,
                "body": body,
                "true_label": true_label,
                "label_id": LABEL2ID[true_label],
            })

    print(f"  BASIL loaded: {len(articles)} articles")
    print(f"  Skipped: {skipped}")
    dist = Counter(a["true_label"] for a in articles)
    print(f"  Label distribution: {dict(dist)}")

    return articles


# ═══════════════════════════════════════════════════════════════════════
# Event-Level Cross-Validation Splits
# ═══════════════════════════════════════════════════════════════════════

def create_event_splits(articles, n_folds=5, seed=42):
    """
    Create cross-validation splits at the EVENT level.

    Critical: Articles from the same event (same story, 3 outlets)
    must stay together in either train or val — never split across.
    This prevents the model from memorizing event-specific patterns.
    """
    # Group articles by event
    events = defaultdict(list)
    for a in articles:
        events[a["event_id"]].append(a)

    event_ids = sorted(events.keys())
    random.seed(seed)
    random.shuffle(event_ids)

    # Create folds
    fold_size = len(event_ids) // n_folds
    folds = []

    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            val_events = set(event_ids[start:])
        else:
            val_events = set(event_ids[start:start + fold_size])

        train_events = set(event_ids) - val_events

        train_articles = [a for a in articles if a["event_id"] in train_events]
        val_articles = [a for a in articles if a["event_id"] in val_events]

        folds.append({
            "fold": i,
            "train": train_articles,
            "val": val_articles,
            "n_train_events": len(train_events),
            "n_val_events": len(val_events),
        })

        train_dist = Counter(a["true_label"] for a in train_articles)
        val_dist = Counter(a["true_label"] for a in val_articles)
        print(f"  Fold {i}: train={len(train_articles)} ({dict(train_dist)}), "
              f"val={len(val_articles)} ({dict(val_dist)})")

    return folds


# ═══════════════════════════════════════════════════════════════════════
# Dataset Class
# ═══════════════════════════════════════════════════════════════════════

class BiasDataset(torch.utils.data.Dataset):
    """Simple dataset for bias detection fine-tuning."""

    def __init__(self, articles, tokenizer, max_length=512):
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]

        # Combine headline + body (same as pipeline_v2)
        text = article["body"]
        if article.get("headline"):
            text = article["headline"] + ". " + text

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(article["label_id"], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════
# Label-Smoothed Cross-Entropy Loss
# ═══════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing + optional class weights.

    Label smoothing helps generalization by preventing the model from
    becoming overconfident on the training data. With 300 articles,
    this is especially important to avoid overfitting.
    """

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Create smoothed labels
        true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            # Per-sample weight based on target class
            sample_weights = weight[target].unsqueeze(1)
            loss = (-true_dist * log_probs * sample_weights).sum(dim=-1)
        else:
            loss = (-true_dist * log_probs).sum(dim=-1)

        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_one_fold(fold_data, model, tokenizer, args, fold_idx=0):
    """
    Train LoRA adapter for one CV fold.
    Returns validation metrics.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 60}")
    print(f"  FOLD {fold_idx} — Device: {device}")
    print(f"{'═' * 60}")

    # ── Fresh LoRA adapter each fold ──
    # We re-apply LoRA each fold so folds are independent
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # DeBERTa-v3 attention layer names
        target_modules=["query_proj", "value_proj", "key_proj"],
        modules_to_save=["classifier"],  # Also fine-tune the classification head
    )

    # Deep copy the base model for this fold
    import copy
    fold_model = copy.deepcopy(model)
    peft_model = get_peft_model(fold_model, lora_config)
    peft_model.to(device)
    peft_model.print_trainable_parameters()

    # ── Datasets ──
    train_dataset = BiasDataset(fold_data["train"], tokenizer, max_length=args.max_length)
    val_dataset = BiasDataset(fold_data["val"], tokenizer, max_length=args.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Class weights (inverse frequency) ──
    label_counts = Counter(a["label_id"] for a in fold_data["train"])
    total = sum(label_counts.values())
    class_weights = torch.tensor([
        total / (3 * label_counts.get(i, 1)) for i in range(3)
    ], dtype=torch.float32)
    print(f"  Class weights: {class_weights.tolist()}")

    # ── Loss function ──
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        weight=class_weights,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        peft_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ── LR scheduler (linear warmup + cosine decay) ──
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(args.epochs):
        # ── Train ──
        peft_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(outputs.logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        peft_model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(f"  Epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Early stopping on macro F1 ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in peft_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best was epoch {best_epoch}, F1={best_val_f1:.3f})")
                break

    # ── Restore best model and get final metrics ──
    if best_state is not None:
        peft_model.load_state_dict(best_state)
        peft_model.to(device)

    # Final validation pass
    peft_model.eval()
    val_preds = []
    val_labels = []
    val_probs_all = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(labels.cpu().tolist())
            val_probs_all.extend(probs.cpu().tolist())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Final val accuracy: {val_acc:.3f}")
    print(f"  Final val macro F1: {val_f1:.3f}")
    print(f"\n{classification_report(val_labels, val_preds, target_names=LABELS, zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>10} {'Left':>8} {'Center':>8} {'Right':>8}")
    for i, label in enumerate(LABELS):
        row = "".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {label:>10}{row}")

    # Clean up GPU memory
    del peft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "val_preds": val_preds,
        "val_labels": val_labels,
        "per_class": precision_recall_fscore_support(
            val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# Train Final Model (on all data, after CV confirms it works)
# ═══════════════════════════════════════════════════════════════════════

def train_final_model(articles, model, tokenizer, args):
    """
    Train on ALL BASIL data and save the adapter.
    Only do this after CV results look good.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 60}")
    print(f"  TRAINING FINAL MODEL ON ALL DATA")
    print(f"  {len(articles)} articles — Device: {device}")
    print(f"{'═' * 60}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_proj", "value_proj", "key_proj"],
        modules_to_save=["classifier"],
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)
    peft_model.print_trainable_parameters()

    dataset = BiasDataset(articles, tokenizer, max_length=args.max_length)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    # Class weights
    label_counts = Counter(a["label_id"] for a in articles)
    total = sum(label_counts.values())
    class_weights = torch.tensor([
        total / (3 * label_counts.get(i, 1)) for i in range(3)
    ], dtype=torch.float32)

    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        peft_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Use the best epoch count from CV (or args.epochs)
    for epoch in range(args.final_epochs):
        peft_model.train()
        epoch_loss = 0
        correct = 0
        total_samples = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        print(f"  Epoch {epoch+1}/{args.final_epochs}  "
              f"loss={epoch_loss/total_samples:.4f}  "
              f"acc={correct/total_samples:.3f}")

    # Save the LoRA adapter
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n  ✓ LoRA adapter saved to {output_dir}/")
    print(f"    To load: model = PeftModel.from_pretrained(base_model, '{output_dir}')")

    return peft_model


# ═══════════════════════════════════════════════════════════════════════
# AllSides Data Augmentation
# ═══════════════════════════════════════════════════════════════════════

ALLSIDES_MAP = {
    "Left": "Left", "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right", "Right": "Right",
}


def load_allsides_augmentation(filepaths, max_per_class=100):
    """
    Load AllSides articles as additional training data.
    Caps per class to avoid overwhelming BASIL signal.
    """
    articles = []
    seen_urls = set()

    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = json.load(f)

        for story in data:
            for side in story.get("sides", []):
                bias_detail = side.get("bias_detail")
                body = (side.get("body") or "").strip()
                url = side.get("original_url", "")

                if not body or not bias_detail or bias_detail not in ALLSIDES_MAP:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                label = ALLSIDES_MAP[bias_detail]
                articles.append({
                    "event_id": f"allsides_{len(articles)}",
                    "source": side.get("source", "Unknown"),
                    "headline": side.get("headline", ""),
                    "body": body,
                    "true_label": label,
                    "label_id": LABEL2ID[label],
                })

    # Balance by class (cap at max_per_class)
    by_class = defaultdict(list)
    for a in articles:
        by_class[a["true_label"]].append(a)

    balanced = []
    for label, items in by_class.items():
        random.shuffle(items)
        balanced.extend(items[:max_per_class])

    random.shuffle(balanced)
    dist = Counter(a["true_label"] for a in balanced)
    print(f"AllSides augmentation: {len(balanced)} articles {dict(dist)}")
    return balanced


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune DeBERTa-large for political bias on BASIL"
    )

    # Data
    parser.add_argument("--basil-dir", default="BASIL",
                        help="Path to BASIL repository")
    parser.add_argument("--allsides", nargs="*", default=[],
                        help="AllSides JSON files for augmentation")
    parser.add_argument("--allsides-max-per-class", type=int, default=100,
                        help="Max AllSides articles per class")

    # Model
    parser.add_argument("--model-path", default="models/bias_detector",
                        help="Path to base model (local or HuggingFace)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Tokenizer path (defaults to model-path)")
    parser.add_argument("--max-length", type=int, default=512)

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (8-32 typical)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (usually 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=15,
                        help="Max epochs per fold")
    parser.add_argument("--final-epochs", type=int, default=8,
                        help="Epochs for final model (all data)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (4 fits 6GB VRAM)")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience")

    # CV config
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (set to 1 for quick test)")
    parser.add_argument("--seed", type=int, default=42)

    # Actions
    parser.add_argument("--train-final", action="store_true",
                        help="Train final model on all data (skip CV)")
    parser.add_argument("--output-dir", default="models/bias_detector_finetuned",
                        help="Where to save the LoRA adapter")
    parser.add_argument("--results-json", default="finetune_results.json",
                        help="Save CV results to JSON")

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # ── Reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load base model ──
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("  Loading base model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=3,
        problem_type="single_label_classification",
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load BASIL ──
    print(f"\n{'=' * 60}")
    print(f"  Loading BASIL dataset...")
    print(f"{'=' * 60}")

    articles = load_basil_data(args.basil_dir, label_source="source")

    if not articles:
        print("ERROR: No articles loaded from BASIL. Check the path.")
        sys.exit(1)

    # ── Optional AllSides augmentation ──
    augmentation = []
    if args.allsides:
        augmentation = load_allsides_augmentation(
            args.allsides,
            max_per_class=args.allsides_max_per_class,
        )

    # ── Train final model (skip CV) ──
    if args.train_final:
        all_data = articles + augmentation
        print(f"\n  Total training data: {len(all_data)} articles")
        train_final_model(all_data, model, tokenizer, args)
        return

    # ── Cross-validation ──
    print(f"\n{'=' * 60}")
    print(f"  {args.folds}-Fold Event-Level Cross-Validation")
    print(f"{'=' * 60}")

    folds = create_event_splits(articles, n_folds=args.folds, seed=args.seed)

    all_fold_results = []
    all_val_preds = []
    all_val_labels = []

    for fold_data in folds:
        # Add augmentation data to training set only
        fold_train = fold_data["train"] + augmentation

        fold_data_aug = {
            **fold_data,
            "train": fold_train,
        }

        result = train_one_fold(
            fold_data_aug, model, tokenizer, args,
            fold_idx=fold_data["fold"],
        )
        all_fold_results.append(result)
        all_val_preds.extend(result["val_preds"])
        all_val_labels.extend(result["val_labels"])

    # ── Aggregate CV results ──
    print(f"\n{'═' * 60}")
    print(f"  CROSS-VALIDATION SUMMARY ({args.folds} folds)")
    print(f"{'═' * 60}")

    accs = [r["val_accuracy"] for r in all_fold_results]
    f1s = [r["val_macro_f1"] for r in all_fold_results]

    print(f"\n  Per-fold results:")
    for r in all_fold_results:
        print(f"    Fold {r['fold']}: acc={r['val_accuracy']:.3f}  "
              f"macro_f1={r['val_macro_f1']:.3f}  "
              f"best_epoch={r['best_epoch']}")

    print(f"\n  Accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # Overall classification report (all folds combined)
    print(f"\n  Combined classification report:")
    print(classification_report(
        all_val_labels, all_val_preds,
        target_names=LABELS, zero_division=0,
    ))

    # Overall confusion matrix
    cm = confusion_matrix(all_val_labels, all_val_preds)
    print(f"  Combined confusion matrix:")
    print(f"  {'':>10} {'Left':>8} {'Center':>8} {'Right':>8}")
    for i, label in enumerate(LABELS):
        row = "".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {label:>10}{row}")

    # ── Save results ──
    output = {
        "n_folds": args.folds,
        "n_articles": len(articles),
        "n_augmentation": len(augmentation),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "per_fold": [{
            "fold": r["fold"],
            "accuracy": r["val_accuracy"],
            "macro_f1": r["val_macro_f1"],
            "best_epoch": r["best_epoch"],
        } for r in all_fold_results],
    }

    with open(args.results_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.results_json}")

    # ── Recommendation ──
    avg_best_epoch = np.mean([r["best_epoch"] for r in all_fold_results])
    print(f"\n{'═' * 60}")
    print(f"  NEXT STEPS")
    print(f"{'═' * 60}")
    print(f"  Average best epoch: {avg_best_epoch:.1f}")
    print(f"  → Use --final-epochs {int(avg_best_epoch)} when training final model")
    print(f"  → Run: python finetune_basil.py --basil-dir {args.basil_dir} "
          f"--train-final --final-epochs {int(avg_best_epoch)}")


if __name__ == "__main__":
    main()