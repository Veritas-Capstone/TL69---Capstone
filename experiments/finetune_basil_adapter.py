#!/usr/bin/env python3
"""
Bottleneck Adapter Fine-Tuning on BASIL for Political Bias Detection
=====================================================================
Replaces the LoRA-based finetune_basil.py with a Pfeiffer bottleneck adapter
so the trained weights can be stacked with an AllSides domain adapter later.

Key difference from LoRA version:
  - Saves a SEPARABLE adapter (not merged weights) that can be composed
  - Uses adapters library (AdapterHub) instead of peft
  - adapter stacking at inference: base → BASIL adapter → AllSides adapter

Usage:
  pip install adapters

  # 5-fold CV
  python finetune_basil_adapter.py --basil-dir BASIL

  # Train final adapter on all 300 articles
  python finetune_basil_adapter.py --basil-dir BASIL --train-final

  # With AllSides augmentation
  python finetune_basil_adapter.py --basil-dir BASIL --allsides allsides_data.json
"""

import os, sys, json, argparse, random, warnings, copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

import adapters
import adapters.composition as ac
from adapters import AutoAdapterModel, SeqBnConfig

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Keep all your existing constants, loaders, dataset, loss class ──
# (LABELS, LABEL2ID, ID2LABEL, SOURCE_LABEL_MAP, load_basil_data,
#  create_event_splits, BiasDataset, LabelSmoothingCrossEntropy,
#  load_allsides_augmentation are 100% unchanged — copy them here)
LABELS    = ["Left", "Center", "Right"]
LABEL2ID  = {"Left": 0, "Center": 1, "Right": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

SOURCE_LABEL_MAP = {
    "hpo": "Left",
    "nyt": "Center",
    "fox": "Right",
}

ALLSIDES_MAP = {
    "Left": "Left", "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right", "Right": "Right",
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


ADAPTER_NAME = "basil_bias"   # name used throughout for saving/loading


# ═══════════════════════════════════════════════════════════════════════
# Model Loading  ← CHANGED: use AutoAdapterModel instead of
#                   AutoModelForSequenceClassification
# ═══════════════════════════════════════════════════════════════════════

def load_base_model(model_path, tokenizer_path=None):
    from transformers import AutoTokenizer

    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # AutoAdapterModel adds adapter support on top of any HuggingFace model
    model = AutoAdapterModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Training Loop  ← CHANGED: adapter setup replaces LoRA setup
# ═══════════════════════════════════════════════════════════════════════

def train_one_fold(fold_data, base_model, tokenizer, args, fold_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}\n  FOLD {fold_idx} — Device: {device}\n{'═'*60}")

    # ── Deep copy base model for this fold ──
    fold_model = copy.deepcopy(base_model)

    # ── Add Pfeiffer bottleneck adapter ──
    # reduction_factor=16 means bottleneck = hidden_size / 16
    # For DeBERTa-large (hidden=1024): bottleneck = 64 dims (~0.5% params)
    adapter_config = SeqBnConfig(
        reduction_factor=args.reduction_factor,
        non_linearity="gelu",
    )
    fold_model.add_adapter(ADAPTER_NAME, config=adapter_config)

    # Add the classification head (3 labels) tied to this adapter
    fold_model.add_classification_head(
        ADAPTER_NAME,
        num_labels=3,
        id2label={0: "Left", 1: "Center", 2: "Right"},
    )

    # ── train_adapter() freezes ALL base weights, only adapter + head train ──
    fold_model.train_adapter(ADAPTER_NAME)
    fold_model.set_active_adapters(ADAPTER_NAME)
    fold_model.to(device)

    # Print trainable param count
    trainable = sum(p.numel() for p in fold_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in fold_model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Datasets & loaders (unchanged from your existing code) ──
    train_dataset = BiasDataset(fold_data["train"], tokenizer, max_length=args.max_length)
    val_dataset   = BiasDataset(fold_data["val"],   tokenizer, max_length=args.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    # ── Class weights ──
    label_counts = Counter(a["label_id"] for a in fold_data["train"])
    total_n = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_n / (3 * label_counts.get(i, 1)) for i in range(3)],
        dtype=torch.float32)
    print(f"  Class weights: {class_weights.tolist()}")

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fold_model.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop (identical logic to your existing code) ──
    best_val_f1, best_epoch, patience_counter, best_state = 0, 0, 0, None

    for epoch in range(args.epochs):
        fold_model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss    += loss.item() * labels.size(0)
            train_correct += (torch.argmax(outputs.logits, -1) == labels).sum().item()
            train_total   += labels.size(0)

        # ── Validation ──
        fold_model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = fold_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device))
                val_preds.extend(torch.argmax(out.logits, -1).cpu().tolist())
                val_labels_list.extend(batch["labels"].tolist())

        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1  = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)

        print(f"  Epoch {epoch+1}/{args.epochs}  "
              f"loss={train_loss/train_total:.4f}  acc={train_correct/train_total:.3f}  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, patience_counter = val_f1, epoch + 1, 0
            best_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (best={best_epoch}, F1={best_val_f1:.3f})")
                break

    if best_state:
        fold_model.load_state_dict(best_state)
        fold_model.to(device)

    # ── Final eval (same as your existing code) ──
    fold_model.eval()
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = fold_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device))
            val_preds.extend(torch.argmax(out.logits, -1).cpu().tolist())
            val_labels_list.extend(batch["labels"].tolist())

    val_acc = accuracy_score(val_labels_list, val_preds)
    val_f1  = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)
    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Final val accuracy: {val_acc:.3f}")
    print(f"  Final val macro F1: {val_f1:.3f}")
    print(f"\n{classification_report(val_labels_list, val_preds, target_names=LABELS, zero_division=0)}")

    del fold_model
    torch.cuda.empty_cache()

    return {
        "fold": fold_idx, "best_epoch": best_epoch,
        "val_accuracy": val_acc, "val_macro_f1": val_f1,
        "val_preds": val_preds, "val_labels": val_labels_list,
    }


# ═══════════════════════════════════════════════════════════════════════
# Train Final Adapter  ← CHANGED: saves adapter file, not merged weights
# ═══════════════════════════════════════════════════════════════════════

def train_final_adapter(articles, base_model, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}\n  TRAINING FINAL BASIL ADAPTER — {len(articles)} articles\n{'═'*60}")

    model = copy.deepcopy(base_model)

    adapter_config = SeqBnConfig(
        reduction_factor=args.reduction_factor,
        non_linearity="gelu",
    )
    model.add_adapter(ADAPTER_NAME, config=adapter_config)
    model.add_classification_head(
        ADAPTER_NAME, num_labels=3,
        id2label={0: "Left", 1: "Center", 2: "Right"})
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.to(device)

    dataset = BiasDataset(articles, tokenizer, max_length=args.max_length)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    label_counts = Counter(a["label_id"] for a in articles)
    total_n = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_n / (3 * label_counts.get(i, 1)) for i in range(3)], dtype=torch.float32)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(args.final_epochs):
        model.train()
        epoch_loss, correct, total_s = 0, 0, 0
        for batch in loader:
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
        print(f"  Epoch {epoch+1}/{args.final_epochs}  "
              f"loss={epoch_loss/total_s:.4f}  acc={correct/total_s:.3f}")

    # ── Save ONLY the adapter (not the full model) ──
    # This is the key difference from LoRA: the adapter is a small separable file
    # (~8MB) that can be loaded on top of the base model and stacked later.
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_adapter(args.output_dir, ADAPTER_NAME)         # adapter weights only
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n  ✓ BASIL adapter saved to {args.output_dir}/")
    print(f"    Size: ~{sum(os.path.getsize(os.path.join(args.output_dir, f)) for f in os.listdir(args.output_dir)) / 1e6:.1f} MB")
    print(f"\n  Next step: train AllSides adapter on top →")
    print(f"    python finetune_allsides_adapter.py \\")
    print(f"      --base-model {args.model_path} \\")
    print(f"      --basil-adapter {args.output_dir} \\")
    print(f"      --allsides ../../AllSides/allsides_data.json ../../AllSides/allsides_data_2_20.json")

    return model


# ═══════════════════════════════════════════════════════════════════════
# Main  ← CHANGED: --lora-r/alpha replaced with --reduction-factor,
#                   model loaded via AutoAdapterModel
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basil-dir",     default="BASIL")
    parser.add_argument("--allsides",      nargs="*", default=[])
    parser.add_argument("--allsides-max-per-class", type=int, default=100)
    parser.add_argument("--model-path",    default="models/bias_detector")
    parser.add_argument("--tokenizer-path",default=None)
    parser.add_argument("--max-length",    type=int, default=512)

    # ← NEW: bottleneck reduction factor instead of LoRA rank
    parser.add_argument("--reduction-factor", type=int, default=16,
                        help="Bottleneck size = hidden_dim / reduction_factor. "
                             "16 → 64 dims for DeBERTa-large (~0.5% params)")

    # Training (same as before)
    parser.add_argument("--epochs",        type=int,   default=15)
    parser.add_argument("--final-epochs",  type=int,   default=8)
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--warmup-ratio",  type=float, default=0.1)
    parser.add_argument("--label-smoothing",type=float,default=0.1)
    parser.add_argument("--patience",      type=int,   default=4)
    parser.add_argument("--folds",         type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)

    # Actions
    parser.add_argument("--train-final",   action="store_true")
    parser.add_argument("--output-dir",    default="models/basil_adapter")
    parser.add_argument("--results-json",  default="finetune_adapter_results.json")

    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("="*60 + "\n  Loading base model (AutoAdapterModel)...\n" + "="*60)
    model, tokenizer = load_base_model(args.model_path, args.tokenizer_path)

    print(f"\n{'='*60}\n  Loading BASIL...\n{'='*60}")
    articles = load_basil_data(args.basil_dir)
    if not articles:
        sys.exit("ERROR: No articles loaded from BASIL.")

    augmentation = []
    if args.allsides:
        augmentation = load_allsides_augmentation(args.allsides, args.allsides_max_per_class)

    if args.train_final:
        train_final_adapter(articles + augmentation, model, tokenizer, args)
        return

    # ── CV (same flow as your existing script) ──
    print(f"\n{'='*60}\n  {args.folds}-Fold Event-Level CV\n{'='*60}")
    folds = create_event_splits(articles, n_folds=args.folds, seed=args.seed)

    all_results, all_preds, all_labels = [], [], []
    for fold in folds:
        fold["train"] = fold["train"] + augmentation
        result = train_one_fold(fold, model, tokenizer, args, fold_idx=fold["fold"])
        all_results.append(result)
        all_preds.extend(result["val_preds"])
        all_labels.extend(result["val_labels"])

    # ── Summary (identical to your existing code) ──
    accs = [r["val_accuracy"] for r in all_results]
    f1s  = [r["val_macro_f1"]  for r in all_results]
    print(f"\n{'═'*60}\n  CV SUMMARY\n{'═'*60}")
    for r in all_results:
        print(f"  Fold {r['fold']}: acc={r['val_accuracy']:.3f}  f1={r['val_macro_f1']:.3f}  best_epoch={r['best_epoch']}")
    print(f"\n  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0)}")

    avg_best = int(np.mean([r['best_epoch'] for r in all_results]))
    print(f"\n  → Recommended --final-epochs {avg_best}")

if __name__ == "__main__":
    main()
