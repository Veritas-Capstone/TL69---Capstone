"""
LoRA fine-tuning on BASIL
"""

import os
import sys
import json
import random
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

# config settings
config = SimpleNamespace(
    basil_dir="../../datasets/other_data/BASIL",
    allsides=[],
    allsides_max_per_class=100,
    model_path="../../models/demo_models/bias_detector",
    tokenizer_path=None,
    max_length=512,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    epochs=15,
    final_epochs=8,
    batch_size=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label_smoothing=0.1,
    patience=4,
    folds=5,
    seed=42,
    train_final=False,
    output_dir="../../models/bias_detector_finetuned",
    results_json="finetune_results.json",
)

LABELS = ["Left", "Center", "Right"]
LABEL2ID = {"Left": 0, "Center": 1, "Right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ALLSIDES_MAP = {
    "Left": "Left", "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right", "Right": "Right",
}


def load_basil_data(basil_dir, label_source="source"):
    # basil_loader.py is in experiments/ (two levels up from experiments/training/v1/)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from basil_loader import load_basil

    articles = load_basil(basil_dir, label_source=label_source)
    for a in articles:
        a["label_id"] = LABEL2ID.get(a["true_label"], 1)
        if "headline" not in a:
            a["headline"] = ""
    return articles


def create_event_splits(articles, n_folds=5, seed=42):
    # group articles by event - articles from the same event (same story, 3 outlets)
    # must stay together in either train or val to prevent data leakage
    events = defaultdict(list)
    for a in articles:
        events[a["event_id"]].append(a)

    event_ids = sorted(events.keys())
    random.seed(seed)
    random.shuffle(event_ids)

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


class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, articles, tokenizer, max_length=512):
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        text = article["body"]
        if article.get("headline"):
            text = article["headline"] + ". " + text

        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(article["label_id"], dtype=torch.long),
        }


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        if self.weight is not None:
            weight = self.weight.to(logits.device)
            sample_weights = weight[target].unsqueeze(1)
            loss = (-true_dist * log_probs * sample_weights).sum(dim=-1)
        else:
            loss = (-true_dist * log_probs).sum(dim=-1)

        return loss.mean()


def train_one_fold(fold_data, model, tokenizer, cfg, fold_idx=0):
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Fold {fold_idx} — Device: {device}")

    # fresh LoRA adapter each fold so folds are independent
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query_proj", "value_proj", "key_proj"],
        modules_to_save=["classifier"],
    )

    fold_model = copy.deepcopy(model)
    peft_model = get_peft_model(fold_model, lora_config)
    peft_model.to(device)
    peft_model.print_trainable_parameters()

    train_dataset = BiasDataset(fold_data["train"], tokenizer, max_length=cfg.max_length)
    val_dataset = BiasDataset(fold_data["val"], tokenizer, max_length=cfg.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # class weights (inverse frequency)
    label_counts = Counter(a["label_id"] for a in fold_data["train"])
    total = sum(label_counts.values())
    class_weights = torch.tensor([
        total / (3 * label_counts.get(i, 1)) for i in range(3)
    ], dtype=torch.float32)
    print(f"  Class weights: {class_weights.tolist()}")

    criterion = LabelSmoothingCrossEntropy(
        smoothing=cfg.label_smoothing, weight=class_weights,
    )
    optimizer = torch.optim.AdamW(
        peft_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # LR scheduler (linear warmup + cosine decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(cfg.epochs):
        # train
        peft_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # validate
        peft_model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(f"  Epoch {epoch+1}/{cfg.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # early stopping on macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in peft_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best was epoch {best_epoch}, F1={best_val_f1:.3f})")
                break

    # restore best model and get final metrics
    if best_state is not None:
        peft_model.load_state_dict(best_state)
        peft_model.to(device)

    peft_model.eval()
    val_preds = []
    val_labels = []
    val_probs_all = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
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

    # confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>10} {'Left':>8} {'Center':>8} {'Right':>8}")
    for i, label in enumerate(LABELS):
        row = "".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {label:>10}{row}")

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


def train_final_model(articles, model, tokenizer, cfg):
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training final model on all data")
    print(f"  {len(articles)} articles — Device: {device}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query_proj", "value_proj", "key_proj"],
        modules_to_save=["classifier"],
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)
    peft_model.print_trainable_parameters()

    dataset = BiasDataset(articles, tokenizer, max_length=cfg.max_length)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    label_counts = Counter(a["label_id"] for a in articles)
    total = sum(label_counts.values())
    class_weights = torch.tensor([
        total / (3 * label_counts.get(i, 1)) for i in range(3)
    ], dtype=torch.float32)

    criterion = LabelSmoothingCrossEntropy(
        smoothing=cfg.label_smoothing, weight=class_weights,
    )
    optimizer = torch.optim.AdamW(
        peft_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.final_epochs):
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

        print(f"  Epoch {epoch+1}/{cfg.final_epochs}  "
              f"loss={epoch_loss/total_samples:.4f}  "
              f"acc={correct/total_samples:.3f}")

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n  LoRA adapter saved to {output_dir}/")
    print(f"  To load: model = PeftModel.from_pretrained(base_model, '{output_dir}')")

    return peft_model


def load_allsides_augmentation(filepaths, max_per_class=100):
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

    # balance by class
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


if __name__ == "__main__":
    cfg = config
    if cfg.tokenizer_path is None:
        cfg.tokenizer_path = cfg.model_path

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("Loading base model..")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path, num_labels=3, problem_type="single_label_classification",
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # load BASIL
    articles = load_basil_data(cfg.basil_dir, label_source="source")
    if not articles:
        print("ERROR: No articles loaded from BASIL. Check the path.")
        sys.exit(1)

    # optional AllSides augmentation
    augmentation = []
    if cfg.allsides:
        augmentation = load_allsides_augmentation(
            cfg.allsides, max_per_class=cfg.allsides_max_per_class,
        )

    # train final model (skip CV)
    if cfg.train_final:
        all_data = articles + augmentation
        print(f"\n  Total training data: {len(all_data)} articles")
        train_final_model(all_data, model, tokenizer, cfg)
        sys.exit(0)

    # cross-validation
    print(f"\n  {cfg.folds}-Fold Event-Level Cross-Validation")

    folds = create_event_splits(articles, n_folds=cfg.folds, seed=cfg.seed)

    all_fold_results = []
    all_val_preds = []
    all_val_labels = []

    for fold_data in folds:
        fold_train = fold_data["train"] + augmentation
        fold_data_aug = {**fold_data, "train": fold_train}

        result = train_one_fold(
            fold_data_aug, model, tokenizer, cfg,
            fold_idx=fold_data["fold"],
        )
        all_fold_results.append(result)
        all_val_preds.extend(result["val_preds"])
        all_val_labels.extend(result["val_labels"])

    # aggregate CV results
    accs = [r["val_accuracy"] for r in all_fold_results]
    f1s = [r["val_macro_f1"] for r in all_fold_results]

    print(f"\n  CV Summary ({cfg.folds} folds)")
    for r in all_fold_results:
        print(f"    Fold {r['fold']}: acc={r['val_accuracy']:.3f}  "
              f"macro_f1={r['val_macro_f1']:.3f}  "
              f"best_epoch={r['best_epoch']}")

    print(f"\n  Accuracy:  {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  Macro F1:  {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    # combined classification report (all folds)
    print(f"\n  Combined classification report:")
    print(classification_report(
        all_val_labels, all_val_preds,
        target_names=LABELS, zero_division=0,
    ))

    # combined confusion matrix
    cm = confusion_matrix(all_val_labels, all_val_preds)
    print(f"  Combined confusion matrix:")
    print(f"  {'':>10} {'Left':>8} {'Center':>8} {'Right':>8}")
    for i, label in enumerate(LABELS):
        row = "".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {label:>10}{row}")

    # save results
    output = {
        "n_folds": cfg.folds,
        "n_articles": len(articles),
        "n_augmentation": len(augmentation),
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "label_smoothing": cfg.label_smoothing,
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

    with open(cfg.results_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {cfg.results_json}")

    avg_best_epoch = np.mean([r["best_epoch"] for r in all_fold_results])
    print(f"\n  Average best epoch: {avg_best_epoch:.1f}")