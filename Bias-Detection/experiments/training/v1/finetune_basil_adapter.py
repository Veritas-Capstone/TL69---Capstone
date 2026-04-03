"""
Bottleneck adapter fine-tuning on BASIL for bias detection
"""

import os, sys, json, random, warnings, copy
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

import adapters
import adapters.composition as ac
from adapters import AutoAdapterModel, SeqBnConfig

warnings.filterwarnings("ignore", category=FutureWarning)

# config settings
config = SimpleNamespace(
    basil_dir="../../datasets/other_data/BASIL",
    allsides=[],
    allsides_max_per_class=100,
    model_path="../../models/demo_models/bias_detector",
    tokenizer_path=None,
    max_length=512,
    reduction_factor=16, 
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
    output_dir="../../models/basil_adapter",
    results_json="finetune_adapter_results.json",
)

LABELS = ["Left", "Center", "Right"]
LABEL2ID = {"Left": 0, "Center": 1, "Right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ALLSIDES_MAP = {
    "Left": "Left", "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right", "Right": "Right",
}

ADAPTER_NAME = "basil_bias"


def load_basil_data(basil_dir, label_source="source"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from basil_loader import load_basil

    articles = load_basil(basil_dir, label_source=label_source)
    for a in articles:
        a["label_id"] = LABEL2ID.get(a["true_label"], 1)
        if "headline" not in a:
            a["headline"] = ""
    return articles


def create_event_splits(articles, n_folds=5, seed=42):
    # articles from the same event must stay together to prevent data leakage
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
            "fold": i, "train": train_articles, "val": val_articles,
            "n_train_events": len(train_events), "n_val_events": len(val_events),
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


def load_base_model(model_path, tokenizer_path=None):
    from transformers import AutoTokenizer

    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # AutoAdapterModel adds adapter support on top of any HuggingFace model
    model = AutoAdapterModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def train_one_fold(fold_data, base_model, tokenizer, cfg, fold_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Fold {fold_idx} - Device: {device}")

    fold_model = copy.deepcopy(base_model)

    # add Pfeiffer bottleneck adapter
    adapter_config = SeqBnConfig(
        reduction_factor=cfg.reduction_factor, non_linearity="gelu",
    )
    fold_model.add_adapter(ADAPTER_NAME, config=adapter_config)

    # classification head tied to this adapter
    fold_model.add_classification_head(
        ADAPTER_NAME, num_labels=3,
        id2label={0: "Left", 1: "Center", 2: "Right"},
    )

    # train_adapter() freezes ALL base weights, only adapter + head train
    fold_model.train_adapter(ADAPTER_NAME)
    fold_model.set_active_adapters(ADAPTER_NAME)
    fold_model.to(device)

    trainable = sum(p.numel() for p in fold_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in fold_model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    train_dataset = BiasDataset(fold_data["train"], tokenizer, max_length=cfg.max_length)
    val_dataset   = BiasDataset(fold_data["val"],   tokenizer, max_length=cfg.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)

    # class weights
    label_counts = Counter(a["label_id"] for a in fold_data["train"])
    total_n = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_n / (3 * label_counts.get(i, 1)) for i in range(3)],
        dtype=torch.float32)
    print(f"  Class weights: {class_weights.tolist()}")

    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing, weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fold_model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # training loop
    best_val_f1, best_epoch, patience_counter, best_state = 0, 0, 0, None

    for epoch in range(cfg.epochs):
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

        # validation
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

        print(f"  Epoch {epoch+1}/{cfg.epochs}  "
              f"loss={train_loss/train_total:.4f}  acc={train_correct/train_total:.3f}  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, patience_counter = val_f1, epoch + 1, 0
            best_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1} (best={best_epoch}, F1={best_val_f1:.3f})")
                break

    if best_state:
        fold_model.load_state_dict(best_state)
        fold_model.to(device)

    # final eval
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


def train_final_adapter(articles, base_model, tokenizer, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training final BASIL adapter - {len(articles)} articles")

    model = copy.deepcopy(base_model)

    adapter_config = SeqBnConfig(
        reduction_factor=cfg.reduction_factor, non_linearity="gelu",
    )
    model.add_adapter(ADAPTER_NAME, config=adapter_config)
    model.add_classification_head(
        ADAPTER_NAME, num_labels=3,
        id2label={0: "Left", 1: "Center", 2: "Right"})
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.to(device)

    dataset = BiasDataset(articles, tokenizer, max_length=cfg.max_length)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    label_counts = Counter(a["label_id"] for a in articles)
    total_n = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_n / (3 * label_counts.get(i, 1)) for i in range(3)], dtype=torch.float32)

    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing, weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.final_epochs):
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
        print(f"  Epoch {epoch+1}/{cfg.final_epochs}  "
              f"loss={epoch_loss/total_s:.4f}  acc={correct/total_s:.3f}")

    # save only the adapter
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_adapter(cfg.output_dir, ADAPTER_NAME)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"\n  BASIL adapter saved to {cfg.output_dir}/")
    print(f"    Size: {sum(os.path.getsize(os.path.join(cfg.output_dir, f)) for f in os.listdir(cfg.output_dir)) / 1e6:.1f} MB")

    return model


if __name__ == "__main__":
    cfg = config
    if cfg.tokenizer_path is None:
        cfg.tokenizer_path = cfg.model_path

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    print("Loading base model (AutoAdapterModel)..")
    model, tokenizer = load_base_model(cfg.model_path, cfg.tokenizer_path)

    print("\nLoading BASIL..")
    articles = load_basil_data(cfg.basil_dir)
    if not articles:
        sys.exit("ERROR: No articles loaded from BASIL.")

    augmentation = []
    if cfg.allsides:
        augmentation = load_allsides_augmentation(cfg.allsides, cfg.allsides_max_per_class)

    if cfg.train_final:
        train_final_adapter(articles + augmentation, model, tokenizer, cfg)
    else:
        # cross-validation
        print(f"\n  {cfg.folds}-Fold Event-Level CV")
        folds = create_event_splits(articles, n_folds=cfg.folds, seed=cfg.seed)

        all_results, all_preds, all_labels = [], [], []
        for fold in folds:
            fold["train"] = fold["train"] + augmentation
            result = train_one_fold(fold, model, tokenizer, cfg, fold_idx=fold["fold"])
            all_results.append(result)
            all_preds.extend(result["val_preds"])
            all_labels.extend(result["val_labels"])

        # summary
        accs = [r["val_accuracy"] for r in all_results]
        f1s  = [r["val_macro_f1"]  for r in all_results]
        print(f"\n  CV Summary")
        for r in all_results:
            print(f"  Fold {r['fold']}: acc={r['val_accuracy']:.3f}  f1={r['val_macro_f1']:.3f}  best_epoch={r['best_epoch']}")
        print(f"\n  Accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        print(f"  Macro F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        print(f"\n{classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0)}")

        avg_best = int(np.mean([r['best_epoch'] for r in all_results]))
        print(f"\n  rec final_epochs={avg_best}")