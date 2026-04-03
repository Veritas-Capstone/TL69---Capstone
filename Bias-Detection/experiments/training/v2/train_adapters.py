"""
Adapter training pipeline for political bias detection.
Step 1: Train individual bottleneck adapters per dataset
Step 2: Combine with AdapterFusion
Step 3: Evaluate on AllSides
"""

import json, os, time, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_MODEL = "microsoft/deberta-v3-large"
PARQUET_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets"
ALLSIDES_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets/other_data"
BASIL_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/datasets/other_data/BASIL"
ADAPTER_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/adapter/trained_adapters"
FUSION_DIR = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/adapter/trained_fusion"

LABELS = ["left", "center", "right"]
LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {0: "left", 1: "center", 2: "right"}
ALLSIDES_MAP = {"Left": "left", "Lean Left": "left", "Center": "center",
                "Lean Right": "right", "Right": "right"}
BASIL_SOURCE_MAP = {"hpo": "left", "nyt": "center", "fox": "right"}

PARQUET_DATASETS = [
    "article_bias_prediction", "dem_rep_party_platform_topics",
    "gpt4_political_bias", "gpt4_political_ideologies",
    "political_tweets", "qbias", "webis_bias_flipper_18", "webis_news_bias_20",
]
BASIL_ADAPTER_NAME = "basil"
MAX_LENGTH = 256

# config settings
STEP = "all" #adapters, fusion, eval, all
SAMPLE_SIZE = 10000
FUSION_SAMPLE_SIZE = 5000
EPOCHS = 5
FUSION_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
FUSION_LR = 5e-5

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
def load_parquet_dataset(filepath, sample_size=None):
    df = pd.read_parquet(filepath)
    if "title" in df.columns:
        df["text"] = df.apply(lambda r: (str(r["title"]) + "\n\n" + str(r["body"]))
            if pd.notna(r.get("title")) else str(r["body"]), axis=1)
    else: df["text"] = df["body"].astype(str)
    df["label"] = df["leaning"].map(LABEL2ID)
    df = df.dropna(subset=["label"]); df["label"] = df["label"].astype(int)
    if sample_size and len(df) > sample_size:
        per_class = sample_size // len(df["label"].unique())
        sampled = [df[df["label"] == c].sample(n=min(per_class, len(df[df["label"] == c])), random_state=42)
                   for c in df["label"].unique()]
        df = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    return df["text"].tolist(), df["label"].tolist()


def load_basil_dataset(basil_dir, sample_size=None):
    basil_path = Path(basil_dir)
    articles_dir = basil_path / "articles" if (basil_path / "articles").is_dir() else basil_path
    texts, labels = [], []
    for year_dir in sorted(articles_dir.iterdir()):
        if not year_dir.is_dir(): continue
        for json_file in sorted(year_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f: data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError): continue
            if not isinstance(data, dict): continue
            source = data.get("source", "").lower()
            if source not in BASIL_SOURCE_MAP: continue
            body_parts = []
            for para in data.get("body-paragraphs", []):
                if isinstance(para, list):
                    body_parts.extend([s for s in para if isinstance(s, str) and s.strip()])
                elif isinstance(para, str) and para.strip():
                    body_parts.append(para.strip())
            title = data.get("title", "")
            if title: body_parts.insert(0, title)
            body = " ".join(body_parts)
            if len(body.split()) < 20: continue
            texts.append(body); labels.append(LABEL2ID[BASIL_SOURCE_MAP[source]])
    print(f"  BASIL: {len(texts)} articles, {dict(Counter(labels))}")
    return texts, labels


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

# train each adapter
def train_individual_adapters(sample_size=SAMPLE_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    import adapters
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    print(f"  Training individual adapters (sample={sample_size}, epochs={epochs})")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    adapters.init(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # all datasets including BASIL
    all_datasets = []
    for name in PARQUET_DATASETS:
        path = os.path.join(PARQUET_DIR, f"{name}.parquet")
        if os.path.exists(path): all_datasets.append(("parquet", name, path))
    if os.path.exists(BASIL_DIR): all_datasets.append(("basil", BASIL_ADAPTER_NAME, BASIL_DIR))

    print(f"  {len(all_datasets)} adapters to train")
    results = {}

    for data_type, dataset_name, data_path in all_datasets:
        save_path = os.path.join(ADAPTER_DIR, dataset_name)
        if os.path.exists(save_path):
            print(f"\n  SKIP: {dataset_name} (already exists)"); continue

        # load data
        if data_type == "parquet":
            texts, labels = load_parquet_dataset(data_path, sample_size=sample_size)
        else:
            texts, labels = load_basil_dataset(data_path, sample_size=sample_size)
        if not texts:
            print(f" No data loaded for {dataset_name}")
            continue

        print(f"\n  {dataset_name}: {len(texts)} samples")

        # split 90/10
        split_idx = int(len(texts) * 0.9)
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        train_loader = DataLoader(PoliticalTextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(PoliticalTextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH),
                                batch_size=batch_size)

        # add adapter
        adapter_config = adapters.BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
        model.add_adapter(dataset_name, config=adapter_config)
        model.add_classification_head(dataset_name, num_labels=3, id2label=ID2LABEL)
        model.train_adapter(dataset_name); model.set_active_adapters(dataset_name); model.to(device)

        adapter_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

        best_f1, best_epoch = 0, 0

        for epoch in range(epochs):
            model.train(); total_loss, steps = 0, 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss; loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += loss.item(); steps += 1

            # validation
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
                model.save_adapter(os.path.join(ADAPTER_DIR, dataset_name), dataset_name)
                model.save_head(os.path.join(ADAPTER_DIR, f"{dataset_name}_head"), dataset_name)

        print(f"  Best: epoch {best_epoch}, F1={best_f1:.3f}")
        results[dataset_name] = {"best_f1": float(best_f1), "best_epoch": best_epoch, "train_size": len(train_texts)}
        
        # cleanup for next adapter
        model.delete_adapter(dataset_name)
        try: model.delete_head(dataset_name)
        except: pass
        torch.cuda.empty_cache()

    # summary
    print(f"\n  Adapter summary:")
    for name, r in results.items():
        print(f"    {name:<35} F1={r['best_f1']:.3f}  epoch={r['best_epoch']}  n={r['train_size']}")
    with open(os.path.join(ADAPTER_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

# adapter fusion layer
def train_adapter_fusion(sample_size=FUSION_SAMPLE_SIZE, epochs=FUSION_EPOCHS, batch_size=BATCH_SIZE, lr=FUSION_LR):
    import adapters
    from adapters import AutoAdapterModel
    from adapters.composition import Fuse
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    os.makedirs(FUSION_DIR, exist_ok=True)
    print(f"\n  Training AdapterFusion")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL); adapters.init(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_adapter_names = PARQUET_DATASETS + [BASIL_ADAPTER_NAME]
    loaded = []
    for name in all_adapter_names:
        path = os.path.join(ADAPTER_DIR, name)
        if os.path.exists(path):
            model.load_adapter(path, load_as=name); loaded.append(name)
            print(f"  Loaded: {name}")

    if len(loaded) < 2: print("  Need at least 2 adapters"); return

    adapter_setup = Fuse(*loaded)
    model.add_adapter_fusion(adapter_setup)
    model.add_classification_head("fusion", num_labels=3, id2label=ID2LABEL)
    model.train_adapter_fusion(adapter_setup); model.set_active_adapters(adapter_setup); model.to(device)

    # build combined training data from all adapter datasets
    all_texts, all_labels = [], []
    for name in loaded:
        if name == BASIL_ADAPTER_NAME:
            texts, labels = load_basil_dataset(BASIL_DIR, sample_size=sample_size)
        else:
            path = os.path.join(PARQUET_DIR, f"{name}.parquet")
            if not os.path.exists(path): continue
            texts, labels = load_parquet_dataset(path, sample_size=sample_size)
        all_texts.extend(texts); all_labels.extend(labels)

    combined = list(zip(all_texts, all_labels))
    np.random.seed(42); np.random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    all_texts, all_labels = list(all_texts), list(all_labels)
    print(f"  Fusion data: {len(all_texts)} samples")

    split_idx = int(len(all_texts) * 0.9)
    train_loader = DataLoader(PoliticalTextDataset(all_texts[:split_idx], all_labels[:split_idx], tokenizer, MAX_LENGTH),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PoliticalTextDataset(all_texts[split_idx:], all_labels[split_idx:], tokenizer, MAX_LENGTH),
                            batch_size=batch_size)

    fusion_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    for epoch in range(epochs):
        model.train(); total_loss, steps = 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); steps += 1

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
            best_f1 = val_f1
            model.save_adapter_fusion(FUSION_DIR, ",".join(loaded))
            model.save_head(os.path.join(FUSION_DIR, "fusion_head"), "fusion")
            print(f"    Saved (F1={best_f1:.3f})")

    print(f"\n  Best fusion F1: {best_f1:.3f}")
    return best_f1

# OOD evaluation
def evaluate_on_allsides():
    import adapters
    from adapters import AutoAdapterModel
    from adapters.composition import Fuse
    from transformers import AutoTokenizer

    print(f"\n  Evaluating on AllSides")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL); adapters.init(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_adapter_names = PARQUET_DATASETS + [BASIL_ADAPTER_NAME]
    loaded = []
    for name in all_adapter_names:
        path = os.path.join(ADAPTER_DIR, name)
        if os.path.exists(path): model.load_adapter(path, load_as=name); loaded.append(name)
    print(f"  Loaded {len(loaded)} adapters")

    fusion_key = ",".join(loaded)
    model.load_adapter_fusion(FUSION_DIR, fusion_key)
    model.load_head(os.path.join(FUSION_DIR, "fusion_head"), "fusion")
    model.set_active_adapters(Fuse(*loaded)); model.to(device); model.eval()

    articles = load_allsides_data(ALLSIDES_DIR)
    print(f"  {len(articles)} articles")

    preds, truths = [], []
    for article in articles:
        inputs = tokenizer(article["text"], truncation=True, max_length=MAX_LENGTH,
                          padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            pred = ID2LABEL[torch.argmax(model(**inputs).logits, dim=-1).item()]
        preds.append(pred); truths.append(article["true_label"])

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro", zero_division=0, labels=LABELS)
    print(f"\n  Adapter Fusion AllSides OOD: acc={acc:.1%}, F1={f1:.3f}")
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

    with open(os.path.join(FUSION_DIR, "allsides_eval_results.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(f1)}, f, indent=2)


if __name__ == "__main__":
    start = time.time()
    if STEP in ("adapters", "all"): train_individual_adapters()
    if STEP in ("fusion", "all"): train_adapter_fusion()
    if STEP in ("eval", "all"): evaluate_on_allsides()
    print(f"\n  Total: {(time.time() - start) / 60:.1f} minutes")