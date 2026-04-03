"""
LEACE source erasure (Belrose et al., NeurIPS 2023).
Erases source-outlet identity from [CLS] embeddings in closed form,
then trains MLP classifier on deconfounded representations.
"""

import json, os, random, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from torch.autograd import Function

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
MODEL_PATH = "/nfs/veritas/kazim18/TL69---Capstone/Bias-Detection/experiments/models/v3/v3_deberta_l0.7_ml512_ep10/model"  
ERASE_DIMS = None 
MLP_EPOCHS = 50
MAX_LENGTH = 256
SEED = 42


# AA model architecture (needed to load trained checkpoints)
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

class AdversarialBiasModel(nn.Module):
    def __init__(self, model_name, num_labels=3, num_domains=NUM_DOMAINS, lambda_adv=0.7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.bias_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, num_labels))
        self.gradient_reversal = GradientReversalLayer(alpha=lambda_adv)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_domains))
        self.lambda_adv = lambda_adv

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]


# manual LEACE: removes linear directions most predictive of source identity
class ManualLEACE:
    def __init__(self, erase_dims=None):
        self.erase_dims = erase_dims
        self.projection = None

    def fit(self, X, Z):
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        Z = Z.numpy() if isinstance(Z, torch.Tensor) else Z

        # class-conditional means
        classes = np.unique(Z)
        grand_mean = X.mean(axis=0)
        means = np.stack([X[Z == c].mean(axis=0) - grand_mean for c in classes if (Z == c).sum() > 0])

        # weight by class frequency
        weights = np.array([np.sqrt((Z == c).sum()) for c in classes])
        weighted_means = means * weights[:, None]

        # SVD to find source-predictive directions
        U, S, Vt = np.linalg.svd(weighted_means, full_matrices=False)

        if self.erase_dims is None:
            n_erase = max(1, (S > S.max() * 0.01).sum())
        else:
            n_erase = min(self.erase_dims, len(S))

        print(f"  LEACE: erasing {n_erase} directions (singular values: {S[:n_erase].round(2)})")

        # projection: I - V V^T removes concept directions
        V = Vt[:n_erase].T
        self.projection = np.eye(X.shape[1]) - V @ V.T
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            P = torch.tensor(self.projection, dtype=X.dtype, device=X.device)
            return X @ P.T
        return X @ self.projection.T


def get_leace_eraser(X, Z, erase_dims):
    try:
        from concept_erasure import LeaceFitter
        print("  Using concept-erasure library")
        fitter = LeaceFitter(X.shape[1], len(torch.unique(Z)), device=X.device, dtype=X.dtype)
        for start in range(0, len(X), 2048):
            fitter.update(X[start:start+2048], Z[start:start+2048])
        return fitter.eraser, "concept-erasure"
    except ImportError:
        print("  concept-erasure not installed, using manual SVD-based LEACE")
        leace = ManualLEACE(erase_dims=erase_dims)
        leace.fit(X.cpu(), Z.cpu())
        return leace, "manual"


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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


@torch.no_grad()
def extract_embeddings(encoder_model, tokenizer, texts, device, max_length,
                       batch_size=32, is_aa_model=False):
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        enc = tokenizer(batch_texts, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt").to(device)

        if is_aa_model:
            embeds = encoder_model.encode(enc["input_ids"], enc["attention_mask"])
        else:
            embeds = encoder_model(enc["input_ids"], enc["attention_mask"]).last_hidden_state[:, 0, :]

        all_embeds.append(embeds.cpu())

        if (start // batch_size) % 50 == 0:
            print(f"    {min(start+batch_size, len(texts))}/{len(texts)}...")

    return torch.cat(all_embeds, dim=0)


def train():
    config_name = f"leace_dims{ERASE_DIMS}"
    if MODEL_PATH:
        aa_name = os.path.basename(os.path.dirname(MODEL_PATH))
        config_name += f"_from_{aa_name}"
    else:
        config_name += "_fromBase"
    if MAX_LENGTH != 256:
        config_name += f"_ml{MAX_LENGTH}"

    output_dir  = os.path.join(BASE_OUTPUT, "leace", config_name, "model")
    results_dir = os.path.join(BASE_OUTPUT, "leace", config_name, "results")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  LEACE: {config_name}, erase_dims={ERASE_DIMS}, device={device}")
    print(f"  Model: {MODEL_PATH or 'base DeBERTa'}")

    (train_t, train_b, train_d, val_t, val_b, val_d) = load_datasets(seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # load encoder (either AA model or base DeBERTa)
    is_aa = False
    if MODEL_PATH and os.path.exists(os.path.join(MODEL_PATH, "model.pt")):
        print(f"\n  Loading AA model from {MODEL_PATH}")
        cfg_path = os.path.join(MODEL_PATH, "config.json")
        lam = 0.7
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                lam = json.load(f).get("lambda_adv", 0.7)
        encoder = AdversarialBiasModel(BASE_MODEL, lambda_adv=lam).to(device)
        encoder.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt"), map_location=device))
        is_aa = True
    else:
        print(f"\n  Loading base DeBERTa encoder")
        encoder = AutoModel.from_pretrained(BASE_MODEL).to(device)
    encoder.eval()

    # step 1: extract embeddings
    print(f"\n  Extracting train embeddings...")
    train_embeds = extract_embeddings(encoder, tokenizer, train_t, device, MAX_LENGTH, is_aa_model=is_aa)
    print(f"  Extracting val embeddings...")
    val_embeds = extract_embeddings(encoder, tokenizer, val_t, device, MAX_LENGTH, is_aa_model=is_aa)

    train_labels = torch.tensor(train_b, dtype=torch.long)
    train_domains = torch.tensor(train_d, dtype=torch.long)
    val_labels = torch.tensor(val_b, dtype=torch.long)
    embed_dim = train_embeds.shape[1]
    print(f"  Embedding dim: {embed_dim}")

    # source predictability before erasure
    from sklearn.linear_model import LogisticRegression
    clf_before = LogisticRegression(max_iter=500, random_state=42)
    clf_before.fit(train_embeds.numpy(), train_domains.numpy())
    source_acc_before = clf_before.score(train_embeds.numpy(), train_domains.numpy())
    print(f"\n  Source predictability before erasure: {source_acc_before:.1%}")

    # step 2: fit LEACE and erase
    print(f"\n  Fitting LEACE eraser...")
    eraser, method = get_leace_eraser(train_embeds.to(device), train_domains.to(device), ERASE_DIMS)

    if method == "concept-erasure":
        train_erased = eraser(train_embeds.to(device)).cpu()
        val_erased = eraser(val_embeds.to(device)).cpu()
    else:
        train_erased = torch.tensor(eraser.transform(train_embeds), dtype=torch.float32)
        val_erased = torch.tensor(eraser.transform(val_embeds), dtype=torch.float32)

    clf_after = LogisticRegression(max_iter=500, random_state=42)
    clf_after.fit(train_erased.numpy(), train_domains.numpy())
    source_acc_after = clf_after.score(train_erased.numpy(), train_domains.numpy())
    print(f"  Source predictability after erasure: {source_acc_after:.1%}")
    print(f"  Source info removed: {source_acc_before - source_acc_after:.1%}")

    # step 3: train MLP on erased embeddings
    print(f"\n  Training MLP classifier...")
    classifier = MLPClassifier(embed_dim, num_classes=3, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.01)

    train_loader = DataLoader(TensorDataset(train_erased, train_labels), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_erased, val_labels), batch_size=512)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_f1 = 0

    for epoch in range(MLP_EPOCHS):
        classifier.train()
        total_loss, steps = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = ce_fn(classifier(X_batch), y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); steps += 1

        classifier.eval()
        vp, vt = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                vp.extend(torch.argmax(classifier(X_batch.to(device)), dim=-1).cpu().tolist())
                vt.extend(y_batch.tolist())

        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        vacc = accuracy_score(vt, vp)
        print(f"  Epoch {epoch+1}/{MLP_EPOCHS}: loss={total_loss/steps:.4f}, "
              f"val_acc={vacc:.3f}, val_f1={vf1:.3f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
            if method == "manual":
                np.save(os.path.join(output_dir, "leace_projection.npy"), eraser.projection)
            print(f"    Saved (F1={best_f1:.3f})")

    # step 4: evaluate on AllSides OOD
    print(f"\n  Evaluating on AllSides OOD...")
    classifier.load_state_dict(torch.load(os.path.join(output_dir, "classifier.pt"), map_location=device))
    classifier.eval()

    articles = load_allsides()
    ood_texts = [a["text"] for a in articles]
    ood_truths = [a["true_label"] for a in articles]

    ood_embeds = extract_embeddings(encoder, tokenizer, ood_texts, device, MAX_LENGTH, is_aa_model=is_aa)
    if method == "concept-erasure":
        ood_erased = eraser(ood_embeds.to(device)).cpu()
    else:
        ood_erased = torch.tensor(eraser.transform(ood_embeds), dtype=torch.float32)

    with torch.no_grad():
        preds = torch.argmax(classifier(ood_erased.to(device)), dim=-1).cpu().tolist()

    preds_labels = [ID2LABEL[p] for p in preds]
    acc = accuracy_score(ood_truths, preds_labels)
    f1 = f1_score(ood_truths, preds_labels, average="macro", zero_division=0, labels=LABELS)

    print(f"\n  [{config_name}] AllSides OOD: acc={acc:.1%}, F1={f1:.3f}")
    print(classification_report(ood_truths, preds_labels, labels=LABELS, zero_division=0))

    cm = confusion_matrix(ood_truths, preds_labels, labels=LABELS)
    print(f"  Confusion Matrix:")
    print(f"  {'':>10} {'left':>8} {'center':>8} {'right':>8}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    results = {
        "config": config_name,
        "source_acc_before_erasure": float(source_acc_before),
        "source_acc_after_erasure": float(source_acc_after),
        "ood_accuracy": float(acc),
        "ood_macro_f1": float(f1),
        "best_val_f1": float(best_f1),
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()