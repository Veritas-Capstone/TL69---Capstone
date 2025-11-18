# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv (3.12.9)
#     language: python
#     name: python3
# ---

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from torch.utils.data import Subset
import os
from datetime import datetime


MODEL = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
DATA_SET = "fever_train_claims"

DATA_PATH = f"../data/processed/{DATA_SET}.csv"
LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}

# %%
class PairwiseExpansionDataset(Dataset):
    """
    Expands a CSV row into multiple pairs:
      (claim, evidence_sentence, label_id)
    Better for initially training and fine-tuning the model.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        pairs = []
        
        for _, row in df.iterrows():
            claim = str(row["claim"]).strip()
            label_id = LABEL_MAP[row["label"].upper()]

            # Parse JSON list from CSV
            evid_list = json.loads(row["evidence"])

            # Only keep string evidence
            evid_list = [ev.strip() for ev in evid_list if isinstance(ev, str) and ev.strip()]

            for ev in evid_list:
                pairs.append((claim, ev, label_id))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_pairwise(batch, tokenizer, max_length=256):
    """
    Collate for PairwiseExpansionDataset.
    batch: list of (claim, evidence_sent, label_id)
    Returns tokenized tensors and labels.
    """
    claims, evids, labels = zip(*batch)
    enc = tokenizer(
        list(evids), list(claims),
        padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return enc, labels


# %%
# Training the roberta model on the dataset with pairwise expansion
if __name__ == "__main__":
    data_path = DATA_PATH
    pair_ds = PairwiseExpansionDataset(data_path)
    print(f"Total pairwise examples: {len(pair_ds)}")

    # simple random split; good enough for a demo
    indices = np.arange(len(pair_ds))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=10)

    train_ds = Subset(pair_ds, train_idx)
    val_ds   = Subset(pair_ds, val_idx)

    BATCH_SIZE = 16
    MAX_LENGTH = 256
    THRESHOLD = 0.05  # for early stopping
    NUM_EPOCHS = 4
    PATIENCE = 2
    LR = 2e-5

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_pairwise(batch, tokenizer, max_length=MAX_LENGTH),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_pairwise(batch, tokenizer, max_length=MAX_LENGTH),
    )

    num_labels = len(LABEL_MAP)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=num_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    from transformers import get_linear_schedule_with_warmup

    # --- scheduler setup ---
    num_training_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    # Create unique run directories
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'../training_metrics/{DATA_SET}_{run_id}'
    os.makedirs(run_dir, exist_ok=True)
    model_dir = f"../models/{DATA_SET}_{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    # write the hyperparameters to a json file
    run_hyperparams = {
        "MODEL": MODEL,
        "DATA_SET": DATA_SET,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_LENGTH": MAX_LENGTH,
        "THRESHOLD": THRESHOLD,
        "NUM_EPOCHS": NUM_EPOCHS,
        "PATIENCE": PATIENCE,
        "LR": LR,
    }
    with open(f'{run_dir}/hyperparameters.json', 'w') as f:
        json.dump(run_hyperparams, f, indent=4)


    train_losses_per_epoch = []
    val_losses_per_epoch = []
    val_accs_per_epoch = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{NUM_EPOCHS} ==========")

        # ---------- TRAINING ----------
        model.train()
        total_train_loss = 0.0

        for batch_idx, (enc, labels) in enumerate(train_loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            outputs = model(**enc, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(train_loader)} "
                    f"| Batch loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses_per_epoch.append(avg_train_loss)
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for enc, labels in val_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                outputs = model(**enc, labels=labels)
                logits = outputs.logits
                loss = outputs.loss

                val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = np.mean(val_losses)
        val_losses_per_epoch.append(avg_val_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        val_accs_per_epoch.append(val_acc)

        print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1} validation accuracy: {val_acc:.4f}")
        ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=list(LABEL_MAP.keys()), labels=[0,1,2])
        plt.tight_layout()
        plt.savefig(f"{run_dir}/confusion_matrix_epoch_{epoch+1}.png")
        # ---------- EARLY STOPPING CHECK ----------
        loss_diff = avg_val_loss - best_val_loss

        if loss_diff < -THRESHOLD:
            # real improvement
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{model_dir}/{epoch+1}_best_model.pt")
            print(f"  -> New best model saved (val loss improved by {abs(loss_diff):.4f}).")

        elif abs(loss_diff) <= THRESHOLD:
            if val_accs_per_epoch[-1] > max(val_accs_per_epoch[:-1]):
                # accuracy improved, even if loss didn't change much
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"{model_dir}/{epoch+1}_best_acc_model.pt")
                print(f"  -> New best model saved (val accuracy improved).")
            print(f"  -> Small change in val loss ({loss_diff:.4f}), ignored (within threshold).")
        else:
            # real degradation
            epochs_no_improve += 1
            print(f"  -> Val loss worsened by {loss_diff:.4f}; epochs_no_improve = {epochs_no_improve}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered (threshold-aware).")
                break

    # plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, label="Train Loss")
    plt.plot(range(1, len(val_losses_per_epoch) + 1), val_losses_per_epoch, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.savefig(f"{run_dir}/loss_per_epoch.png")


