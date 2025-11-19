# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import json
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import os
from datetime import datetime
from training_helpers import PairwiseExpansionDataset, collate_pairwise

# Base HF model
MODEL = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Default dataset for this script (AveriTeC)
# Change the name to match your processed CSV prefix if needed.
DATA_SET = "averitec"
DATA_PATH = f"../data/processed/{DATA_SET}.csv"

LABEL_MAP = {"REFUTED": 0, "NOT ENOUGH INFO": 1, "SUPPORTED": 2}


# %%
def train_averitec(
    init_weights = "hf",
    data_set = DATA_SET,
    data_path = None,
    batch_size = 2,
    max_length = 256,
    threshold = 0.05,
    num_epochs = 6,
    patience = 2,
    lr = 2e-5,
):
    """
    Train RoBERTa on AveriTeC (pairwise expansion).
    
    Parameters
    ----------
    init_weights : str
        "hf" to start from the base HF model, or a path to a .pt state_dict
        (e.g., a FEVER checkpoint) to initialize from.
    data_set : str
        Name of the dataset (used in output directories).
    data_path : str | None
        Path to the CSV. If None, defaults to ../data/processed/{data_set}.csv
    """

    if data_path is None:
        data_path = f"../data/processed/{data_set}.csv"
        
    df = pd.read_csv(data_path)

    print(f"Loading dataset from: {data_path}")
    pair_ds = PairwiseExpansionDataset(df, LABEL_MAP)
    print(f"Total pairwise examples: {len(pair_ds)}")

    # simple random split; good enough for a demo
    indices = np.arange(len(pair_ds))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=10, shuffle=True)

    train_ds = Subset(pair_ds, train_idx)
    val_ds   = Subset(pair_ds, val_idx)

    BATCH_SIZE = batch_size
    MAX_LENGTH = max_length
    THRESHOLD = threshold
    NUM_EPOCHS = num_epochs
    PATIENCE = patience
    LR = lr

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

    # --- Load initial weights if provided (e.g., FEVER checkpoint) ---
    if init_weights != "hf":
        print(f"Initializing AveriTeC training from checkpoint: {init_weights}")
        state_dict = torch.load(init_weights, map_location=device)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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
    run_dir = f'../training_metrics/{data_set}_{run_id}'
    os.makedirs(run_dir, exist_ok=True)
    model_dir = f"../models/{data_set}/{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    
    # store best model path
    best_model_path = None

    # write the hyperparameters to a json file
    run_hyperparams = {
        "MODEL": MODEL,
        "DATA_SET": data_set,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_LENGTH": MAX_LENGTH,
        "THRESHOLD": THRESHOLD,
        "NUM_EPOCHS": NUM_EPOCHS,
        "PATIENCE": PATIENCE,
        "LR": LR,
        "INIT_WEIGHTS": init_weights,
    }
    import json as _json
    with open(f'{run_dir}/hyperparameters.json', 'w') as f:
        _json.dump(run_hyperparams, f, indent=4)

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
        ConfusionMatrixDisplay.from_predictions(
            all_labels, all_preds,
            display_labels=list(LABEL_MAP.keys()),
            labels=[0, 1, 2]
        )
        plt.tight_layout()
        plt.savefig(f"{run_dir}/confusion_matrix_epoch_{epoch+1}.png")
        plt.close()

        # ---------- EARLY STOPPING CHECK ----------
        loss_diff = avg_val_loss - best_val_loss

        if loss_diff < -THRESHOLD:
            # real improvement
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ckpt_path = f"{model_dir}/{epoch+1}_best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_model_path = ckpt_path
            print(f"  -> New best model saved at {ckpt_path} (val loss improved by {abs(loss_diff):.4f}).")

        elif abs(loss_diff) <= THRESHOLD:
            if len(val_accs_per_epoch) > 1 and val_accs_per_epoch[-1] > max(val_accs_per_epoch[:-1]):
                # accuracy improved, even if loss didn't change much
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                ckpt_path = f"{model_dir}/{epoch+1}_best_acc_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                best_model_path = ckpt_path
                print(f"  -> New best model saved at {ckpt_path} (val accuracy improved).")
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
    plt.title(f"Training and Validation Loss per Epoch ({data_set})")
    plt.legend()
    plt.savefig(f"{run_dir}/loss_per_epoch.png")
    plt.close()

    print(f"\nFinished training on {data_set}.")
    print(f"  Metrics dir: {run_dir}")
    print(f"  Model dir:   {model_dir}")
    
    if best_model_path is not None:
        model_root = f"../models/{DATA_SET}"
        os.makedirs(model_root, exist_ok=True)   

        latest_path = os.path.join(model_root, "latest.pt")
        shutil.copy(best_model_path, latest_path)  # <-- overwrites if it already exists

        print(f"[INFO] Best checkpoint: {best_model_path}")
        print(f"[INFO] latest.pt updated at: {latest_path}")
    else:
        print("[WARN] No best checkpoint found; latest.pt not updated.")

    # Return info for training_order.py
    return {
        "dataset": data_set,
        "run_dir": run_dir,
        "model_dir": model_dir,
    }

# %%
if __name__ == "__main__":
    train_averitec()


