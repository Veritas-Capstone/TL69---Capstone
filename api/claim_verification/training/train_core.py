import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from api.claim_verification.training.training_helpers import (
    PairwiseExpansionDataset,
    collate_pairwise,
)
from api.claim_verification.training.training_joint_helpers import (
    JointEvidenceDataset,
    collate_joint_batch,
)
from api.claim_verification.models.claim_evidence_attention import (
    ClaimEvidenceAttentionModel,
)


def _split_df(df, val_size=0.2, seed=10):
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, shuffle=True
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val


def _build_loader(
    df,
    label_map,
    tokenizer,
    batch_size,
    max_length,
    use_attention_model,
    max_evidence,
    shuffle,
    nei_fill=False,
    nei_fill_k=2,
    nei_fill_seed=10,
    nei_fill_prob=1.0,
):
    if use_attention_model:
        dataset = JointEvidenceDataset(
            df,
            label_map,
            max_evidence=max_evidence,
            nei_fill=nei_fill,
            nei_fill_prob=nei_fill_prob,
            nei_fill_k=nei_fill_k,
            nei_fill_seed=nei_fill_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_joint_batch(
                batch,
                tokenizer,
                max_length=max_length,
            ),
        )
    else:
        dataset = PairwiseExpansionDataset(
            df,
            label_map,
            nei_fill=nei_fill,
            nei_fill_prob=nei_fill_prob,
            nei_fill_k=nei_fill_k,
            nei_fill_seed=nei_fill_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_pairwise(batch, tokenizer, max_length=max_length),
        )
    return loader


def _jsonify(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _jsonify(v) for v in obj ]
    return obj


def train_claim_model(
    data_set,
    data_path,
    label_map,
    model_name,
    metrics_dir,
    models_dir,
    init_weights="hf",
    batch_size=2,
    max_length=256,
    threshold=0.05,
    num_epochs=6,
    patience=2,
    lr=2e-5,
    use_attention_model=True,
    max_evidence=5,
    accum_steps=8,
    num_heads=4,
    mix_config=None,
    nei_fill=False,
    nei_fill_k=2,
    nei_fill_seed=10,
    nei_fill_prob=1.0,
):
    metrics_dir = Path(metrics_dir)
    models_dir = Path(models_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = pd.read_csv(data_path)
    df_aux = None
    if mix_config:
        aux_path = mix_config.get("aux_data_path")
        if not aux_path:
            raise ValueError("mix_config requires 'aux_data_path'")
        df_aux = pd.read_csv(aux_path)

    if mix_config:
        val_size = mix_config.get("val_size", 0.2)
        df_main_train, df_main_val = _split_df(df, val_size=val_size)
        df_aux_train, _df_aux_val = _split_df(df_aux, val_size=val_size)
    else:
        df_train, df_val = _split_df(df, val_size=0.2)

    BATCH_SIZE = batch_size
    MAX_LENGTH = max_length
    THRESHOLD = threshold
    NUM_EPOCHS = num_epochs
    PATIENCE = patience
    LR = lr
    ACCUM_STEPS = max(1, int(accum_steps))

    if use_attention_model and mix_config:
        from api.claim_verification.training.mixed_sampling import build_mixed_joint_loader
        train_loader = build_mixed_joint_loader(
            df_main=df_main_train,
            df_aux=df_aux_train,
            label_map=label_map,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            max_evidence=max_evidence,
            ratio_main=mix_config.get("ratio_main", 0.7),
            ratio_aux=mix_config.get("ratio_aux", 0.3),
            epoch_size=mix_config.get("epoch_size"),
            balance_labels=mix_config.get("balance_labels", True),
            balance_labels_main=mix_config.get("balance_labels_main"),
            balance_labels_aux=mix_config.get("balance_labels_aux"),
            nei_fill_main=mix_config.get("nei_fill_main", False),
            nei_fill_aux=mix_config.get("nei_fill_aux", False),
            nei_fill_k=mix_config.get("nei_fill_k", nei_fill_k),
            nei_fill_seed=mix_config.get("nei_fill_seed", nei_fill_seed),
            nei_fill_prob=mix_config.get("nei_fill_prob", nei_fill_prob),
            label_weights_main=mix_config.get("label_weights_main"),
            label_weights_aux=mix_config.get("label_weights_aux"),
        )
        val_loader = _build_loader(
            df_main_val,
            label_map,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            use_attention_model=use_attention_model,
            max_evidence=max_evidence,
            shuffle=False,
            nei_fill=nei_fill,
            nei_fill_k=nei_fill_k,
            nei_fill_seed=nei_fill_seed,
            nei_fill_prob=nei_fill_prob,
        )
    elif (not use_attention_model) and mix_config:
        from api.claim_verification.training.mixed_sampling import build_mixed_pairwise_loader
        train_loader = build_mixed_pairwise_loader(
            df_main=df_main_train,
            df_aux=df_aux_train,
            label_map=label_map,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            ratio_main=mix_config.get("ratio_main", 0.7),
            ratio_aux=mix_config.get("ratio_aux", 0.3),
            epoch_size=mix_config.get("epoch_size"),
            balance_labels=mix_config.get("balance_labels", True),
            balance_labels_main=mix_config.get("balance_labels_main"),
            balance_labels_aux=mix_config.get("balance_labels_aux"),
            nei_fill_main=mix_config.get("nei_fill_main", False),
            nei_fill_aux=mix_config.get("nei_fill_aux", False),
            nei_fill_k=mix_config.get("nei_fill_k", nei_fill_k),
            nei_fill_seed=mix_config.get("nei_fill_seed", nei_fill_seed),
            nei_fill_prob=mix_config.get("nei_fill_prob", nei_fill_prob),
            label_weights_main=mix_config.get("label_weights_main"),
            label_weights_aux=mix_config.get("label_weights_aux"),
        )
        val_loader = _build_loader(
            df_main_val,
            label_map,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            use_attention_model=use_attention_model,
            max_evidence=max_evidence,
            shuffle=False,
            nei_fill=nei_fill,
            nei_fill_k=nei_fill_k,
            nei_fill_seed=nei_fill_seed,
            nei_fill_prob=nei_fill_prob,
        )
    else:
        train_loader = _build_loader(
            df_train,
            label_map,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            use_attention_model=use_attention_model,
            max_evidence=max_evidence,
            shuffle=True,
            nei_fill=nei_fill,
            nei_fill_k=nei_fill_k,
            nei_fill_seed=nei_fill_seed,
            nei_fill_prob=nei_fill_prob,
        )
        val_loader = _build_loader(
            df_val,
            label_map,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            use_attention_model=use_attention_model,
            max_evidence=max_evidence,
            shuffle=False,
            nei_fill=nei_fill,
            nei_fill_k=nei_fill_k,
            nei_fill_seed=nei_fill_seed,
            nei_fill_prob=nei_fill_prob,
        )

    num_labels = len(label_map)

    if use_attention_model:
        encoder = AutoModel.from_pretrained(model_name)
        model = ClaimEvidenceAttentionModel(
            encoder,
            hidden_dim=encoder.config.hidden_size,
            num_labels=num_labels,
            num_heads=num_heads,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if init_weights != "hf":
        state_dict = torch.load(init_weights, map_location=device)
        if use_attention_model:
            if any(k.startswith("encoder.") for k in state_dict.keys()):
                encoder_state = {
                    k[len("encoder."):] : v for k, v in state_dict.items() if k.startswith("encoder.")
                }
            else:
                encoder_state = {
                    k.replace("roberta.", "") if k.startswith("roberta.") else k: v
                    for k, v in state_dict.items()
                    if not k.startswith("classifier")
                }
            model.encoder.load_state_dict(encoder_state, strict=False)
        else:
            model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * math.ceil(len(train_loader) / ACCUM_STEPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = metrics_dir / f"{data_set}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = models_dir / data_set / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = None

    run_hyperparams = {
        "MODEL": model_name,
        "DATA_SET": data_set,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_LENGTH": MAX_LENGTH,
        "THRESHOLD": THRESHOLD,
        "NUM_EPOCHS": NUM_EPOCHS,
        "PATIENCE": PATIENCE,
        "LR": LR,
        "INIT_WEIGHTS": init_weights,
        "USE_ATTENTION_MODEL": use_attention_model,
        "MAX_EVIDENCE": max_evidence,
        "ACCUM_STEPS": ACCUM_STEPS,
        "NUM_HEADS": num_heads,
        "MIX_CONFIG": mix_config,
        "NEI_FILL": nei_fill,
        "NEI_FILL_K": nei_fill_k,
        "NEI_FILL_SEED": nei_fill_seed,
        "NEI_FILL_PROB": nei_fill_prob,
    }
    import json as _json
    with open(run_dir / "hyperparameters.json", "w") as f:
        _json.dump(_jsonify(run_hyperparams), f, indent=4)

    train_losses_per_epoch = []
    val_losses_per_epoch = []
    val_accs_per_epoch = []

    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{NUM_EPOCHS} ==========")

        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if use_attention_model:
                claim_enc, ev_enc, evid_mask, labels = batch
                claim_enc = {k: v.to(device) for k, v in claim_enc.items()}
                ev_enc = {k: v.to(device) for k, v in ev_enc.items()}
                evid_mask = evid_mask.to(device)
                labels = labels.to(device)
                outputs = model(claim_enc, ev_enc, evid_mask, labels)
                loss = outputs["loss"]
            else:
                enc, labels = batch
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                outputs = model(**enc, labels=labels)
                loss = outputs.loss

            total_train_loss += loss.item()
            (loss / ACCUM_STEPS).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses_per_epoch.append(avg_train_loss)
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if use_attention_model:
                    claim_enc, ev_enc, evid_mask, labels = batch
                    claim_enc = {k: v.to(device) for k, v in claim_enc.items()}
                    ev_enc = {k: v.to(device) for k, v in ev_enc.items()}
                    evid_mask = evid_mask.to(device)
                    labels = labels.to(device)
                    outputs = model(claim_enc, ev_enc, evid_mask, labels)
                    logits = outputs["logits"]
                    loss = outputs["loss"]
                else:
                    enc, labels = batch
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
            display_labels=list(label_map.keys()),
            labels=list(range(len(label_map))),
        )
        plt.tight_layout()
        plt.savefig(run_dir / f"confusion_matrix_epoch_{epoch+1}.png")
        plt.close()

        loss_diff = avg_val_loss - best_val_loss
        if loss_diff < -THRESHOLD:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ckpt_path = model_dir / f"{epoch+1}_best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_model_path = ckpt_path
            print(
                f"  -> New best model saved at {ckpt_path} (val loss improved by {abs(loss_diff):.4f})."
            )
        elif abs(loss_diff) <= THRESHOLD:
            if len(val_accs_per_epoch) > 1 and val_accs_per_epoch[-1] > max(val_accs_per_epoch[:-1]):
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                ckpt_path = model_dir / f"{epoch+1}_best_acc_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                best_model_path = ckpt_path
                print(f"  -> New best model saved at {ckpt_path} (val accuracy improved).")
            print(f"  -> Small change in val loss ({loss_diff:.4f}), ignored (within threshold).")
        else:
            epochs_no_improve += 1
            print(f"  -> Val loss worsened by {loss_diff:.4f}; epochs_no_improve = {epochs_no_improve}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered (threshold-aware).")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, label="Train Loss")
    plt.plot(range(1, len(val_losses_per_epoch) + 1), val_losses_per_epoch, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss per Epoch ({data_set})")
    plt.legend()
    plt.savefig(run_dir / "loss_per_epoch.png")
    plt.close()

    print(f"\nFinished training on {data_set}.")
    print(f"  Metrics dir: {run_dir}")
    print(f"  Model dir:   {model_dir}")

    if best_model_path is not None:
        model_root = models_dir / data_set
        model_root.mkdir(parents=True, exist_ok=True)
        latest_path = model_root / "latest.pt"
        import shutil
        shutil.copy(best_model_path, latest_path)
        print(f"[INFO] Best checkpoint: {best_model_path}")
        print(f"[INFO] latest.pt updated at: {latest_path}")
    else:
        print("[WARN] No best checkpoint found; latest.pt not updated.")

    return {
        "dataset": data_set,
        "run_dir": run_dir,
        "model_dir": model_dir,
    }
