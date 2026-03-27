import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from api.claim_verification.training.training_helpers import (
    PairwiseExpansionDataset,
    collate_pairwise,
)
from api.claim_verification.training.training_joint_helpers import (
    JointEvidenceDataset,
    collate_joint_batch,
)


def _label_counts(dataset, num_labels):
    counts = np.zeros(num_labels, dtype=np.float64) + 1e-6
    for sample in dataset:
        # sample structure: (claim, evid, label_id) or (claim, evid_list, label_id)
        label_id = int(sample[-1])
        counts[label_id] += 1.0
    return counts


def _resolve_label_weight(label_weights, label_id, label_map):
    if not label_weights:
        return 1.0
    # Support label names or numeric ids
    if isinstance(label_weights, dict):
        for key, val in label_weights.items():
            if isinstance(key, str) and key in label_map and label_map[key] == label_id:
                return float(val)
            if isinstance(key, int) and key == label_id:
                return float(val)
    return 1.0


def _default_epoch_size(len_main, len_aux, ratio_main, ratio_aux):
    # Ensure the smaller dataset (relative to its ratio) is fully seen in expectation.
    if len_main / max(ratio_main, 1e-6) <= len_aux / max(ratio_aux, 1e-6):
        return int(len_main / max(ratio_main, 1e-6))
    return int(len_aux / max(ratio_aux, 1e-6))


def _resolve_balance_flags(balance_labels, balance_labels_main, balance_labels_aux):
    if balance_labels_main is None and balance_labels_aux is None:
        flag = bool(balance_labels)
        return flag, flag
    if balance_labels_main is None:
        balance_labels_main = False
    if balance_labels_aux is None:
        balance_labels_aux = False
    return bool(balance_labels_main), bool(balance_labels_aux)


def build_mixed_joint_loader(
    df_main,
    df_aux,
    label_map,
    tokenizer,
    batch_size=8,
    max_length=256,
    max_evidence=5,
    ratio_main=0.7,
    ratio_aux=0.3,
    epoch_size=None,
    balance_labels=True,
    balance_labels_main=None,
    balance_labels_aux=None,
    nei_fill_main=False,
    nei_fill_aux=False,
    nei_fill_k=2,
    nei_fill_seed=10,
    nei_fill_prob=1.0,
    label_weights_main=None,
    label_weights_aux=None,
):
    ds_main = JointEvidenceDataset(
        df_main,
        label_map,
        max_evidence=max_evidence,
        nei_fill=nei_fill_main,
        nei_fill_prob=nei_fill_prob,
        nei_fill_k=nei_fill_k,
        nei_fill_seed=nei_fill_seed,
    )
    ds_aux = JointEvidenceDataset(
        df_aux,
        label_map,
        max_evidence=max_evidence,
        nei_fill=nei_fill_aux,
        nei_fill_prob=nei_fill_prob,
        nei_fill_k=nei_fill_k,
        nei_fill_seed=nei_fill_seed,
    )

    concat = ConcatDataset([ds_main, ds_aux])

    w_main = ratio_main / max(len(ds_main), 1)
    w_aux = ratio_aux / max(len(ds_aux), 1)

    bal_main, bal_aux = _resolve_balance_flags(
        balance_labels, balance_labels_main, balance_labels_aux
    )
    if bal_main:
        main_counts = _label_counts(ds_main, num_labels=len(label_map))
        main_weights = [
            (w_main / main_counts[int(y)])
            * _resolve_label_weight(label_weights_main, int(y), label_map)
            for *_, y in ds_main
        ]
    else:
        main_weights = [
            w_main * _resolve_label_weight(label_weights_main, int(y), label_map)
            for *_, y in ds_main
        ]

    if bal_aux:
        aux_counts = _label_counts(ds_aux, num_labels=len(label_map))
        aux_weights = [
            (w_aux / aux_counts[int(y)])
            * _resolve_label_weight(label_weights_aux, int(y), label_map)
            for *_, y in ds_aux
        ]
    else:
        aux_weights = [
            w_aux * _resolve_label_weight(label_weights_aux, int(y), label_map)
            for *_, y in ds_aux
        ]

    weights = main_weights + aux_weights

    if epoch_size is None:
        epoch_size = _default_epoch_size(len(ds_main), len(ds_aux), ratio_main, ratio_aux)

    sampler = WeightedRandomSampler(weights, num_samples=epoch_size, replacement=True)
    loader = DataLoader(
        concat,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_joint_batch(batch, tokenizer, max_length=max_length),
    )
    return loader


def build_mixed_pairwise_loader(
    df_main,
    df_aux,
    label_map,
    tokenizer,
    batch_size=8,
    max_length=256,
    ratio_main=0.7,
    ratio_aux=0.3,
    epoch_size=None,
    balance_labels=True,
    balance_labels_main=None,
    balance_labels_aux=None,
    nei_fill_main=False,
    nei_fill_aux=False,
    nei_fill_k=2,
    nei_fill_seed=10,
    nei_fill_prob=1.0,
    label_weights_main=None,
    label_weights_aux=None,
):
    ds_main = PairwiseExpansionDataset(
        df_main,
        label_map,
        nei_fill=nei_fill_main,
        nei_fill_prob=nei_fill_prob,
        nei_fill_k=nei_fill_k,
        nei_fill_seed=nei_fill_seed,
    )
    ds_aux = PairwiseExpansionDataset(
        df_aux,
        label_map,
        nei_fill=nei_fill_aux,
        nei_fill_prob=nei_fill_prob,
        nei_fill_k=nei_fill_k,
        nei_fill_seed=nei_fill_seed,
    )

    concat = ConcatDataset([ds_main, ds_aux])

    w_main = ratio_main / max(len(ds_main), 1)
    w_aux = ratio_aux / max(len(ds_aux), 1)

    bal_main, bal_aux = _resolve_balance_flags(
        balance_labels, balance_labels_main, balance_labels_aux
    )
    if bal_main:
        main_counts = _label_counts(ds_main, num_labels=len(label_map))
        main_weights = [
            (w_main / main_counts[int(y)])
            * _resolve_label_weight(label_weights_main, int(y), label_map)
            for *_, y in ds_main
        ]
    else:
        main_weights = [
            w_main * _resolve_label_weight(label_weights_main, int(y), label_map)
            for *_, y in ds_main
        ]

    if bal_aux:
        aux_counts = _label_counts(ds_aux, num_labels=len(label_map))
        aux_weights = [
            (w_aux / aux_counts[int(y)])
            * _resolve_label_weight(label_weights_aux, int(y), label_map)
            for *_, y in ds_aux
        ]
    else:
        aux_weights = [
            w_aux * _resolve_label_weight(label_weights_aux, int(y), label_map)
            for *_, y in ds_aux
        ]

    weights = main_weights + aux_weights

    if epoch_size is None:
        epoch_size = _default_epoch_size(len(ds_main), len(ds_aux), ratio_main, ratio_aux)

    sampler = WeightedRandomSampler(weights, num_samples=epoch_size, replacement=True)
    loader = DataLoader(
        concat,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_pairwise(batch, tokenizer, max_length=max_length),
    )
    return loader
