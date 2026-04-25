from torch_geometric.datasets import TUDataset
import numpy as np
from torch_geometric.data import Dataset
from typing import Optional
from gplab.utils.registry import TU_DATASETS


def load_dataset(dataset: str) -> Dataset:
    _dataset = None
    if dataset in TU_DATASETS:
        _dataset = TUDataset(root="/tmp/TUDataset", name=dataset, use_node_attr=True)
    return _dataset


def build_split_indices(
        dataset_size: int,
        seed: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
) -> dict:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1")

    rng = np.random.default_rng(seed)
    rnd_idx = rng.permutation(dataset_size).tolist()

    train_end = int(train_ratio * dataset_size)
    val_end = int((train_ratio + val_ratio) * dataset_size)

    return {
        "train": rnd_idx[:train_end],
        "val": rnd_idx[train_end:val_end],
        "test": rnd_idx[val_end:],
    }


def split_dataset(
        dataset: Dataset,
        seed: Optional[int] = None,
        split_indices: Optional[dict] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
):
    if split_indices is None:
        if seed is None:
            rnd_idx = np.random.permutation(len(dataset)).tolist()
            train_end = int(train_ratio * len(dataset))
            val_end = int((train_ratio + val_ratio) * len(dataset))
            split_indices = {"train": rnd_idx[:train_end], "val": rnd_idx[train_end:val_end], "test": rnd_idx[val_end:]}
        else:
            split_indices = build_split_indices(len(dataset), seed, train_ratio=train_ratio, val_ratio=val_ratio)

    train_dataset = dataset[split_indices["train"]]
    val_dataset = dataset[split_indices["val"]]
    test_dataset = dataset[split_indices["test"]]

    return train_dataset, val_dataset, test_dataset
