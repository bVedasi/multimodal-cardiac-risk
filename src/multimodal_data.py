"""Data loading utilities for the multimodal PTB-XL pipeline.

This module consumes the arrays produced by preprocess.py and prepares:
- train/validation/test dataset objects
- PyTorch DataLoaders
- a small in-memory split utility for the train set

The goal is to make the next architecture steps easy to plug in:
ECG encoder, metadata encoder, SCP-code encoder, and multimodal fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError as exc:  # pragma: no cover - handled when dependencies are installed
    raise ImportError(
        "PyTorch is required for the multimodal data pipeline. Install it before using src/multimodal_data.py."
    ) from exc


@dataclass(frozen=True)
class MultimodalDataConfig:
    processed_dir: Path
    batch_size: int = 32
    validation_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True


class PTBXLMultimodalDataset(Dataset):
    """Torch dataset for the saved PTB-XL multimodal arrays."""

    def __init__(self, ecg: np.ndarray, tab: np.ndarray, scp: np.ndarray, labels: np.ndarray) -> None:
        if not (len(ecg) == len(tab) == len(scp) == len(labels)):
            raise ValueError("All arrays must have the same number of samples.")

        self.ecg = ecg.astype(np.float32, copy=False)
        self.tab = tab.astype(np.float32, copy=False)
        self.scp = scp.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ecg_sample = self.ecg[index].T
        return {
            "ecg": torch.tensor(ecg_sample.tolist(), dtype=torch.float32),
            "tab": torch.tensor(self.tab[index].tolist(), dtype=torch.float32),
            "scp": torch.tensor(self.scp[index].tolist(), dtype=torch.float32),
            "label": torch.tensor(self.labels[index].tolist(), dtype=torch.float32),
        }


def _load_split_arrays(processed_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ecg = np.load(processed_dir / f"X_ecg_{split}.npy", allow_pickle=True)
    tab = np.load(processed_dir / f"X_tab_{split}.npy", allow_pickle=True)
    scp = np.load(processed_dir / f"X_scp_{split}.npy", allow_pickle=True)
    labels = np.load(processed_dir / f"y_{split}.npy", allow_pickle=True)
    return (
        np.asarray(ecg, dtype=np.float32),
        np.asarray(tab, dtype=np.float32),
        np.asarray(scp, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def load_processed_datasets(processed_dir: Path | str) -> Dict[str, PTBXLMultimodalDataset]:
    """Load train and test datasets from the processed folder."""

    processed_dir = Path(processed_dir)
    train_arrays = _load_split_arrays(processed_dir, "train")
    test_arrays = _load_split_arrays(processed_dir, "test")

    return {
        "train": PTBXLMultimodalDataset(*train_arrays),
        "test": PTBXLMultimodalDataset(*test_arrays),
    }


def create_train_val_split(dataset: Dataset, validation_fraction: float = 0.1, seed: int = 42) -> Tuple[Subset, Subset]:
    """Create a deterministic train/validation split from the training dataset."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    n_samples = len(dataset)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(n_samples * validation_fraction)))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_dataloaders(config: MultimodalDataConfig) -> Dict[str, DataLoader]:
    """Create train/validation/test DataLoaders."""

    datasets = load_processed_datasets(config.processed_dir)
    train_dataset, val_dataset = create_train_val_split(
        datasets["train"],
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def describe_processed_data(processed_dir: Path | str) -> Dict[str, tuple]:
    """Return array shapes for a quick sanity check."""

    processed_dir = Path(processed_dir)
    shapes = {}
    for split in ("train", "test"):
        ecg, tab, scp, labels = _load_split_arrays(processed_dir, split)
        shapes[f"{split}_ecg"] = ecg.shape
        shapes[f"{split}_tab"] = tab.shape
        shapes[f"{split}_scp"] = scp.shape
        shapes[f"{split}_labels"] = labels.shape
    return shapes


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "processed"
    print(describe_processed_data(base_dir))