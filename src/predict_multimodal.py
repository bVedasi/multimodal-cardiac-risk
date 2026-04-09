"""Inference entry point for trained multimodal PTB-XL models.

This script can:
- load one sample from the processed PTB-XL arrays
- load a user-provided JSON sample containing ECG, clinical tabular data,
    SCP codes, and optional true labels
- run prediction with every checkpoint in checkpoints/
- report whether the prediction matches the provided labels

Expected JSON input format:
{
    "ecg": [[...], ...],
    "tab": [...],
    "scp": [...],
    "labels": [0, 1, 0, 0, 1]   // optional
}

ECG can be provided as either [5000, 12] or [12, 5000].
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

multimodal_data = importlib.import_module("src.multimodal_data")
multimodal_model = importlib.import_module("src.multimodal_model")
load_processed_datasets = multimodal_data.load_processed_datasets
MultimodalPTBXLNet = multimodal_model.MultimodalPTBXLNet
ModelConfig = multimodal_model.ModelConfig


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_input_dimensions(checkpoint: dict) -> tuple[int, int, int]:
    state_dict = checkpoint["model_state_dict"]
    ecg_channels = int(state_dict["ecg_encoder.stem.0.weight"].shape[1])
    tabular_dim = int(state_dict["tab_encoder.net.0.weight"].shape[1])
    scp_dim = int(state_dict["scp_encoder.net.0.weight"].shape[1])
    return ecg_channels, tabular_dim, scp_dim


@dataclass(frozen=True)
class TrainConfig:
    processed_dir: Path
    epochs: int = 5
    batch_size: int = 32
    validation_fraction: float = 0.1
    seed: int = 42
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    checkpoint_dir: Path = Path("checkpoints")


def load_checkpoint(checkpoint_path: Path, processed_dir: Path) -> tuple[MultimodalPTBXLNet, dict]:
    checkpoint = safe_torch_load(checkpoint_path)

    model_config = checkpoint["model_config"]
    if isinstance(model_config, dict):
        config = ModelConfig(**model_config)
    else:
        config = model_config

    ecg_channels, tabular_dim, scp_dim = infer_input_dimensions(checkpoint)
    config = ModelConfig(
        ecg_channels=ecg_channels,
        ecg_embedding_dim=config.ecg_embedding_dim,
        tabular_embedding_dim=config.tabular_embedding_dim,
        scp_embedding_dim=config.scp_embedding_dim,
        metadata_embedding_dim=config.metadata_embedding_dim,
        fusion_dim=config.fusion_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )

    model = MultimodalPTBXLNet(
        tabular_dim=tabular_dim,
        scp_dim=scp_dim,
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_label_names(processed_dir: Path) -> List[str]:
    info_path = processed_dir / "preprocessing_info.json"
    if not info_path.exists():
        return []

    with open(info_path, "r", encoding="utf-8") as fh:
        info = json.load(fh)
    return list(info.get("diagnostic_classes", []))


def _as_numpy_vector(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D list/array.")
    return arr


def _align_vector(values: object, expected_dim: int, *, name: str) -> torch.Tensor:
    arr = _as_numpy_vector(values, name=name)
    if arr.size < expected_dim:
        arr = np.pad(arr, (0, expected_dim - arr.size), mode="constant")
    elif arr.size > expected_dim:
        arr = arr[:expected_dim]
    return torch.tensor(arr, dtype=torch.float32)


def _align_ecg(values: object, expected_channels: int = 12) -> torch.Tensor:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("ecg must be a 2D array with shape [N, 12] or [12, N].")

    if arr.shape[1] == expected_channels:
        arr = arr.T
    elif arr.shape[0] != expected_channels:
        raise ValueError(f"ecg must have {expected_channels} channels.")

    if arr.shape[0] != expected_channels:
        raise ValueError(f"ecg must have {expected_channels} channels after alignment.")

    return torch.tensor(arr, dtype=torch.float32)


def _as_ecg_tensor(values: object) -> torch.Tensor:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("ecg must be a 2D array with shape [N, 12] or [12, N].")

    if arr.shape[1] == 12:
        arr = arr.T
    elif arr.shape[0] != 12:
        raise ValueError("ecg must have 12 channels.")

    return torch.tensor(arr, dtype=torch.float32)


def load_user_sample(input_path: Path) -> Dict[str, torch.Tensor]:
    if input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    elif input_path.suffix.lower() == ".npz":
        payload = dict(np.load(input_path, allow_pickle=True))
    else:
        raise ValueError("Input file must be .json or .npz")

    if not all(key in payload for key in ("ecg", "tab", "scp")):
        raise ValueError("Input must contain ecg, tab, and scp.")

    sample = {
        "ecg": _as_ecg_tensor(payload["ecg"]),
        "tab": _as_numpy_vector(payload["tab"], name="tab"),
        "scp": _as_numpy_vector(payload["scp"], name="scp"),
    }

    if "labels" in payload and payload["labels"] is not None:
        sample["label"] = torch.tensor(_as_numpy_vector(payload["labels"], name="labels"), dtype=torch.float32)
    return sample


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    return sorted(checkpoint_dir.glob("best_multimodal_ptbxl_*.pt"))


def build_model_from_checkpoint(checkpoint_path: Path, processed_dir: Path) -> Tuple[MultimodalPTBXLNet, dict]:
    return load_checkpoint(checkpoint_path, processed_dir)


def predict_sample(model: MultimodalPTBXLNet, sample: dict) -> torch.Tensor:
    with torch.no_grad():
        probs = model.predict_proba(
            sample["ecg"].unsqueeze(0),
            sample["tab"].unsqueeze(0),
            sample["scp"].unsqueeze(0),
        )
    return probs.squeeze(0)


def compare_prediction_to_truth(probs: torch.Tensor, sample: Dict[str, torch.Tensor], threshold: float = 0.5) -> Dict[str, object]:
    result: Dict[str, object] = {
        "predicted_indices": (probs >= threshold).nonzero(as_tuple=False).flatten().tolist(),
        "predicted_binary": (probs >= threshold).to(torch.int32).tolist(),
        "correct": None,
        "label_accuracy": None,
        "exact_match": None,
    }

    if "label" not in sample:
        return result

    true = sample["label"].to(torch.int32)
    pred = (probs >= threshold).to(torch.int32)
    result["correct"] = bool(torch.equal(pred, true))
    result["exact_match"] = bool(torch.equal(pred, true))
    result["label_accuracy"] = float((pred == true).to(torch.float32).mean().item())
    return result


def prepare_sample_for_model(sample: Dict[str, torch.Tensor], checkpoint: dict) -> Dict[str, torch.Tensor]:
    ecg_channels, tabular_dim, scp_dim = infer_input_dimensions(checkpoint)
    prepared = dict(sample)
    prepared["ecg"] = _align_ecg(prepared["ecg"], expected_channels=ecg_channels)
    prepared["tab"] = _align_vector(prepared["tab"], tabular_dim, name="tab")
    prepared["scp"] = _align_vector(prepared["scp"], scp_dim, name="scp")
    return prepared


def print_single_prediction(optimizer: str, probs: torch.Tensor, sample: Dict[str, torch.Tensor], label_names: List[str], threshold: float) -> Dict[str, object]:
    comparison = compare_prediction_to_truth(probs, sample, threshold=threshold)
    pred_indices = comparison["predicted_indices"]
    pred_names = [label_names[i] if i < len(label_names) else str(i) for i in pred_indices]

    print(f"\n=== {optimizer.upper()} ===")
    print(f"Probabilities: {probs.tolist()}")
    print(f"Predicted labels above {threshold:.2f}: {pred_indices}")
    if pred_names:
        print(f"Predicted names: {pred_names}")

    if "label" in sample:
        true = sample["label"].to(torch.int32).tolist()
        true_indices = [i for i, value in enumerate(true) if value == 1]
        true_names = [label_names[i] if i < len(label_names) else str(i) for i in true_indices]
        print(f"True labels: {true_indices}")
        if true_names:
            print(f"True names: {true_names}")
        print(f"Exact match: {comparison['correct']}")
        print(f"Label-wise accuracy: {comparison['label_accuracy']:.3f}" if comparison["label_accuracy"] is not None else "Label-wise accuracy: n/a")

    return {
        "optimizer": optimizer,
        "probabilities": probs.tolist(),
        **comparison,
    }


def predict_with_all_checkpoints(checkpoint_dir: Path, processed_dir: Path, sample: Dict[str, torch.Tensor], threshold: float) -> List[Dict[str, object]]:
    label_names = load_label_names(processed_dir)
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    rows: List[Dict[str, object]] = []
    for checkpoint_path in checkpoints:
        optimizer = checkpoint_path.stem.replace("best_multimodal_ptbxl_", "")
        model, checkpoint = build_model_from_checkpoint(checkpoint_path, processed_dir)
        aligned_sample = prepare_sample_for_model(sample, checkpoint)
        probs = predict_sample(model, aligned_sample)
        row = print_single_prediction(optimizer, probs, aligned_sample, label_names, threshold)
        rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved multimodal PTB-XL checkpoint.")
    parser.add_argument("--processed-dir", default="processed", help="Folder containing the saved .npy arrays.")
    parser.add_argument("--checkpoint", default=None, help="Path to one saved model checkpoint. If omitted, all checkpoints in --checkpoint-dir are used.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Folder containing the optimizer checkpoints.")
    parser.add_argument("--input-json", default=None, help="Path to a JSON or NPZ file containing ecg, tab, scp, and optional labels.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Which processed split to use.")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the selected split.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for multilabel predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    label_names = load_label_names(processed_dir)

    if args.input_json:
        sample = load_user_sample(Path(args.input_json))
        source = f"user file: {args.input_json}"
    else:
        datasets = load_processed_datasets(processed_dir)
        sample = datasets[args.split][args.index]
        source = f"processed split: {args.split} | index: {args.index}"

    print(f"Input source: {source}")
    if label_names:
        print(f"Label names: {label_names}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        model, checkpoint = load_checkpoint(checkpoint_path, processed_dir)
        aligned_sample = prepare_sample_for_model(sample, checkpoint)
        probs = predict_sample(model, aligned_sample)
        print(f"\nLoaded checkpoint: {checkpoint_path}")
        print_single_prediction(checkpoint_path.stem.replace("best_multimodal_ptbxl_", ""), probs, aligned_sample, label_names, args.threshold)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        predict_with_all_checkpoints(checkpoint_dir, processed_dir, sample, args.threshold)


if __name__ == "__main__":
    main()
