"""Inference entry point for a trained multimodal PTB-XL model."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

multimodal_data = importlib.import_module("src.multimodal_data")
multimodal_model = importlib.import_module("src.multimodal_model")
load_processed_datasets = multimodal_data.load_processed_datasets
MultimodalPTBXLNet = multimodal_model.MultimodalPTBXLNet
ModelConfig = multimodal_model.ModelConfig


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    datasets = load_processed_datasets(processed_dir)
    sample = datasets["train"][0]

    model_config = checkpoint["model_config"]
    if isinstance(model_config, dict):
        config = ModelConfig(**model_config)
    else:
        config = model_config

    model = MultimodalPTBXLNet(
        tabular_dim=sample["tab"].shape[-1],
        scp_dim=sample["scp"].shape[-1],
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_sample(model: MultimodalPTBXLNet, sample: dict) -> torch.Tensor:
    with torch.no_grad():
        probs = model.predict_proba(
            sample["ecg"].unsqueeze(0),
            sample["tab"].unsqueeze(0),
            sample["scp"].unsqueeze(0),
        )
    return probs.squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved multimodal PTB-XL checkpoint.")
    parser.add_argument("--processed-dir", default="processed", help="Folder containing the saved .npy arrays.")
    parser.add_argument("--checkpoint", default="checkpoints/best_multimodal_ptbxl_adamw.pt", help="Path to the saved model checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Which processed split to use.")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the selected split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    checkpoint_path = Path(args.checkpoint)

    model, checkpoint = load_checkpoint(checkpoint_path, processed_dir)
    datasets = load_processed_datasets(processed_dir)
    sample = datasets[args.split][args.index]

    probs = predict_sample(model, sample)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Split: {args.split} | index: {args.index}")
    print(f"Probabilities: {probs.tolist()}")
    print(f"Predicted labels above 0.5: {(probs >= 0.5).nonzero(as_tuple=False).flatten().tolist()}")


if __name__ == "__main__":
    main()
