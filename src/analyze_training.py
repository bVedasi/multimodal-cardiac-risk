"""Analyze multimodal PTB-XL training results.

This script is separate from training and is used to:
- inspect overfitting curves from saved history JSON files
- compare optimizer performance from the empirical study
- summarize train/validation/test metrics
- build a multi-metric comparison chart across all optimizers

Run examples:
- python src/analyze_training.py --checkpoint-dir checkpoints --optimizer adamw
- python src/analyze_training.py --checkpoint-dir checkpoints --optimizer all
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

multimodal_data = importlib.import_module("src.multimodal_data")
multimodal_model = importlib.import_module("src.multimodal_model")
MultimodalDataConfig = multimodal_data.MultimodalDataConfig
create_dataloaders = multimodal_data.create_dataloaders
load_processed_datasets = multimodal_data.load_processed_datasets
MultimodalPTBXLNet = multimodal_model.MultimodalPTBXLNet
ModelConfig = multimodal_model.ModelConfig


def safe_torch_load(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_history(checkpoint_dir: Path, optimizer: str) -> Dict[str, list]:
    history_path = checkpoint_dir / f"training_history_{optimizer}.json"
    with open(history_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_results(checkpoint_dir: Path) -> Dict[str, Dict[str, float]]:
    results_path = checkpoint_dir / "optimizer_study_results.json"
    with open(results_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_checkpoint(checkpoint_dir: Path, optimizer: str, processed_dir: Path) -> tuple[MultimodalPTBXLNet, Dict[str, object]]:
    checkpoint_path = checkpoint_dir / f"best_multimodal_ptbxl_{optimizer}.pt"
    checkpoint = safe_torch_load(checkpoint_path)

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


def summarize_history(history: Dict[str, list]) -> Dict[str, float]:
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_f1 = history["train_f1_micro"]
    val_f1 = history["val_f1_micro"]

    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    return {
        "best_epoch": float(epochs[best_idx]),
        "best_train_loss": float(train_loss[best_idx]),
        "best_val_loss": float(val_loss[best_idx]),
        "best_train_f1_micro": float(train_f1[best_idx]),
        "best_val_f1_micro": float(val_f1[best_idx]),
        "loss_gap": float(val_loss[best_idx] - train_loss[best_idx]),
        "f1_gap": float(train_f1[best_idx] - val_f1[best_idx]),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "final_train_f1_micro": float(train_f1[-1]),
        "final_val_f1_micro": float(val_f1[-1]),
    }


def best_epoch_index(history: Dict[str, list]) -> int:
    val_loss = history["val_loss"]
    return int(min(range(len(val_loss)), key=lambda i: val_loss[i]))


def compute_gradient_norm(model: MultimodalPTBXLNet, loader: torch.utils.data.DataLoader, device: str, max_batches: int = 10) -> float:
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_norms: List[float] = []

    model.eval()
    for batch_idx, batch in enumerate(loader, start=1):
        ecg = batch["ecg"].to(device)
        tab = batch["tab"].to(device)
        scp = batch["scp"].to(device)
        labels = batch["label"].to(device)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(ecg, tab, scp)
            loss = criterion(logits, labels)
            loss.backward()

        total_sq = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm = float(parameter.grad.detach().norm(2).cpu())
                total_sq += grad_norm * grad_norm
        batch_norms.append(float(np.sqrt(total_sq)))

        if batch_idx >= max_batches:
            break

    if not batch_norms:
        return float("nan")
    return float(np.mean(batch_norms))


def build_optimizer_summary(checkpoint_dir: Path) -> List[Dict[str, float]]:
    results = load_results(checkpoint_dir)
    rows: List[Dict[str, float]] = []

    for optimizer_name, metrics in results.items():
        history = load_history(checkpoint_dir, optimizer_name)
        history_summary = summarize_history(history)

        checkpoint_path = checkpoint_dir / f"best_multimodal_ptbxl_{optimizer_name}.pt"
        checkpoint = safe_torch_load(checkpoint_path)
        processed_dir = Path(checkpoint["config"]["processed_dir"])

        dataloaders = create_dataloaders(
            MultimodalDataConfig(
                processed_dir=processed_dir,
                batch_size=int(checkpoint["config"]["batch_size"]),
                validation_fraction=float(checkpoint["config"]["validation_fraction"]),
                seed=int(checkpoint["config"]["seed"]),
            )
        )

        model, _ = load_checkpoint(checkpoint_dir, optimizer_name, processed_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        grad_norm = compute_gradient_norm(model, dataloaders["train"], device=device)

        best_epoch = float(history_summary["best_epoch"])
        row = {
            "optimizer": optimizer_name,
            "train_loss": float(history_summary["best_train_loss"]),
            "val_loss": float(history_summary["best_val_loss"]),
            "auc_macro": float(metrics.get("val_auc_macro", float("nan"))),
            "f1_micro": float(metrics.get("val_f1_micro", float("nan"))),
            "gradient_norm": float(grad_norm),
            "best_epoch": best_epoch,
            "convergence_speed": float(1.0 / best_epoch) if best_epoch > 0 else float("nan"),
            "train_f1_micro": float(history_summary["best_train_f1_micro"]),
            "val_f1_micro": float(history_summary["best_val_f1_micro"]),
            "train_auc_macro": float(metrics.get("train_auc_macro", float("nan"))),
            "val_auc_macro": float(metrics.get("val_auc_macro", float("nan"))),
            "test_loss": float(metrics.get("test_loss", float("nan"))),
            "test_f1_micro": float(metrics.get("test_f1_micro", float("nan"))),
            "test_auc_macro": float(metrics.get("test_auc_macro", float("nan"))),
        }
        rows.append(row)

    return rows


def print_summary_table(name: str, summary: Dict[str, float]) -> None:
    print(f"\n=== {name.upper()} ===")
    for key, value in summary.items():
        print(f"{key:24s}: {value:.6f}")


def plot_history(history: Dict[str, list], optimizer: str, checkpoint_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping plots.")
        return

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train_loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val_loss")
    axes[0].set_title(f"Loss curves - {optimizer}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_f1_micro"], marker="o", label="train_f1_micro")
    axes[1].plot(epochs, history["val_f1_micro"], marker="o", label="val_f1_micro")
    axes[1].set_title(f"F1 curves - {optimizer}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = checkpoint_dir / f"analysis_curves_{optimizer}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def analyze_single_optimizer(checkpoint_dir: Path, optimizer: str) -> None:
    history = load_history(checkpoint_dir, optimizer)
    summary = summarize_history(history)
    print_summary_table(optimizer, summary)
    plot_history(history, optimizer, checkpoint_dir)


def analyze_all_optimizers(checkpoint_dir: Path) -> None:
    rows = build_optimizer_summary(checkpoint_dir)

    print("\n=== OPTIMIZER COMPARISON ===")
    header = (
        f"{'optimizer':12s} {'train_loss':>10s} {'val_loss':>10s} {'AUC':>10s} {'F1':>10s} "
        f"{'grad_norm':>12s} {'conv_speed':>12s} {'epoch':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['optimizer']:12s} "
            f"{row['train_loss']:10.6f} "
            f"{row['val_loss']:10.6f} "
            f"{row['val_auc_macro']:10.6f} "
            f"{row['val_f1_micro']:10.6f} "
            f"{row['gradient_norm']:12.6f} "
            f"{row['convergence_speed']:12.6f} "
            f"{row['best_epoch']:8.0f}"
        )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping comparison plot.")
        return

    names = [row["optimizer"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]
    aucs = [row["val_auc_macro"] for row in rows]
    f1s = [row["val_f1_micro"] for row in rows]
    grad_norms = [row["gradient_norm"] for row in rows]
    convergence_speed = [row["convergence_speed"] for row in rows]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plots = [
        (axes[0, 0], train_losses, "Training loss"),
        (axes[0, 1], val_losses, "Validation loss"),
        (axes[0, 2], aucs, "Validation AUC"),
        (axes[1, 0], f1s, "Validation F1 score"),
        (axes[1, 1], grad_norms, "Gradient norm"),
        (axes[1, 2], convergence_speed, "Convergence speed (1 / epoch)"),
    ]

    for ax, values, title in plots:
        bars = ax.bar(names, values, color="#4C78A8", alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 2].set_ylabel("AUC")
    axes[1, 0].set_ylabel("F1")
    axes[1, 1].set_ylabel("L2 norm")
    axes[1, 2].set_ylabel("Relative speed")

    plt.tight_layout()
    out_path = checkpoint_dir / "optimizer_full_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multimodal PTB-XL training output.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd", "rmsprop", "adagrad", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.optimizer == "all":
        analyze_all_optimizers(checkpoint_dir)
    else:
        analyze_single_optimizer(checkpoint_dir, args.optimizer)


if __name__ == "__main__":
    main()