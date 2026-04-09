"""Analyze multimodal PTB-XL training results.

This script is separate from training and is used to:
- inspect overfitting curves from saved history JSON files
- compare optimizer performance from the empirical study
- summarize train/validation/test metrics

Run examples:
- python src/analyze_training.py --checkpoint-dir checkpoints --optimizer adamw
- python src/analyze_training.py --checkpoint-dir checkpoints --optimizer all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_history(checkpoint_dir: Path, optimizer: str) -> Dict[str, list]:
    history_path = checkpoint_dir / f"training_history_{optimizer}.json"
    with open(history_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_results(checkpoint_dir: Path) -> Dict[str, Dict[str, float]]:
    results_path = checkpoint_dir / "optimizer_study_results.json"
    with open(results_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


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
    results = load_results(checkpoint_dir)
    print("\n=== OPTIMIZER COMPARISON ===")
    header = f"{'optimizer':12s} {'val_loss':>10s} {'val_f1':>10s} {'test_loss':>10s} {'test_f1':>10s}"
    print(header)
    print("-" * len(header))
    for optimizer_name, metrics in results.items():
        print(
            f"{optimizer_name:12s} "
            f"{metrics.get('val_loss', float('nan')):10.6f} "
            f"{metrics.get('val_f1_micro', float('nan')):10.6f} "
            f"{metrics.get('test_loss', float('nan')):10.6f} "
            f"{metrics.get('test_f1_micro', float('nan')):10.6f}"
        )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping comparison plot.")
        return

    names = list(results.keys())
    val_losses = [results[n].get("val_loss", float("nan")) for n in names]
    test_losses = [results[n].get("test_loss", float("nan")) for n in names]
    val_f1 = [results[n].get("val_f1_micro", float("nan")) for n in names]
    test_f1 = [results[n].get("test_f1_micro", float("nan")) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(names, val_losses, label="val_loss", alpha=0.8)
    axes[0].bar(names, test_losses, label="test_loss", alpha=0.6)
    axes[0].set_title("Optimizer loss comparison")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, val_f1, label="val_f1_micro", alpha=0.8)
    axes[1].bar(names, test_f1, label="test_f1_micro", alpha=0.6)
    axes[1].set_title("Optimizer F1 comparison")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = checkpoint_dir / "optimizer_comparison.png"
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