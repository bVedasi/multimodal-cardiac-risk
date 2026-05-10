"""Training entry point for the multimodal PTB-XL model."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

multimodal_data = importlib.import_module("src.multimodal_data")
multimodal_model = importlib.import_module("src.multimodal_model")
MultimodalDataConfig = multimodal_data.MultimodalDataConfig
create_dataloaders = multimodal_data.create_dataloaders
build_model_from_batches = multimodal_model.build_model_from_batches


@dataclass(frozen=True)
class TrainConfig:
    processed_dir: Path
    epochs: int = 5 #epoch is here
    batch_size: int = 32
    validation_fraction: float = 0.1
    seed: int = 42
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("checkpoints")


def build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int32)
    metrics = {
        "accuracy": float(np.mean(np.all(y_pred == y_true.astype(np.int32), axis=1))),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["auc_macro"] = roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        metrics["auc_macro"] = float("nan")
    return metrics


def _optimizer_dir(config: TrainConfig) -> Path:
    return config.checkpoint_dir / config.optimizer


def save_checkpoint(
    model: torch.nn.Module,
    config: TrainConfig,
    num_classes: int,
    epoch: int,
    is_best: bool = False,
) -> None:

    optimizer_dir = _optimizer_dir(config)
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": {
            "processed_dir": str(config.processed_dir),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "device": config.device,
            "checkpoint_dir": str(_optimizer_dir(config)),
        },
        "model_config": {
            "ecg_channels": model.config.ecg_channels,
            "ecg_embedding_dim": model.config.ecg_embedding_dim,
            "tabular_embedding_dim": model.config.tabular_embedding_dim,
            "metadata_embedding_dim": model.config.metadata_embedding_dim,
            "fusion_dim": model.config.fusion_dim,
            "num_heads": model.config.num_heads,
            "dropout": model.config.dropout,
            "num_classes": model.config.num_classes,
        },
        "num_classes": num_classes,
    }

    # Save every epoch
    epoch_path = optimizer_dir / f"epoch_{epoch}.pt"
    torch.save(checkpoint, epoch_path)

    # Save best model separately
    if is_best:
        best_path = optimizer_dir / "best.pt"
        torch.save(checkpoint, best_path)



def save_epoch_table_csv(optimizer_name: str, history: Dict[str, list], checkpoint_dir: Path) -> None:
    """Save the per-epoch validation metrics table as a CSV file."""
    import csv
    csv_path = checkpoint_dir / f"epoch_metrics_{optimizer_name}.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "f1_score", "recall", "precision"])
        for i, epoch in enumerate(history["epoch"]):
            writer.writerow([
                epoch,
                round(history["val_accuracy"][i], 6),
                round(history["val_f1_micro"][i], 6),
                round(history["val_recall_micro"][i], 6),
                round(history["val_precision_micro"][i], 6),
            ])
    print(f"Epoch metrics saved -> {csv_path}")


def print_epoch_table(optimizer_name: str, history: Dict[str, list]) -> None:
    """Print a per-epoch validation metrics table for an optimizer."""
    col_w = [7, 10, 10, 10, 11]
    headers = ["Epoch", "Accuracy", "F1-Score", "Recall", "Precision"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_row = "|" + "|".join(h.center(w) for h, w in zip(headers, col_w)) + "|"

    print(f"\n=== {optimizer_name.upper()} — Per-Epoch Validation Metrics ===")
    print(sep)
    print(header_row)
    print(sep)
    for i, epoch in enumerate(history["epoch"]):
        acc  = history["val_accuracy"][i]
        f1   = history["val_f1_micro"][i]
        rec  = history["val_recall_micro"][i]
        prec = history["val_precision_micro"][i]
        row = (
            f"|{str(epoch).center(col_w[0])}"
            f"|{f'{acc:.4f}'.center(col_w[1])}"
            f"|{f'{f1:.4f}'.center(col_w[2])}"
            f"|{f'{rec:.4f}'.center(col_w[3])}"
            f"|{f'{prec:.4f}'.center(col_w[4])}|"
        )
        print(row)
    print(sep)


def save_training_curves(history: Dict[str, list], config: TrainConfig) -> None:
    """Save training history to JSON and plot loss/F1 curves when matplotlib is available."""

    _optimizer_dir(config).mkdir(parents=True, exist_ok=True)
    history_path = _optimizer_dir(config) / f"training_history_{config.optimizer}.json"
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title(f"Loss curves ({config.optimizer})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_f1_micro"], label="train_f1_micro")
    axes[1].plot(epochs, history["val_f1_micro"], label="val_f1_micro")
    axes[1].set_title(f"F1 curves ({config.optimizer})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(_optimizer_dir(config) / f"training_curves_{config.optimizer}.png", dpi=150)
    plt.close(fig)

def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    stage: str,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    y_true_batches = []
    y_score_batches = []
    total_batches = len(loader)
    
    grad_norms = []

    for batch_idx, batch in enumerate(loader, start=1):
        ecg = batch["ecg"].to(device)
        tab = batch["tab"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(ecg, tab)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            
            # --- Gradient Tracking ---
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            grad_norms.append(grad_norm)
            # -------------------------
            
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(labels)
        y_true_batches.append(labels.detach().cpu().numpy())
        y_score_batches.append(torch.sigmoid(logits).detach().cpu().numpy())

        percent = (batch_idx / total_batches) * 100
        print(
            f"\rEpoch {epoch}/{total_epochs} [{stage}] "
            f"batch {batch_idx}/{total_batches} ({percent:.1f}%)",
            end="",
            flush=True,
        )

    print()

    y_true = np.concatenate(y_true_batches, axis=0)
    y_score = np.concatenate(y_score_batches, axis=0)
    metrics = compute_metrics(y_true, y_score)
    avg_loss = total_loss / len(loader.dataset)
    
    extra_stats = {}
    if is_train and grad_norms:
        extra_stats["grad_norm_mean"] = float(np.mean(grad_norms))
        extra_stats["grad_norm_var"] = float(np.var(grad_norms))
    
    if str(device) != "cpu" and torch.cuda.is_available():
        extra_stats["gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    return avg_loss, metrics, extra_stats


def train_single(config: TrainConfig) -> Dict[str, float]:
    print(f"Using device: {config.device}")
    dataloaders = create_dataloaders(
        MultimodalDataConfig(
            processed_dir=config.processed_dir,
            batch_size=config.batch_size,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    )

    first_batch = next(iter(dataloaders["train"]))
    num_classes = first_batch["label"].shape[-1]
    model = build_model_from_batches(first_batch, num_classes=num_classes).to(config.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

    _optimizer_dir(config).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1_micro": [],
        "val_f1_micro": [],
        "train_precision_micro": [],
        "val_precision_micro": [],
        "train_recall_micro": [],
        "val_recall_micro": [],
        "train_grad_norm_mean": [],
        "train_grad_norm_var": [],
        "epoch_time_seconds": [],
        "gpu_memory_mb": [],
    }

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        print(f"Starting epoch {epoch}/{config.epochs}")
        train_loss, train_metrics, train_extra = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            torch.device(config.device),
            epoch,
            config.epochs,
            "train",
        )
        val_loss, val_metrics, val_extra = run_epoch(
            model,
            dataloaders["val"],
            criterion,
            None,
            torch.device(config.device),
            epoch,
            config.epochs,
            "val",
        )

        epoch_secs = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_f1={train_metrics['f1_micro']:.4f} val_f1={val_metrics['f1_micro']:.4f} | "
            f"grad_norm={train_extra.get('grad_norm_mean', 0.0):.4f} | "
            f"time={epoch_secs:.1f}s"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["train_f1_micro"].append(float(train_metrics["f1_micro"]))
        history["val_f1_micro"].append(float(val_metrics["f1_micro"]))
        history["train_precision_micro"].append(float(train_metrics["precision_micro"]))
        history["val_precision_micro"].append(float(val_metrics["precision_micro"]))
        history["train_recall_micro"].append(float(train_metrics["recall_micro"]))
        history["val_recall_micro"].append(float(val_metrics["recall_micro"]))
        
        history["train_grad_norm_mean"].append(train_extra.get("grad_norm_mean", 0.0))
        history["train_grad_norm_var"].append(train_extra.get("grad_norm_var", 0.0))
        history["epoch_time_seconds"].append(epoch_secs)
        history["gpu_memory_mb"].append(train_extra.get("gpu_memory_mb", 0.0))
        
        # Save every epoch
        save_checkpoint(
            model,
            config,
            num_classes,
            epoch,
            is_best=False,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            }
            save_checkpoint(
                model,
                config,
                num_classes,
                epoch,
                is_best=True,
            )

    test_loss, test_metrics = run_epoch(
        model,
        dataloaders["test"],
        criterion,
        None,
        torch.device(config.device),
        config.epochs,
        config.epochs,
        "test",
    )
    print(f"Test loss={test_loss:.4f} | test_f1={test_metrics['f1_micro']:.4f}")

    save_training_curves(history, config)
    save_epoch_table_csv(config.optimizer, history, _optimizer_dir(config))
    print_epoch_table(config.optimizer, history)

    results = {
        **best_metrics,
        "test_loss": float(test_loss),
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        "optimizer": config.optimizer,
    }
    return results


def train(config: TrainConfig) -> Dict[str, float] | Dict[str, Dict[str, float]]:
    if config.optimizer == "all":
        optimizers = ["adagrad", "adamw"]
        summary: Dict[str, Dict[str, float]] = {}
        for optimizer_name in optimizers:
            print(f"\n=== Optimizer study: {optimizer_name} ===")
            study_config = TrainConfig(
                processed_dir=config.processed_dir,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_fraction=config.validation_fraction,
                seed=config.seed,
                optimizer=optimizer_name,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                device=config.device,
                checkpoint_dir=config.checkpoint_dir,
            )
            summary[optimizer_name] = train_single(study_config)

        _optimizer_dir(config).mkdir(parents=True, exist_ok=True)
        with open(config.checkpoint_dir / "optimizer_study_results.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        import csv
        csv_path = _optimizer_dir(config) / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "optimizer", "epoch",
                "val_accuracy", "val_f1_micro",
                "val_recall_micro", "val_precision_micro",
                "train_loss", "val_loss",
                "train_f1_micro",
                "train_precision_micro",
                "train_recall_micro",
            ])
            for opt_name in optimizers:
                hist_file = config.checkpoint_dir / opt_name / f"training_history_{opt_name}.json"
                if hist_file.exists():
                    with open(hist_file, "r") as hf:
                        hist = json.load(hf)
                        for i in range(len(hist["epoch"])):
                            writer.writerow([
                                opt_name,
                                hist["epoch"][i],
                                hist["val_accuracy"][i],
                                hist["val_f1_micro"][i],
                                hist["val_recall_micro"][i],
                                hist["val_precision_micro"][i],
                                hist["train_loss"][i],
                                hist["val_loss"][i],
                                hist["train_f1_micro"][i],
                                hist["train_precision_micro"][i],
                                hist["train_recall_micro"][i],
                            ])
        print(f"Combined results saved -> {csv_path}")

        return summary

    return train_single(config)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the multimodal PTB-XL model.")
    parser.add_argument("--processed-dir", default="processed")
    #epoch is here
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="all", choices=["adamw", "adagrad", "all"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    return TrainConfig(
        processed_dir=Path(args.processed_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

if __name__ == "__main__":
    results = train(parse_args())
    print(results)
