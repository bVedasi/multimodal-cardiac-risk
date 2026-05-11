"""Visualizes cross-attention weights for interpretability."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal_data import MultimodalDataConfig, create_dataloaders
from src.multimodal_model import build_model_from_batches


def load_model_and_data(checkpoint_path: Path, data_dir: Path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    num_classes = checkpoint["num_classes"]

    dataloaders = create_dataloaders(
        MultimodalDataConfig(
            processed_dir=data_dir,
            batch_size=32,
            validation_fraction=0.1,
            seed=42,
        )
    )
    first_batch = next(iter(dataloaders["test"]))
    model = build_model_from_batches(first_batch, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, dataloaders["test"]


def visualize_collective_attention(model, test_loader, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ecg_to_meta = []
    all_labels = []
    
    print("Collecting attention weights across test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ecg = batch["ecg"]
            tab = batch["tab"]
            labels = batch["label"]
            
            # Forward pass to get attention weights
            logits, attn_weights = model.forward_with_attention(ecg, tab)
            
            # Extract ECG to Meta attention 
            # Dim: [batch_size, num_heads, ecg_seq, meta_seq]
            # We average across the 'heads' (dim=1) to get a single 2D map per patient
            ecg_to_meta = attn_weights["ecg_to_meta"].mean(dim=1).numpy()
            
            all_ecg_to_meta.append(ecg_to_meta)
            all_labels.append(labels.numpy())

    all_ecg_to_meta = np.concatenate(all_ecg_to_meta, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # PTB-XL Superclasses
    class_names = ["NORM (Normal)", "MI (Myocardial Infarction)", "STTC (ST/T Change)", "CD (Conduction Disturbance)", "HYP (Hypertrophy)"]
    
    print("\nGenerating class-wise aggregated attention heatmaps...")
    for class_idx, class_name in enumerate(class_names):
        # Find patients positive for this exact class
        class_mask = all_labels[:, class_idx] == 1
        num_patients = np.sum(class_mask)
        
        if num_patients == 0:
            continue
            
        # Average the attention maps across all patients in this disease group
        class_attention = all_ecg_to_meta[class_mask].mean(axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(class_attention, cmap="viridis", cbar_kws={'label': 'Attention Score'})
        plt.title(f"Collective Attention Map: {class_name}\n(Averaged over {num_patients} patients)")
        plt.xlabel("Tabular Metadata Features (Latent Dimensions)")
        plt.ylabel("ECG Waveform (Downsampled Timesteps)")
        plt.tight_layout()
        
        filename = class_name.split()[0]
        save_path = output_dir / f"collective_attention_{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    # Generate one single random patient sample to show the Per-Patient Case study
    plt.figure(figsize=(10, 8))
    sns.heatmap(all_ecg_to_meta[0], cmap="magma", cbar_kws={'label': 'Attention Score'})
    plt.title("Per-Patient Case Study (Single Patient Attention)")
    plt.xlabel("Tabular Metadata Features")
    plt.ylabel("ECG Timesteps")
    plt.tight_layout()
    plt.savefig(output_dir / "single_patient_attention_example.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'single_patient_attention_example.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interpretable attention heatmaps.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/adam/best.pt", help="Path to a trained model best.pt file")
    parser.add_argument("--data_dir", type=str, default="processed", help="Path to processed dataset folder")
    parser.add_argument("--output_dir", type=str, default="results/attention_maps", help="Where to save the heatmap PNG files")
    args = parser.parse_args()
    
    checkpoint_file = Path(args.checkpoint)
    if not checkpoint_file.exists():
        print(f"Error: Could not find model checkpoint at '{checkpoint_file}'.")
        print("Please train the model first by running one of the training scripts!")
        sys.exit(1)
        
    model, test_loader = load_model_and_data(checkpoint_file, Path(args.data_dir))
    visualize_collective_attention(model, test_loader, Path(args.output_dir))
