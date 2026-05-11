"""Visualizes 1D clinical importance on raw ECG using Saliency mapping."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal_data import MultimodalDataConfig, create_dataloaders
from src.multimodal_model import build_model_from_batches

def load_model_and_data(checkpoint_path: Path, data_dir: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    num_classes = checkpoint["num_classes"]

    dataloaders = create_dataloaders(
        MultimodalDataConfig(processed_dir=data_dir, batch_size=32, validation_fraction=0.1, seed=42)
    )
    first_batch = next(iter(dataloaders["test"]))
    model = build_model_from_batches(first_batch, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataloaders["test"]

def plot_colored_ecg(ecg_signal, attention_weights, save_path, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = np.arange(len(ecg_signal))
    points = np.array([x, ecg_signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(attention_weights.min(), attention_weights.max())
    lc = LineCollection(segments, cmap='Reds', norm=norm)
    lc.set_array(attention_weights)
    lc.set_linewidth(2)

    line = ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(title)
    ax.set_xlabel("Time (Samples, 100 Hz)")
    ax.set_ylabel("Amplitude")
    fig.colorbar(line, ax=ax, label="Clinical Importance (Gradient Saliency)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_1d_ecg_saliency(model, test_loader, output_dir: Path, opt_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Let's shuffle the dataloader implicitly by just running multiple batches if needed,
    # or just picking random slices so we aren't always looking at "CD" patients
    all_ecgs, all_tabs, all_labels = [], [], []
    for batch in test_loader:
        all_ecgs.append(batch["ecg"])
        all_tabs.append(batch["tab"])
        all_labels.append(batch["label"])
        if len(all_labels) > 3: # get a few batches to ensure label diversity
            break
            
    ecg = torch.cat(all_ecgs, dim=0)
    tab = torch.cat(all_tabs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    ecg_raw = ecg.detach().numpy()
    
    # We use Gradient Saliency to project the model's prediction 
    # back onto the exact 1000 temporal indices of the ECG limit
    ecg.requires_grad = True
    
    logits = model(ecg, tab)
    
    # We trace backward from the most confident predictions
    preds = logits.max(dim=1).values
    preds.sum().backward()
    
    # Take absolute gradients and mean across the 12 leads to find important timesteps
    saliency_map = ecg.grad.abs().mean(dim=1) # Shape: [Batch, 1000]
    
    # Slightly smooth the saliency map for visual clarity
    saliency_np = saliency_map.numpy()
    
    class_names = ["NORM (Normal)", "MI (Myocardial Infarction)", "STTC (ST/T Change)", "CD (Conduction Disturbance)", "HYP (Hypertrophy)"]
    
    # To get variety, let's pick 5 patients from the accumulated batches pool, e.g., jumping by 15
    indices_to_plot = [0, 15, 30, 45, 60]
    for n, idx in enumerate(indices_to_plot):
        if idx >= len(labels):
             continue
        patient_labels = [c for j, c in enumerate(class_names) if labels[idx, j] == 1]
        label_str = ", ".join(patient_labels) if patient_labels else "Unknown"
        
        lead_idx = 0 # Visualizing Lead 0 (Lead I)
        ecg_signal = ecg_raw[idx, lead_idx, :]
        
        # Min-Max scale the saliency for this specific patient so colors pop
        attn_signal = saliency_np[idx, :]
        attn_min, attn_max = attn_signal.min(), attn_signal.max()
        if attn_max > attn_min:
            attn_signal = (attn_signal - attn_min) / (attn_max - attn_min)
        
        save_path = output_dir / f"ecg_importance_{opt_name}_patient_{n}.png"
        title = f"1D ECG Regions of Importance ({opt_name}) - Patient {n} | Diagnoses: [{label_str}]"
        
        plot_colored_ecg(ecg_signal, attn_signal, save_path, title)
    
    print(f"[{opt_name}] Saved 1D ECG colored line plots to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--output_dir", type=str, default="results/attention_maps")
    args = parser.parse_args()
    
    optimizers = ["adam", "adamw", "adagrad", "sgd", "rmsprop"]
    base_checkpoints = Path("checkpoints")
    
    for opt in optimizers:
        ckpt_path = base_checkpoints / opt / "best.pt"
        if not ckpt_path.exists():
            print(f"Skipping {opt}: {ckpt_path} not found.")
            continue
            
        print(f"\nProcessing {opt}...")
        model, test_loader = load_model_and_data(ckpt_path, Path(args.data_dir))
        opt_out_dir = Path(args.output_dir) / opt
        visualize_1d_ecg_saliency(model, test_loader, opt_out_dir, opt)

if __name__ == '__main__':
    main()