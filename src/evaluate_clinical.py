import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from scipy.stats import wilcoxon

from multimodal_model import MultimodalClassifier
from multimodal_data import create_dataloaders, MultimodalDataConfig

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def evaluate_calibration(y_true, y_prob):
    results = {}
    n_classes = y_true.shape[1]
    
    ece_scores = []
    brier_scores = []
    
    for i in range(n_classes):
        ece = expected_calibration_error(y_true[:, i], y_prob[:, i])
        brier = brier_score_loss(y_true[:, i], y_prob[:, i])
        ece_scores.append(ece)
        brier_scores.append(brier)
        
    results["mean_ece"] = float(np.mean(ece_scores))
    results["mean_brier"] = float(np.mean(brier_scores))
    return results

def statistical_testing(model1_probs, model2_probs, y_true):
    # Wilcoxon signed-rank test on squared errors
    n_classes = y_true.shape[1]
    p_values = []
    
    for i in range(n_classes):
        err1 = (model1_probs[:, i] - y_true[:, i])**2
        err2 = (model2_probs[:, i] - y_true[:, i])**2
        
        # Only run if differences exist
        if np.any(err1 != err2):
            stat, p = wilcoxon(err1, err2)
            p_values.append(p)
            
    return {"mean_p_value": float(np.mean(p_values)) if p_values else 1.0}

def predict_all(model, loader, device):
    model.eval()
    y_true_all = []
    y_prob_all = []
    
    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device)
            tab = batch["tab"].to(device)
            labels = batch["label"].cpu().numpy()
            
            logits = model(ecg, tab)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            y_true_all.append(labels)
            y_prob_all.append(probs)
            
    return np.concatenate(y_true_all, axis=0), np.concatenate(y_prob_all, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Clinical Evaluation")
    parser.add_argument("--processed_dir", type=str, default="processed")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_multimodal_ptbxl_adam.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    print("Loading data...")
    config = MultimodalDataConfig(processed_dir=args.processed_dir, batch_size=64)
    loaders = create_dataloaders(config)
    val_loader = loaders["val"]
    
    first_batch = next(iter(val_loader))
    num_classes = first_batch["label"].shape[-1]
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = MultimodalClassifier(num_classes=num_classes).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using untrained model for testing.")
        
    print("Generating predictions...")
    y_true, y_prob = predict_all(model, val_loader, device)
    
    print("\n--- Calibration Metrics ---")
    calib_results = evaluate_calibration(y_true, y_prob)
    print(f"Mean Expected Calibration Error (ECE): {calib_results['mean_ece']:.4f}")
    print(f"Mean Brier Score: {calib_results['mean_brier']:.4f}")
    
    # Example statistical testing against dummy/naive predictions
    print("\n--- Statistical Testing (vs Naive Baseline) ---")
    naive_probs = np.ones_like(y_prob) * y_true.mean(axis=0)
    stat_results = statistical_testing(y_prob, naive_probs, y_true)
    print(f"Wilcoxon P-Value vs Naive Baseline: {stat_results['mean_p_value']:.4e}")
    
    print("\nClinical Evaluation Complete.")

if __name__ == "__main__":
    main()
