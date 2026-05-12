"""
TorchViz Architecture Visualization Script
==========================================

This script generates a computational graph diagram of the Multimodal Cardiac Risk model.

Installation:
    pip install torchviz graphviz

Usage:
    python visualize_architecture_torchviz.py

Output Files:
    - multimodal_architecture.png
    - multimodal_architecture.pdf
    - multimodal_architecture.svg

Requirements:
    - graphviz (system package)
      On macOS:   brew install graphviz
      On Linux:   sudo apt-get install graphviz
      On Windows: Download from https://graphviz.org/download/
"""

import sys
from pathlib import Path

import torch
from torchviz import make_dot

# Add src to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.multimodal_model import MultimodalPTBXLNet, ModelConfig


def visualize_full_model():
    """Generate full computational graph visualization."""
    print("=" * 70)
    print("TORCHVIZ: Full Model Architecture Visualization")
    print("=" * 70)
    
    # Initialize model
    print("\n[1/5] Initializing model...")
    model = MultimodalPTBXLNet(tabular_dim=25, config=ModelConfig(num_classes=5))
    model.eval()
    
    # Create dummy inputs matching actual data shapes
    print("[2/5] Creating dummy inputs...")
    ecg_input = torch.randn(1, 12, 5000)  # (batch=1, channels=12, timesteps=5000)
    tab_input = torch.randn(1, 25)         # (batch=1, tabular_features=25)
    
    print(f"    ECG input shape:     {tuple(ecg_input.shape)}")
    print(f"    Tabular input shape: {tuple(tab_input.shape)}")
    
    # Forward pass
    print("[3/5] Running forward pass...")
    with torch.no_grad():
        output = model(ecg_input, tab_input)
    
    print(f"    Output shape: {tuple(output.shape)}")
    print(f"    Output (logits): {output}")
    
    # Generate computational graph
    print("[4/5] Generating computational graph...")
    graph = make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True
    )
    
    # Customize graph appearance
    graph.graph_attr.update(
        rankdir='TB',           # Top-to-Bottom layout
        size='16,24',           # Large canvas
        dpi='300',              # High resolution
        fontsize='10',
        compound='true',
        overlap='false'
    )
    
    # Save in multiple formats
    print("[5/5] Saving visualizations...")
    output_dir = Path("./architecture_diagrams")
    output_dir.mkdir(exist_ok=True)
    
    base_path = output_dir / "multimodal_architecture"
    
    # PNG (best for viewing)
    print(f"    → Saving PNG: {base_path}.png")
    graph.render(str(base_path), format='png', cleanup=True)
    
    # PDF (for documents/papers)
    print(f"    → Saving PDF: {base_path}.pdf")
    graph.render(str(base_path), format='pdf', cleanup=True)
    
    # SVG (scalable vector)
    print(f"    → Saving SVG: {base_path}.svg")
    graph.render(str(base_path), format='svg', cleanup=True)
    
    print("\n" + "=" * 70)
    print("✓ Visualizations saved to: architecture_diagrams/")
    print("=" * 70)


def visualize_ecg_encoder_only():
    """Generate visualization of just the ECG encoder."""
    print("\n" + "=" * 70)
    print("TORCHVIZ: ECG Encoder Visualization")
    print("=" * 70)
    
    from src.multimodal_model import ECGEncoder
    
    print("\n[1/5] Initializing ECG encoder...")
    ecg_encoder = ECGEncoder(in_channels=12, embedding_dim=128, num_heads=4)
    ecg_encoder.eval()
    
    print("[2/5] Creating dummy ECG input...")
    ecg_input = torch.randn(1, 12, 5000)
    print(f"    ECG input shape: {tuple(ecg_input.shape)}")
    
    print("[3/5] Running forward pass...")
    with torch.no_grad():
        output = ecg_encoder(ecg_input)
    print(f"    Output shape: {tuple(output.shape)}")
    
    print("[4/5] Generating computational graph...")
    graph = make_dot(
        output,
        params=dict(ecg_encoder.named_parameters()),
        show_attrs=True
    )
    
    graph.graph_attr.update(rankdir='TB', size='14,20', dpi='300')
    
    print("[5/5] Saving visualization...")
    output_dir = Path("./architecture_diagrams")
    output_dir.mkdir(exist_ok=True)
    base_path = output_dir / "ecg_encoder_only"
    
    graph.render(str(base_path), format='png', cleanup=True)
    graph.render(str(base_path), format='pdf', cleanup=True)
    
    print(f"    → Saved to: {base_path}.png")
    print("=" * 70)


def visualize_tabular_encoder_only():
    """Generate visualization of just the Tabular encoder."""
    print("\n" + "=" * 70)
    print("TORCHVIZ: Tabular Encoder Visualization")
    print("=" * 70)
    
    from src.multimodal_model import TabularEncoder
    
    print("\n[1/5] Initializing Tabular encoder...")
    tab_encoder = TabularEncoder(input_dim=25, embedding_dim=128, dropout=0.2)
    tab_encoder.eval()
    
    print("[2/5] Creating dummy tabular input...")
    tab_input = torch.randn(1, 25)
    print(f"    Tabular input shape: {tuple(tab_input.shape)}")
    
    print("[3/5] Running forward pass...")
    with torch.no_grad():
        output = tab_encoder(tab_input)
    print(f"    Output shape: {tuple(output.shape)}")
    
    print("[4/5] Generating computational graph...")
    graph = make_dot(
        output,
        params=dict(tab_encoder.named_parameters()),
        show_attrs=True
    )
    
    graph.graph_attr.update(rankdir='TB', size='10,12', dpi='300')
    
    print("[5/5] Saving visualization...")
    output_dir = Path("./architecture_diagrams")
    output_dir.mkdir(exist_ok=True)
    base_path = output_dir / "tabular_encoder_only"
    
    graph.render(str(base_path), format='png', cleanup=True)
    graph.render(str(base_path), format='pdf', cleanup=True)
    
    print(f"    → Saved to: {base_path}.png")
    print("=" * 70)


def print_model_summary():
    """Print detailed model summary with layer information."""
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    model = MultimodalPTBXLNet(tabular_dim=25, config=ModelConfig(num_classes=5))
    
    print("\n" + model.__str__())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 70)
    print(f"Total Parameters:       {total_params:,}")
    print(f"Trainable Parameters:   {trainable_params:,}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        print_model_summary()
        visualize_full_model()
        visualize_ecg_encoder_only()
        visualize_tabular_encoder_only()
        
        print("\n" + "🎉 " * 20)
        print("SUCCESS! All visualizations generated.")
        print("Check the 'architecture_diagrams/' folder for output files.")
        print("🎉 " * 20 + "\n")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Missing dependency")
        print(f"\nInstall required packages:")
        print(f"    pip install torchviz graphviz torch")
        print(f"\nAlso install graphviz system package:")
        print(f"    macOS:  brew install graphviz")
        print(f"    Linux:  sudo apt-get install graphviz")
        print(f"    Windows: Download from https://graphviz.org/download/")
        print(f"\nOriginal error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
