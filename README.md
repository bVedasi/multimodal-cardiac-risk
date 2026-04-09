# Multimodal PTB-XL Cardiac Risk Pipeline

## Architecture

ECG → CNN → Residual Blocks → Transformer Encoder → ECG Embedding  
Metadata → MLP → Metadata Embedding  
→ Bidirectional Cross-Attention  
→ Gated Fusion  
→ Residual Addition  
→ Dense Projection  
→ Dropout  
→ Classifier  
→ Sigmoid Output

The current pipeline also uses SCP codes as an explicit third input modality.

## What each Python file does

- [preprocess.py](preprocess.py): loads the PTB-XL CSV files, parses SCP codes, builds tabular features, loads ECG waveforms, creates train/test splits, and saves the preprocessed arrays into [processed](processed).
- [src/multimodal_data.py](src/multimodal_data.py): loads the saved arrays and creates PyTorch datasets and DataLoaders.
- [src/multimodal_model.py](src/multimodal_model.py): defines the ECG encoder, metadata encoder, SCP encoder, bidirectional cross-attention block, gated fusion, residual addition, dense projection, and multilabel classifier.
- [src/train_multimodal.py](src/train_multimodal.py): runs training, validation, checkpoint saving, and test evaluation.

## Workflow

1. Run preprocessing to create the saved arrays under [processed](processed).
2. Load the saved arrays with [src/multimodal_data.py](src/multimodal_data.py).
3. Train the model with [src/train_multimodal.py](src/train_multimodal.py).
4. Evaluate on the held-out test split.

## Notes

- The ECG branch uses the 500 Hz records so the input shape is 12 × 5000.
- The preprocessing step saves arrays at the end of execution so they can be reused later.
- For inference, the model exposes sigmoid probabilities through `predict_proba()`.
