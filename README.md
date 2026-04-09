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

## How to test a new ECG sample

Use the helper script in [Testing/prepare_prediction_input.py](Testing/prepare_prediction_input.py) to convert a record from the Testing folder into a prediction-ready JSON or NPZ file.

### 1) Create the input file

The helper script reads:
- the ECG waveform from .dat/.hea
- clinical features from ptbxl_database.csv
- raw SCP codes from scp_statements.csv
- optional true labels for correctness checking

Example:

```bash
python Testing/prepare_prediction_input.py --ecg-id 1 --output Testing/sample_input.json
```

### 2) Run prediction

The prediction script can use one checkpoint or every checkpoint in [checkpoints](checkpoints):

```bash
python src/predict_multimodal.py --input-json Testing/sample_input.json --checkpoint-dir checkpoints
```

If you want just one model, pass a single checkpoint:

```bash
python src/predict_multimodal.py --input-json Testing/sample_input.json --checkpoint checkpoints/best_multimodal_ptbxl_adamw.pt
```

### 3) Check whether the prediction is correct

If the input file includes labels, the script prints:
- predicted labels
- true labels
- exact match
- label-wise accuracy

## Model artifacts

- [checkpoints](checkpoints): trained PyTorch .pt checkpoints, optimizer comparison metrics, and plots.
- [processed](processed): saved preprocessed train/test arrays.
- [Testing](Testing): raw test records, the sample generator script, and a sample JSON file for inference.

## Notes

- The ECG branch uses the 500 Hz records so the input shape is 12 × 5000.
- The preprocessing step saves arrays at the end of execution so they can be reused later.
- For inference, the model exposes sigmoid probabilities through `predict_proba()`.
- The .pt files in [checkpoints](checkpoints) are model checkpoints containing learned weights and configuration.
