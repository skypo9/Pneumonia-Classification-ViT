# Vision Transformer for Pneumonia Classification

A Vision Transformer (ViT) implementation for pneumonia detection in chest X-ray images with attention visualization capabilities.

## Performance Results

**Model**: ViT-Small trained on 5,856 chest X-ray images
- **Accuracy**: 90.38%
- **F1 Score**: 92.35%
- **AUC-ROC**: 95.63%
- **Sensitivity**: 92.82%
- **Specificity**: 86.32%

### Performance Visualizations

<table>
  <tr>
    <td align="center">
      <b>ROC Curve</b><br>
      <img src="images/roc_curve.png" alt="ROC Curve" width="300"/>
    </td>
    <td align="center">
      <b>Confusion Matrix</b><br>
      <img src="images/confusion_matrix.png" alt="Confusion Matrix" width="300"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Precision-Recall Curve</b><br>
      <img src="images/precision_recall_curve.png" alt="Precision-Recall" width="300"/>
    </td>
    <td align="center">
      <b>Threshold Analysis</b><br>
      <img src="images/threshold_analysis.png" alt="Threshold Analysis" width="300"/>
    </td>
  </tr>
</table>

## Key Features

- **Vision Transformer Models**: ViT-Small, ViT-Base, and ViT-Large configurations
- **Attention Visualization**: Multi-head attention maps and attention rollout
- **Medical AI Optimizations**: Class balancing and medical-specific augmentations
- **Real-time Inference**: Optimized for clinical deployment

## Quick Start

### 1. Setup Environment
```bash
python -m venv vit_env
vit_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Prepare Data

#### Dataset Information
This project uses the **Chest X-Ray Images (Pneumonia)** dataset for binary classification:
- **Dataset Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,856 chest X-ray images
- **Classes**: NORMAL and PNEUMONIA
- **Image Format**: JPEG files
- **Original Split**: Pre-divided into train/validation/test sets

#### How to Get the Dataset

**Option 1: Download from Kaggle**
1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (requires Kaggle account)
3. Extract the zip file to your project directory

**Option 2: Using Kaggle API**
```bash
# Install Kaggle API
pip install kaggle

# Download dataset (requires kaggle.json in ~/.kaggle/)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract the dataset
unzip chest-xray-pneumonia.zip
```

#### Directory Structure Setup
After downloading, organize your dataset as follows:
```
data/chest_xray_pneumonia/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/          # 1,341 normal X-ray images
│   │   └── PNEUMONIA/       # 3,875 pneumonia X-ray images
│   ├── val/
│   │   ├── NORMAL/          # 8 normal X-ray images
│   │   └── PNEUMONIA/       # 8 pneumonia X-ray images
│   └── test/
│       ├── NORMAL/          # 234 normal X-ray images
│       └── PNEUMONIA/       # 390 pneumonia X-ray images
```

**Important Notes:**
- The validation set is very small (16 images total) - the model may use train/test split internally
- Images are in JPEG format with varying resolutions
- The dataset is imbalanced (more pneumonia cases than normal)
- All images are pediatric chest X-rays (ages 1-5 years)

### 3. Training
```bash
# Quick training (2 epochs)
python main.py --epochs 2 --batch_size 16

# With attention visualization
python main.py --epochs 2 --generate_attention

# Different model sizes
python main.py --model_size small --epochs 2 --batch_size 32
```

### 4. Evaluation
```bash
# Evaluate existing model
python main.py --eval_only --checkpoint checkpoints/best_checkpoint.pth

# Generate attention maps
python main.py --eval_only --checkpoint checkpoints/best_checkpoint.pth --generate_attention
```

## Model Options

| Model | Parameters | Batch Size | Training Time |
|-------|------------|------------|---------------|
| ViT-Small | 22M | 32 | Fast |
| ViT-Base | 86M | 16 | Medium |
| ViT-Large | 307M | 8 | Slow |

## Project Structure

```
pneumonia_vit/
├── src/                    # Source code
│   ├── model.py           # ViT model implementation
│   ├── trainer.py         # Training logic
│   ├── evaluator.py       # Evaluation and metrics
│   └── attention_visualization.py
├── configs/               # Model configurations
├── main.py               # Main script
└── requirements.txt      # Dependencies
```

## Key Configuration Parameters

```yaml
training:
  learning_rate: 3e-4
  batch_size: 16
  num_epochs: 50          # Use 2 for quick testing
  warmup_epochs: 5
  weight_decay: 0.3
```

## Attention Visualization

The model provides interpretable attention maps showing:
- **Attention Rollout**: How the model focuses on lung regions
- **Multi-Head Attention**: Different attention patterns across heads
- **Layer-wise Evolution**: Attention development through the network

### Attention Examples

<table>
  <tr>
    <td align="center">
      <b>Attention Rollout</b><br>
      <img src="images/IM-0001-0001_attention_rollout.png" alt="Attention Rollout" width="300"/><br>
      <i>Model focus on pneumonia-affected regions</i>
    </td>
    <td align="center">
      <b>Multi-Head Attention</b><br>
      <img src="images/IM-0001-0001_multihead_attention.png" alt="Multi-Head Attention" width="300"/><br>
      <i>Different attention patterns across heads</i>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/IM-0003-0001_attention_rollout.png" alt="Attention Rollout 2" width="300"/><br>
      <i>Additional attention rollout example</i>
    </td>
    <td align="center">
      <img src="images/IM-0003-0001_multihead_attention.png" alt="Multi-Head Attention 2" width="300"/><br>
      <i>Diverse attention head patterns</i>
    </td>
  </tr>
</table>

## Clinical Applications

- **Explainable AI**: Attention maps for radiologist confidence
- **Real-time Processing**: Fast inference for clinical workflows  
- **Global Context**: Captures relationships between distant anatomical regions
- **Robust Performance**: Less sensitive to local artifacts

## Common Commands

```bash
# Fast training test
python main.py --model_size small --epochs 2 --batch_size 32

# Full training with visualization
python main.py --epochs 10 --generate_attention

# Custom configuration
python main.py --config configs/vit_base_config.yaml

# Evaluation only
python main.py --eval_only --checkpoint path/to/checkpoint.pth
```

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size or use ViT-Small
```bash
python main.py --model_size small --batch_size 8
```

**Missing Dependencies**: Install required packages
```bash
pip install timm>=0.9.0 matplotlib seaborn
```

## Academic Context

This implementation is part of a Master of Computer Science/Cybersecurity capstone project, demonstrating advanced machine learning techniques for medical AI applications.

**Note**: This system is for research and educational purposes only. Not intended for actual clinical diagnosis without proper validation and medical oversight.