# Vision Transformer for Pneumonia Classification

A Vision Transformer (ViT) implementation for pneumonia detection in chest X-ray images with attention visualization capabilities.

## Performance Results

**Model**: ViT-Small trained on 5,856 chest X-ray images
- **Accuracy**: 90.38%
- **F1 Score**: 92.35%
- **AUC-ROC**: 95.63%
- **Sensitivity**: 92.82%
- **Specificity**: 86.32%

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
Organize your chest X-ray dataset as:
```
data/chest_xray_pneumonia/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
```

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