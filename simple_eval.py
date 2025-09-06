"""
Simple evaluation script for ViT pneumonia classification results
"""
import sys
sys.path.append('src')

import torch
from pathlib import Path
from config import get_vit_small_config
from model import create_vit_model
from data import create_data_loaders
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_vit_model():
    """Simple evaluation of trained ViT model"""
    
    print("🔍 ViT Pneumonia Classification - Final Evaluation")
    print("=" * 60)
    
    # Setup
    device = "cpu"
    config = get_vit_small_config()
    config.data.data_root = "../pneumonia_classification/data/chest_xray_pneumonia"
    config.training.batch_size = 8
    
    # Load model
    model = create_vit_model(config.model)
    checkpoint_path = "checkpoints/best_checkpoint.pth"
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        print(f"📊 Best validation F1: {checkpoint['best_val_score']:.4f}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load data
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Evaluate model
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print("\n🧪 Evaluating on test set...")
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            logits = model(images)
            probabilities = torch.sigmoid(logits.squeeze()).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc = roc_auc_score(all_labels, all_probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("\n🎉 Vision Transformer (ViT-Small) - Final Results:")
    print(f"   📊 Test Samples: {len(all_labels)}")
    print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   🎯 Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
    print(f"   🎯 Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"   🎯 F1 Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"   🎯 AUC-ROC: {auc:.4f} ({auc*100:.2f}%)")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"   True Negative (Normal correctly identified): {tn}")
    print(f"   False Positive (Normal incorrectly as Pneumonia): {fp}")
    print(f"   False Negative (Pneumonia incorrectly as Normal): {fn}")
    print(f"   True Positive (Pneumonia correctly identified): {tp}")
    
    print(f"\n🏥 Clinical Relevance:")
    print(f"   🔍 Missed Pneumonia Cases (False Negatives): {fn}")
    print(f"   ⚠️  False Alarms (False Positives): {fp}")
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🤖 Model Information:")
    print(f"   📊 Architecture: {config.model.model_name}")
    print(f"   📊 Total Parameters: {param_count:,}")
    print(f"   📊 Trainable Parameters: {trainable_params:,}")
    print(f"   🎲 Training Epochs: 2 (as requested)")
    print(f"   💾 Batch Size: {config.training.batch_size}")
    
    print("\n✅ ViT Evaluation completed successfully!")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'specificity': specificity,
        'confusion_matrix': cm.tolist()
    }

if __name__ == "__main__":
    evaluate_vit_model()
