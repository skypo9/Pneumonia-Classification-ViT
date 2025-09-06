"""
Main script for Vision Transformer pneumonia classification
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config, get_vit_base_config, get_vit_small_config, get_vit_large_config
from model import create_vit_model, count_parameters
from data import create_data_loaders
from trainer import ViTTrainer
from evaluator import ViTEvaluator


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    return device


def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Pneumonia Classification")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"],
                       help="ViT model size")
    
    # Data
    parser.add_argument("--data_root", type=str, 
                       default="../pneumonia_classification/data/chest_xray_pneumonia",
                       help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size override")
    
    # Training
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Evaluation
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
    parser.add_argument("--generate_attention", action="store_true", default=True,
                       help="Generate attention visualizations")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="vit_pneumonia", 
                       help="Experiment name")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    print("🔬 Vision Transformer Pneumonia Classification")
    print("=" * 60)
    
    # Setup device
    if args.device == "auto":
        device = setup_device()
    else:
        device = args.device
    
    # Load or create configuration
    if args.config:
        config = Config.load(args.config)
        print(f"📋 Loaded configuration from: {args.config}")
    else:
        # Use predefined configuration based on model size
        if args.model_size == "small":
            config = get_vit_small_config()
        elif args.model_size == "large":
            config = get_vit_large_config()
        else:
            config = get_vit_base_config()
        
        print(f"📋 Using default ViT-{args.model_size.title()} configuration")
    
    # Apply command line overrides
    if args.data_root:
        config.data.data_root = args.data_root
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    if args.generate_attention:
        config.evaluation.generate_attention_maps = args.generate_attention
    
    # Update paths
    config.update_paths(str(Path.cwd()))
    
    # Set random seed
    set_seed(config.experiment.random_seed)
    print(f"🎲 Set random seed to: {config.experiment.random_seed}")
    
    # Create model
    print("🏗️  Creating Vision Transformer model...")
    model = create_vit_model(config.model)
    
    # Print model information
    param_counts = count_parameters(model)
    print(f"📊 Model: {config.model.model_name}")
    print(f"📊 Total parameters: {param_counts['total']:,}")
    print(f"📊 Trainable parameters: {param_counts['trainable']:,}")
    print(f"📊 Frozen parameters: {param_counts['frozen']:,}")
    
    # Load checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loaders
    print("📂 Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)
        print(f"✅ Data loaders created successfully!")
        print(f"   📈 Training batches: {len(train_loader)}")
        print(f"   📊 Validation batches: {len(val_loader)}")
        print(f"   🧪 Test batches: {len(test_loader)}")
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"Please check the data path: {config.data.data_root}")
        return
    
    if args.eval_only:
        # Evaluation only
        print("\n🔍 Running evaluation only...")
        
        # Load checkpoint for evaluation
        if args.checkpoint:
            print(f"📥 Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create evaluator and run evaluation
        evaluator = ViTEvaluator(model, config, device)
        evaluation_results = evaluator.evaluate_model(test_loader)
        
        # Print summary
        basic_metrics = evaluation_results['basic_metrics']
        print("\n📋 Evaluation Summary:")
        print(f"   🎯 Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"   🎯 F1 Score: {basic_metrics['f1_score']:.4f}")
        print(f"   🎯 AUC-ROC: {basic_metrics['auc_roc']:.4f}")
        print(f"   🎯 Sensitivity: {basic_metrics['sensitivity']:.4f}")
        print(f"   🎯 Specificity: {basic_metrics['specificity']:.4f}")
        
    else:
        # Training
        print("\n🚀 Starting training...")
        
        # Create trainer
        trainer = ViTTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Train model
        training_results = trainer.train()
        
        print("\n✅ Training completed!")
        print(f"🏆 Best validation {config.evaluation.threshold_metric}: {training_results['best_score']:.4f}")
        print(f"⏱️  Training time: {training_results['training_time']:.2f} seconds")
        
        # Load best checkpoint for evaluation
        best_checkpoint_path = Path(config.experiment.checkpoint_dir) / "best_checkpoint.pth"
        if best_checkpoint_path.exists():
            print(f"📥 Loading best checkpoint for evaluation...")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluation
        print("\n🔍 Running final evaluation...")
        evaluator = ViTEvaluator(model, config, device)
        evaluation_results = evaluator.evaluate_model(test_loader)
        
        # Print final summary
        basic_metrics = evaluation_results['basic_metrics']
        print("\n🎉 Final Results:")
        print(f"   🎯 Test Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"   🎯 Test F1 Score: {basic_metrics['f1_score']:.4f}")
        print(f"   🎯 Test AUC-ROC: {basic_metrics['auc_roc']:.4f}")
        print(f"   🎯 Test Sensitivity: {basic_metrics['sensitivity']:.4f}")
        print(f"   🎯 Test Specificity: {basic_metrics['specificity']:.4f}")
        print(f"   ⚡ Inference Time: {basic_metrics['avg_inference_time']:.4f}s per image")
        
        if config.evaluation.generate_attention_maps:
            attention_results = evaluation_results['attention_results']
            print(f"   🎨 Attention Maps: {attention_results['num_visualizations']} generated")
    
    print(f"\n📁 Results saved to: {config.experiment.output_dir}")
    print("✅ Vision Transformer pneumonia classification completed!")


if __name__ == "__main__":
    main()
