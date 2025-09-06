"""
Training module for Vision Transformer pneumonia classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import mixup_data, cutmix_data


class MixupLoss(nn.Module):
    """Mixup loss function for ViT training"""
    
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)


class CosineWarmupScheduler:
    """Cosine annealing with warmup for ViT training"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


class ViTTrainer:
    """
    Trainer class for Vision Transformer pneumonia classification
    """
    
    def __init__(self, 
                 model,
                 config,
                 train_loader,
                 val_loader,
                 device: str = "cuda"):
        """
        Initialize trainer
        
        Args:
            model: ViT model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup logging
        self.setup_logging()
        
        # Setup loss function
        self.setup_loss_function()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup scheduler
        self.setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
        
        # Create output directories
        self.setup_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"✅ ViT Trainer initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
    
    def setup_loss_function(self):
        """Setup loss function with class balancing"""
        if self.config.training.class_balancing:
            # Calculate class weights from training data
            pos_weight = self._calculate_pos_weight()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Mixup loss wrapper
        self.mixup_criterion = MixupLoss(self.criterion)
    
    def _calculate_pos_weight(self) -> torch.Tensor:
        """Calculate positive class weight for balanced training"""
        pos_count = 0
        neg_count = 0
        
        for batch in self.train_loader:
            labels = batch['label']
            pos_count += torch.sum(labels == 1).item()
            neg_count += torch.sum(labels == 0).item()
        
        pos_weight = torch.tensor(neg_count / pos_count).to(self.device)
        self.logger.info(f"Calculated pos_weight: {pos_weight:.3f}")
        
        return pos_weight
    
    def setup_optimizer(self):
        """Setup optimizer with ViT-specific settings"""
        # Separate parameters for different learning rates
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Use different learning rates for backbone and classifier
        param_groups = [
            {'params': backbone_params, 'lr': self.config.training.learning_rate},
            {'params': classifier_params, 'lr': self.config.training.learning_rate * 10}  # Higher LR for classifier
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.training.use_scheduler:
            if self.config.training.scheduler_type == "cosine_warmup":
                self.scheduler = CosineWarmupScheduler(
                    self.optimizer,
                    warmup_epochs=self.config.training.warmup_epochs,
                    total_epochs=self.config.training.num_epochs,
                    min_lr=self.config.training.min_lr
                )
            elif self.config.training.scheduler_type == "reduce_on_plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=self.config.training.scheduler_factor,
                    patience=self.config.training.scheduler_patience,
                    min_lr=self.config.training.min_lr
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
    
    def setup_directories(self):
        """Create output directories"""
        self.output_dir = Path(self.config.experiment.output_dir)
        self.log_dir = Path(self.config.experiment.log_dir)
        self.checkpoint_dir = Path(self.config.experiment.checkpoint_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Apply mixup/cutmix augmentation
            if self.config.training.use_mixup and np.random.random() > 0.5:
                if np.random.random() > 0.5:
                    # Mixup
                    mixed_images, y_a, y_b, lam = mixup_data(
                        images, labels, self.config.training.mixup_alpha
                    )
                else:
                    # CutMix
                    mixed_images, y_a, y_b, lam = cutmix_data(
                        images, labels, self.config.training.cutmix_alpha
                    )
                
                # Forward pass
                logits = self.model(mixed_images)
                loss = self.mixup_criterion(logits.squeeze(), y_a, y_b, lam)
            else:
                # Standard training
                logits = self.model(images)
                loss = self.criterion(logits.squeeze(), labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_val
                )
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_samples += images.size(0)
            
            # Store predictions for metrics (only for non-mixup batches)
            if not self.config.training.use_mixup or np.random.random() <= 0.5:
                with torch.no_grad():
                    probs = torch.sigmoid(logits.squeeze())
                    predictions.extend(probs.cpu().numpy())
                    targets.extend(labels.cpu().numpy())
            
            # Log batch progress
            if batch_idx % self.config.experiment.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        
        metrics = {}
        if len(predictions) > 0:
            pred_labels = (np.array(predictions) > 0.5).astype(int)
            target_labels = np.array(targets)
            
            metrics = {
                'accuracy': accuracy_score(target_labels, pred_labels),
                'precision': precision_score(target_labels, pred_labels, zero_division=0),
                'recall': recall_score(target_labels, pred_labels, zero_division=0),
                'f1': f1_score(target_labels, pred_labels, zero_division=0),
                'auc': roc_auc_score(target_labels, predictions) if len(np.unique(target_labels)) > 1 else 0.0
            }
        
        return {'loss': avg_loss, **metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits.squeeze(), labels)
                
                total_loss += loss.item()
                
                # Store predictions
                probs = torch.sigmoid(logits.squeeze())
                predictions.extend(probs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        
        pred_labels = (np.array(predictions) > 0.5).astype(int)
        target_labels = np.array(targets)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(target_labels, pred_labels),
            'precision': precision_score(target_labels, pred_labels, zero_division=0),
            'recall': recall_score(target_labels, pred_labels, zero_division=0),
            'f1': f1_score(target_labels, pred_labels, zero_division=0),
            'auc': roc_auc_score(target_labels, predictions) if len(np.unique(target_labels)) > 1 else 0.0
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config.to_dict()
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Saved best checkpoint with {self.config.evaluation.threshold_metric}: {self.best_val_score:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        
        self.logger.info(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, List]:
        """Full training loop"""
        self.logger.info("🚀 Starting ViT training...")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Update learning rate
            if self.scheduler and isinstance(self.scheduler, CosineWarmupScheduler):
                current_lr = self.scheduler.step()
                self.logger.info(f"Epoch {epoch}: Learning rate = {current_lr:.6f}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            self.val_history.append(val_metrics)
            
            # Update scheduler (for ReduceLROnPlateau)
            if self.scheduler and not isinstance(self.scheduler, CosineWarmupScheduler):
                self.scheduler.step(val_metrics[self.config.evaluation.threshold_metric])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val {self.config.evaluation.threshold_metric.upper()}: {val_metrics[self.config.evaluation.threshold_metric]:.4f}"
            )
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            for metric_name, metric_value in val_metrics.items():
                if metric_name != 'loss':
                    self.writer.add_scalar(f'Metrics/Val_{metric_name}', metric_value, epoch)
            
            # Check for improvement
            current_score = val_metrics[self.config.evaluation.threshold_metric]
            is_best = current_score > self.best_val_score
            
            if is_best:
                self.best_val_score = current_score
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if epoch % self.config.experiment.save_interval == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                self.logger.info(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Training completed
        training_time = time.time() - start_time
        self.logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        self.logger.info(f"🏆 Best validation {self.config.evaluation.threshold_metric}: {self.best_val_score:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_score': self.best_val_score,
            'training_time': training_time
        }


if __name__ == "__main__":
    print("✅ ViT training module loaded successfully!")
    print("Use ViTTrainer class to train Vision Transformer models")
