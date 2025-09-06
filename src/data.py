"""
Data loading and preprocessing for ViT pneumonia classification
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
from sklearn.model_selection import train_test_split


class ChestXrayDataset(Dataset):
    """
    Dataset class for chest X-ray pneumonia classification
    Compatible with ViT preprocessing requirements
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = "train",
                 transform: Optional[transforms.Compose] = None,
                 preprocessing_type: str = "vit_standard"):
        """
        Initialize dataset
        
        Args:
            data_root: Root directory containing chest_xray data
            split: Dataset split ("train", "val", "test")
            transform: PyTorch transforms
            preprocessing_type: Type of preprocessing ("vit_standard", "raw", "histogram_matching")
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.preprocessing_type = preprocessing_type
        
        # Load dataset paths and labels
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} {split} samples")
        self._print_class_distribution()
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset file paths and labels"""
        samples = []
        
        # Dataset structure: data_root/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
        split_dir = self.data_root / "chest_xray" / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")
        
        # Load NORMAL samples
        normal_dir = split_dir / "NORMAL"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.jpeg"):
                samples.append({
                    "image_path": str(img_path),
                    "label": 0,  # Normal
                    "patient_id": img_path.stem,
                    "class_name": "NORMAL"
                })
        
        # Load PNEUMONIA samples
        pneumonia_dir = split_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob("*.jpeg"):
                samples.append({
                    "image_path": str(img_path),
                    "label": 1,  # Pneumonia
                    "patient_id": img_path.stem,
                    "class_name": "PNEUMONIA"
                })
        
        return samples
    
    def _print_class_distribution(self):
        """Print class distribution"""
        labels = [sample["label"] for sample in self.samples]
        normal_count = labels.count(0)
        pneumonia_count = labels.count(1)
        
        print(f"{self.split.capitalize()} set distribution:")
        print(f"  Normal: {normal_count}")
        print(f"  Pneumonia: {pneumonia_count}")
        print(f"  Total: {len(labels)}")
        print(f"  Class balance: {pneumonia_count / len(labels):.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = self._load_and_preprocess_image(sample["image_path"])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "patient_id": sample["patient_id"],
            "image_path": sample["image_path"]
        }
    
    def _load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image based on preprocessing type"""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply preprocessing based on type
        if self.preprocessing_type == "histogram_matching":
            image = self._apply_histogram_matching(image)
        elif self.preprocessing_type == "vit_standard":
            image = self._apply_vit_preprocessing(image)
        # "raw" preprocessing - just convert to RGB
        
        # Convert to RGB (ViT expects 3 channels)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        return image
    
    def _apply_vit_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply ViT-specific preprocessing"""
        # Normalize to 0-255 range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Gentle Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return image
    
    def _apply_histogram_matching(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram matching for normalization"""
        # Target histogram (normal distribution)
        target_mean = 127
        target_std = 50
        
        # Current image statistics
        current_mean = np.mean(image)
        current_std = np.std(image)
        
        # Normalize current image
        if current_std > 0:
            normalized = (image - current_mean) / current_std
            # Apply target statistics
            matched = normalized * target_std + target_mean
            matched = np.clip(matched, 0, 255).astype(np.uint8)
        else:
            matched = image
        
        return matched
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        labels = [sample["label"] for sample in self.samples]
        
        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        
        # Calculate weights (inverse frequency)
        total_samples = len(labels)
        weights = total_samples / (len(unique) * counts)
        
        # Convert to tensor
        weight_tensor = torch.zeros(2)
        for class_idx, weight in zip(unique, weights):
            weight_tensor[int(class_idx)] = weight
        
        return weight_tensor


def get_vit_transforms(config, split: str = "train") -> transforms.Compose:
    """
    Get ViT-appropriate transforms for different splits
    
    Args:
        config: Data configuration
        split: Dataset split ("train", "val", "test")
        
    Returns:
        transform: Composed transforms
    """
    image_size = config.image_size
    
    if split == "train" and config.use_augmentation:
        # Training transforms with augmentation
        transform = transforms.Compose([
            # Resize and crop
            transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
            transforms.RandomCrop(image_size),
            
            # ViT-specific augmentations
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            transforms.RandomRotation(degrees=config.rotation_range),
            
            # Color augmentations (careful with medical images)
            transforms.ColorJitter(
                brightness=config.brightness_factor,
                contrast=config.contrast_factor
            ),
            
            # Normalization for ViT (ImageNet stats)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
    """
    
    # Get transforms
    train_transform = get_vit_transforms(config.data, "train")
    val_transform = get_vit_transforms(config.data, "val")
    test_transform = get_vit_transforms(config.data, "test")
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        data_root=config.data.data_root,
        split="train",
        transform=train_transform,
        preprocessing_type=config.data.preprocessing_type
    )
    
    val_dataset = ChestXrayDataset(
        data_root=config.data.data_root,
        split="val",
        transform=val_transform,
        preprocessing_type=config.data.preprocessing_type
    )
    
    test_dataset = ChestXrayDataset(
        data_root=config.data.data_root,
        split="test",
        transform=test_transform,
        preprocessing_type=config.data.preprocessing_type
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation for ViT training
    
    Args:
        x: Input images [batch, channels, height, width]
        y: Labels [batch]
        alpha: Mixup parameter
        
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Mixed labels  
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation for ViT training
    
    Args:
        x: Input images [batch, channels, height, width]
        y: Labels [batch]
        alpha: CutMix parameter
        
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    
    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


if __name__ == "__main__":
    # Test data loading
    from config import get_vit_base_config
    
    config = get_vit_base_config()
    
    # Update data path (adjust as needed)
    config.data.data_root = "../data"  # Adjust to your data location
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        print("✅ Data loaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")  
        print(f"Test batches: {len(test_loader)}")
        
        # Test batch loading
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        
    except FileNotFoundError as e:
        print(f"⚠️  Data directory not found: {e}")
        print("Please update the data_root path in the configuration")
    
    print("✅ Data module test completed!")
