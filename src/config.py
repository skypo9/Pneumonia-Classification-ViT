"""
Configuration management for Vision Transformer Pneumonia Classification
"""

import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Data configuration for chest X-ray dataset"""
    data_root: str = "data/chest_xray_pneumonia"
    preprocessing_type: str = "vit_standard"  # vit_standard, raw, histogram_matching
    image_size: List[int] = None
    use_augmentation: bool = True
    rotation_range: float = 10.0
    horizontal_flip_prob: float = 0.5
    brightness_factor: float = 0.1
    contrast_factor: float = 0.1
    
    def __post_init__(self):
        if self.image_size is None:
            self.image_size = [224, 224]  # ViT standard input size


@dataclass 
class ModelConfig:
    """Vision Transformer model configuration"""
    # ViT Architecture
    model_name: str = "vit_base_patch16_224"  # timm model name
    pretrained: bool = True
    num_classes: int = 1  # Binary classification
    dropout_rate: float = 0.1
    
    # ViT specific parameters
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Fine-tuning options
    freeze_backbone: bool = False
    freeze_layers: int = 0  # Number of transformer blocks to freeze


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for ViT"""
    # Optimization
    learning_rate: float = 3e-4  # ViT typically uses higher LR
    weight_decay: float = 0.3    # Strong weight decay for ViT
    batch_size: int = 16
    num_epochs: int = 50
    patience: int = 10
    
    # Loss function
    loss_type: str = "bce_with_logits"
    class_balancing: bool = True
    pos_weight: Optional[float] = None
    
    # Learning rate scheduling (important for ViT)
    use_scheduler: bool = True
    scheduler_type: str = "cosine_warmup"  # cosine_warmup, reduce_on_plateau
    warmup_epochs: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-7
    
    # ViT specific training
    gradient_clip_val: float = 1.0
    mixup_alpha: float = 0.2  # Mixup augmentation
    cutmix_alpha: float = 1.0  # CutMix augmentation
    use_mixup: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation and interpretability configuration"""
    threshold_metric: str = "f1"
    
    # Attention visualization (ViT interpretability)
    generate_attention_maps: bool = True
    num_attention_samples: int = 20
    attention_layers: List[int] = None  # Which transformer layers to visualize
    attention_heads: List[int] = None   # Which attention heads to visualize
    
    # Traditional GradCAM (for comparison)
    generate_gradcam: bool = True
    gradcam_layer: str = "blocks.11.norm1"  # Last transformer block
    num_gradcam_samples: int = 20
    
    def __post_init__(self):
        if self.attention_layers is None:
            self.attention_layers = [3, 6, 9, 11]  # Different transformer depths
        if self.attention_heads is None:
            self.attention_heads = [0, 3, 6, 9]  # Different attention heads


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    experiment_name: str = "vit_pneumonia"
    description: str = "Vision Transformer for pneumonia classification"
    random_seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    # Paths (relative to project root)
    output_dir: str = "outputs"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    """Complete configuration for ViT pneumonia classification"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    
    @classmethod
    def load(cls, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            experiment=ExperimentConfig(**config_dict['experiment'])
        )
    
    def save(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def update_paths(self, base_dir: str):
        """Update relative paths with base directory"""
        base_path = Path(base_dir)
        self.experiment.output_dir = str(base_path / self.experiment.output_dir)
        self.experiment.log_dir = str(base_path / self.experiment.log_dir)
        self.experiment.checkpoint_dir = str(base_path / self.experiment.checkpoint_dir)


def get_vit_base_config() -> Config:
    """Get default ViT-Base configuration for pneumonia classification"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
        experiment=ExperimentConfig()
    )


def get_vit_small_config() -> Config:
    """Get ViT-Small configuration for faster training"""
    config = get_vit_base_config()
    config.model.model_name = "vit_small_patch16_224"
    config.model.embed_dim = 384
    config.model.depth = 12
    config.model.num_heads = 6
    config.training.batch_size = 32  # Can use larger batch size
    return config


def get_vit_large_config() -> Config:
    """Get ViT-Large configuration for maximum performance"""
    config = get_vit_base_config()
    config.model.model_name = "vit_large_patch16_224"
    config.model.embed_dim = 1024
    config.model.depth = 24
    config.model.num_heads = 16
    config.training.batch_size = 8  # Needs smaller batch size
    config.training.learning_rate = 1e-4  # Lower LR for large model
    return config


def save_config(config: Config, config_path: str):
    """Save configuration to file"""
    config.save(config_path)


if __name__ == "__main__":
    # Create and save default configurations
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # ViT-Base configuration
    vit_base = get_vit_base_config()
    save_config(vit_base, "configs/vit_base_config.yaml")
    
    # ViT-Small configuration
    vit_small = get_vit_small_config()
    save_config(vit_small, "configs/vit_small_config.yaml")
    
    # ViT-Large configuration
    vit_large = get_vit_large_config()
    save_config(vit_large, "configs/vit_large_config.yaml")
    
    print("✅ Default ViT configurations created!")
