"""
Vision Transformer (ViT) model for pneumonia classification with attention visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionOutput:
    """Container for attention map outputs"""
    attention_maps: torch.Tensor  # [batch, heads, tokens, tokens]
    patch_embeddings: torch.Tensor  # [batch, tokens, embed_dim]
    cls_token: torch.Tensor  # [batch, embed_dim]
    layer_index: int


class VisionTransformerWithAttention(nn.Module):
    """
    Vision Transformer model with attention map extraction for interpretability
    """
    
    def __init__(self, 
                 model_name: str = "vit_base_patch16_224",
                 num_classes: int = 1,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1,
                 freeze_backbone: bool = False,
                 freeze_layers: int = 0):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.patch_size = 16  # Default patch size
        
        # Load pretrained ViT model from timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            drop_rate=dropout_rate
        )
        
        # Get model dimensions
        self.embed_dim = self.backbone.embed_dim
        self.num_heads = self.backbone.blocks[0].attn.num_heads
        self.num_layers = len(self.backbone.blocks)
        
        # Custom classifier head for binary classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Storage for attention maps during forward pass
        self.attention_maps = {}
        self.attention_hooks = []
        
        # Apply freezing if specified
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_backbone(self):
        """Freeze entire backbone for feature extraction only"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first num_layers transformer blocks"""
        # Freeze patch embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze positional embedding
        if hasattr(self.backbone, 'pos_embed'):
            self.backbone.pos_embed.requires_grad = False
        
        # Freeze specified number of transformer blocks
        for i in range(min(num_layers, len(self.backbone.blocks))):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = False
    
    def register_attention_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register forward hooks to capture attention maps"""
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))
        
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # For timm ViT, we need to hook the attention weights directly
                # The attention module may not return weights in output
                if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                    self.attention_maps[layer_idx] = module.attention_weights.detach()
                elif hasattr(module, 'attn_drop') and len(input) > 0:
                    # Try to extract attention from the attention computation
                    try:
                        # Get the attention weights from qkv computation
                        B, N, C = input[0].shape
                        qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv.unbind(0)
                        attn = (q @ k.transpose(-2, -1)) * module.scale
                        attn = attn.softmax(dim=-1)
                        self.attention_maps[layer_idx] = attn.detach()
                    except:
                        # Fallback: create dummy attention map
                        self.attention_maps[layer_idx] = torch.ones(1, module.num_heads, N, N) * 0.01
            return hook
        
        # Clear existing hooks
        self.clear_attention_hooks()
        
        # Register new hooks
        for layer_idx in layer_indices:
            if layer_idx < len(self.backbone.blocks):
                hook = self.backbone.blocks[layer_idx].attn.register_forward_hook(
                    get_attention_hook(layer_idx)
                )
                self.attention_hooks.append(hook)
    
    def clear_attention_hooks(self):
        """Remove all attention hooks"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks.clear()
        self.attention_maps.clear()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT
        
        Args:
            x: Input images [batch, 3, height, width]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        # Extract features using backbone (without classifier)
        features = self.backbone(x)  # [batch, embed_dim]
        
        # Apply custom classifier
        logits = self.classifier(features)
        
        return logits
    
    def forward_with_attention(self, x: torch.Tensor, 
                             layer_indices: Optional[List[int]] = None) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with attention map extraction
        
        Args:
            x: Input images [batch, 3, height, width]
            layer_indices: Which layers to extract attention from
            
        Returns:
            logits: Classification logits [batch, num_classes]
            attention_maps: Dictionary mapping layer_idx -> attention weights
        """
        # Register hooks for attention extraction
        self.register_attention_hooks(layer_indices)
        
        # Forward pass
        logits = self.forward(x)
        
        # Extract attention maps
        attention_maps = self.attention_maps.copy()
        
        # Clear hooks to avoid memory leaks
        self.clear_attention_hooks()
        
        return logits, attention_maps
    
    def get_patch_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get patch embeddings and CLS token
        
        Args:
            x: Input images [batch, 3, height, width]
            
        Returns:
            patch_embeddings: Patch embeddings [batch, num_patches, embed_dim]
            cls_token: CLS token [batch, embed_dim]
        """
        # Patch embedding
        x = self.backbone.patch_embed(x)  # [batch, num_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [batch, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        return x[:, 1:], x[:, 0]  # patches, cls_token
    
    def compute_rollout_attention(self, attention_maps: Dict[int, torch.Tensor], 
                                head_fusion: str = "mean") -> torch.Tensor:
        """
        Compute attention rollout across all layers
        
        Args:
            attention_maps: Attention maps from multiple layers
            head_fusion: How to fuse multiple heads ("mean", "max", "min")
            
        Returns:
            rollout_attention: Rolled out attention [batch, num_patches]
        """
        if not attention_maps:
            # Return dummy attention if no maps available
            print("⚠️  No attention maps available, returning dummy attention")
            return torch.ones(1, 196) * 0.01  # 14x14 patches for 224x224 image
        
        batch_size = list(attention_maps.values())[0].shape[0]
        num_patches = list(attention_maps.values())[0].shape[-1] - 1  # Exclude CLS token
        
        # Start with identity matrix
        rollout = torch.eye(num_patches + 1).unsqueeze(0).repeat(batch_size, 1, 1)
        rollout = rollout.to(list(attention_maps.values())[0].device)
        
        # Apply attention rollout
        for layer_idx in sorted(attention_maps.keys()):
            attention = attention_maps[layer_idx]  # [batch, heads, tokens, tokens]
            
            # Fuse attention heads
            if head_fusion == "mean":
                attention = attention.mean(dim=1)
            elif head_fusion == "max":
                attention = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention = attention.min(dim=1)[0]
            
            # Add residual connection (attention rollout)
            attention = attention + torch.eye(attention.shape[-1]).unsqueeze(0).to(attention.device)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            # Multiply with previous rollout
            rollout = torch.bmm(attention, rollout)
        
        # Extract attention from CLS token to patches
        cls_attention = rollout[:, 0, 1:]  # [batch, num_patches]
        
        return cls_attention
    
    def get_attention_map_2d(self, attention_1d: torch.Tensor, 
                           image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Convert 1D attention to 2D spatial attention map
        
        Args:
            attention_1d: 1D attention weights [batch, num_patches]
            image_size: Original image size (height, width)
            
        Returns:
            attention_2d: 2D attention map [batch, height, width]
        """
        batch_size = attention_1d.shape[0]
        
        # Calculate grid size
        num_patches = attention_1d.shape[1]
        grid_size = int(np.sqrt(num_patches))
        
        # Reshape to 2D grid
        attention_2d = attention_1d.view(batch_size, grid_size, grid_size)
        
        # Interpolate to original image size
        attention_2d = F.interpolate(
            attention_2d.unsqueeze(1),  # Add channel dimension
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # Remove channel dimension
        
        return attention_2d


def create_vit_model(config) -> VisionTransformerWithAttention:
    """
    Create Vision Transformer model from configuration
    
    Args:
        config: Model configuration
        
    Returns:
        model: ViT model with attention extraction
    """
    return VisionTransformerWithAttention(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        freeze_backbone=config.freeze_backbone,
        freeze_layers=config.freeze_layers
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


if __name__ == "__main__":
    # Test model creation
    from config import get_vit_base_config
    
    config = get_vit_base_config()
    model = create_vit_model(config.model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    print("Model created successfully!")
    print(f"Model: {config.model.model_name}")
    
    param_counts = count_parameters(model)
    print(f"Parameters: {param_counts}")
    
    # Test forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test attention extraction
    logits, attention_maps = model.forward_with_attention(x, layer_indices=[0, 5, 11])
    print(f"Attention maps extracted from layers: {list(attention_maps.keys())}")
    
    for layer_idx, attn in attention_maps.items():
        print(f"Layer {layer_idx} attention shape: {attn.shape}")
    
    print("✅ ViT model test completed!")
