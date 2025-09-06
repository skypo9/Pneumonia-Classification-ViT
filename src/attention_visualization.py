"""
Attention visualization and interpretability for Vision Transformer pneumonia classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import os

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")


class ViTAttentionVisualizer:
    """
    Visualization tools for Vision Transformer attention maps and interpretability
    """
    
    def __init__(self, 
                 model,
                 device: str = "cpu",
                 save_dir: str = "attention_visualizations"):
        """
        Initialize attention visualizer
        
        Args:
            model: ViT model with attention extraction
            device: Device for computation
            save_dir: Directory to save visualizations
        """
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def visualize_attention_maps(self,
                               images: torch.Tensor,
                               layer_indices: List[int] = [3, 6, 9, 11],
                               head_indices: List[int] = [0, 3, 6, 9],
                               patient_ids: Optional[List[str]] = None,
                               predictions: Optional[torch.Tensor] = None,
                               labels: Optional[torch.Tensor] = None) -> Dict[str, List]:
        """
        Generate comprehensive attention visualizations
        
        Args:
            images: Input images [batch, 3, height, width]
            layer_indices: Which transformer layers to visualize
            head_indices: Which attention heads to visualize
            patient_ids: Patient identifiers
            predictions: Model predictions
            labels: Ground truth labels
            
        Returns:
            visualization_paths: Dictionary of saved visualization paths
        """
        batch_size = images.shape[0]
        if patient_ids is None:
            patient_ids = [f"sample_{i}" for i in range(batch_size)]
        
        visualization_paths = {
            "attention_rollout": [],
            "multi_head_attention": [],
            "layer_comparison": [],
            "attention_statistics": []
        }
        
        with torch.no_grad():
            # Get predictions and attention maps
            logits, attention_maps = self.model.forward_with_attention(
                images.to(self.device), 
                layer_indices=layer_indices
            )
            
            if predictions is None:
                predictions = torch.sigmoid(logits)
            
            # Process each sample in batch
            for i in range(batch_size):
                patient_id = patient_ids[i]
                pred = predictions[i].item()
                label = labels[i].item() if labels is not None else None
                
                # Extract attention maps for this sample
                sample_attention = {
                    layer_idx: attn[i] for layer_idx, attn in attention_maps.items()
                }
                
                # 1. Attention rollout visualization
                rollout_path = self._visualize_attention_rollout(
                    images[i], sample_attention, patient_id, pred, label
                )
                visualization_paths["attention_rollout"].append(rollout_path)
                
                # 2. Multi-head attention visualization
                multihead_path = self._visualize_multihead_attention(
                    images[i], sample_attention, patient_id, head_indices, pred, label
                )
                visualization_paths["multi_head_attention"].append(multihead_path)
                
                # 3. Layer comparison visualization
                layer_comp_path = self._visualize_layer_comparison(
                    images[i], sample_attention, patient_id, pred, label
                )
                visualization_paths["layer_comparison"].append(layer_comp_path)
                
                # 4. Attention statistics
                stats_path = self._visualize_attention_statistics(
                    sample_attention, patient_id, pred, label
                )
                visualization_paths["attention_statistics"].append(stats_path)
        
        return visualization_paths
    
    def _visualize_attention_rollout(self, 
                                   image: torch.Tensor,
                                   attention_maps: Dict[int, torch.Tensor],
                                   patient_id: str,
                                   prediction: float,
                                   label: Optional[float] = None) -> str:
        """Create attention rollout visualization"""
        
        # Compute attention rollout
        rollout_attention = self.model.compute_rollout_attention(
            {k: v.unsqueeze(0) for k, v in attention_maps.items()},
            head_fusion="mean"
        ).squeeze(0)  # [num_patches]
        
        # Convert to 2D spatial map
        attention_2d = self.model.get_attention_map_2d(
            rollout_attention.unsqueeze(0),
            image_size=(224, 224)
        ).squeeze(0)  # [height, width]
        
        # Prepare image for visualization
        vis_image = self._prepare_image_for_vis(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(vis_image, cmap='gray')
        axes[0].set_title("Original X-ray")
        axes[0].axis('off')
        
        # Attention heatmap
        im1 = axes[1].imshow(attention_2d.cpu().numpy(), cmap='hot')
        axes[1].set_title("Attention Rollout")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(vis_image, cmap='gray', alpha=0.7)
        im2 = axes[2].imshow(attention_2d.cpu().numpy(), cmap='hot', alpha=0.5)
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')
        
        # Add prediction info
        pred_class = "Pneumonia" if prediction > 0.5 else "Normal"
        true_class = "Pneumonia" if label and label > 0.5 else "Normal" if label is not None else "Unknown"
        
        fig.suptitle(f"Patient: {patient_id} | Pred: {pred_class} ({prediction:.3f}) | True: {true_class}")
        
        # Save
        save_path = self.save_dir / f"{patient_id}_attention_rollout.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _visualize_multihead_attention(self,
                                     image: torch.Tensor,
                                     attention_maps: Dict[int, torch.Tensor],
                                     patient_id: str,
                                     head_indices: List[int],
                                     prediction: float,
                                     label: Optional[float] = None) -> str:
        """Create multi-head attention visualization"""
        
        # Use last layer for multi-head visualization
        last_layer = max(attention_maps.keys())
        attention = attention_maps[last_layer]  # [heads, tokens, tokens]
        
        # Extract CLS token attention to patches
        cls_attention = attention[:, 0, 1:]  # [heads, num_patches]
        
        # Prepare image
        vis_image = self._prepare_image_for_vis(image)
        
        # Create subplot grid
        n_heads = min(len(head_indices), cls_attention.shape[0])
        fig, axes = plt.subplots(2, n_heads + 1, figsize=(4 * (n_heads + 1), 8))
        
        # Original image (spans both rows)
        axes[0, 0].imshow(vis_image, cmap='gray')
        axes[0, 0].set_title("Original X-ray")
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')  # Hide second row for original image
        
        # Visualize each attention head
        for i, head_idx in enumerate(head_indices[:n_heads]):
            if head_idx < cls_attention.shape[0]:
                # Convert to 2D
                head_attention_2d = self.model.get_attention_map_2d(
                    cls_attention[head_idx].unsqueeze(0),
                    image_size=(224, 224)
                ).squeeze(0)
                
                # Attention heatmap
                im1 = axes[0, i + 1].imshow(head_attention_2d.cpu().numpy(), cmap='hot')
                axes[0, i + 1].set_title(f"Head {head_idx}")
                axes[0, i + 1].axis('off')
                
                # Overlay
                axes[1, i + 1].imshow(vis_image, cmap='gray', alpha=0.7)
                axes[1, i + 1].imshow(head_attention_2d.cpu().numpy(), cmap='hot', alpha=0.5)
                axes[1, i + 1].set_title(f"Head {head_idx} Overlay")
                axes[1, i + 1].axis('off')
        
        # Add prediction info
        pred_class = "Pneumonia" if prediction > 0.5 else "Normal"
        true_class = "Pneumonia" if label and label > 0.5 else "Normal" if label is not None else "Unknown"
        
        fig.suptitle(f"Multi-Head Attention (Layer {last_layer}) | Patient: {patient_id} | Pred: {pred_class} ({prediction:.3f})")
        
        # Save
        save_path = self.save_dir / f"{patient_id}_multihead_attention.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _visualize_layer_comparison(self,
                                  image: torch.Tensor,
                                  attention_maps: Dict[int, torch.Tensor],
                                  patient_id: str,
                                  prediction: float,
                                  label: Optional[float] = None) -> str:
        """Compare attention across different layers"""
        
        # Prepare image
        vis_image = self._prepare_image_for_vis(image)
        
        # Get layer indices
        layer_indices = sorted(attention_maps.keys())
        n_layers = len(layer_indices)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))
        
        # Original image
        axes[0, 0].imshow(vis_image, cmap='gray')
        axes[0, 0].set_title("Original X-ray")
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Visualize each layer
        for i, layer_idx in enumerate(layer_indices):
            attention = attention_maps[layer_idx]  # [heads, tokens, tokens]
            
            # Average across heads and extract CLS attention
            avg_attention = attention.mean(dim=0)  # [tokens, tokens]
            cls_attention = avg_attention[0, 1:]  # [num_patches]
            
            # Convert to 2D
            attention_2d = self.model.get_attention_map_2d(
                cls_attention.unsqueeze(0),
                image_size=(224, 224)
            ).squeeze(0)
            
            # Attention heatmap
            im1 = axes[0, i + 1].imshow(attention_2d.cpu().numpy(), cmap='hot')
            axes[0, i + 1].set_title(f"Layer {layer_idx}")
            axes[0, i + 1].axis('off')
            
            # Overlay
            axes[1, i + 1].imshow(vis_image, cmap='gray', alpha=0.7)
            axes[1, i + 1].imshow(attention_2d.cpu().numpy(), cmap='hot', alpha=0.5)
            axes[1, i + 1].set_title(f"Layer {layer_idx} Overlay")
            axes[1, i + 1].axis('off')
        
        # Add prediction info
        pred_class = "Pneumonia" if prediction > 0.5 else "Normal"
        true_class = "Pneumonia" if label and label > 0.5 else "Normal" if label is not None else "Unknown"
        
        fig.suptitle(f"Layer Comparison | Patient: {patient_id} | Pred: {pred_class} ({prediction:.3f})")
        
        # Save
        save_path = self.save_dir / f"{patient_id}_layer_comparison.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _visualize_attention_statistics(self,
                                      attention_maps: Dict[int, torch.Tensor],
                                      patient_id: str,
                                      prediction: float,
                                      label: Optional[float] = None) -> str:
        """Create attention statistics visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Attention entropy across layers
        entropies = []
        layer_indices = sorted(attention_maps.keys())
        
        for layer_idx in layer_indices:
            attention = attention_maps[layer_idx]  # [heads, tokens, tokens]
            # Average across heads
            avg_attention = attention.mean(dim=0)  # [tokens, tokens]
            # Extract CLS attention to patches
            cls_attention = avg_attention[0, 1:]  # [num_patches]
            # Compute entropy
            entropy = -torch.sum(cls_attention * torch.log(cls_attention + 1e-8))
            entropies.append(entropy.item())
        
        axes[0, 0].plot(layer_indices, entropies, 'o-')
        axes[0, 0].set_xlabel("Layer Index")
        axes[0, 0].set_ylabel("Attention Entropy")
        axes[0, 0].set_title("Attention Entropy Across Layers")
        axes[0, 0].grid(True)
        
        # 2. Attention spread (standard deviation)
        spreads = []
        for layer_idx in layer_indices:
            attention = attention_maps[layer_idx]
            avg_attention = attention.mean(dim=0)
            cls_attention = avg_attention[0, 1:]
            spread = torch.std(cls_attention)
            spreads.append(spread.item())
        
        axes[0, 1].plot(layer_indices, spreads, 'o-', color='orange')
        axes[0, 1].set_xlabel("Layer Index")
        axes[0, 1].set_ylabel("Attention Spread (Std)")
        axes[0, 1].set_title("Attention Spread Across Layers")
        axes[0, 1].grid(True)
        
        # 3. Head diversity in last layer
        last_layer = max(layer_indices)
        last_attention = attention_maps[last_layer]  # [heads, tokens, tokens]
        head_similarities = []
        
        for i in range(last_attention.shape[0]):
            for j in range(i + 1, last_attention.shape[0]):
                # Cosine similarity between attention patterns
                head_i = last_attention[i, 0, 1:].flatten()
                head_j = last_attention[j, 0, 1:].flatten()
                similarity = F.cosine_similarity(head_i, head_j, dim=0)
                head_similarities.append(similarity.item())
        
        axes[1, 0].hist(head_similarities, bins=10, alpha=0.7)
        axes[1, 0].set_xlabel("Cosine Similarity")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title(f"Head Diversity (Layer {last_layer})")
        axes[1, 0].axvline(np.mean(head_similarities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(head_similarities):.3f}')
        axes[1, 0].legend()
        
        # 4. Attention focus (top-k attention concentration)
        k_values = [5, 10, 20, 50]
        concentrations = {k: [] for k in k_values}
        
        for layer_idx in layer_indices:
            attention = attention_maps[layer_idx]
            avg_attention = attention.mean(dim=0)
            cls_attention = avg_attention[0, 1:]
            
            # Sort attention values
            sorted_attention, _ = torch.sort(cls_attention, descending=True)
            
            for k in k_values:
                if k <= len(sorted_attention):
                    top_k_sum = torch.sum(sorted_attention[:k])
                    concentrations[k].append(top_k_sum.item())
        
        for k in k_values:
            axes[1, 1].plot(layer_indices, concentrations[k], 'o-', label=f'Top-{k}')
        
        axes[1, 1].set_xlabel("Layer Index")
        axes[1, 1].set_ylabel("Attention Concentration")
        axes[1, 1].set_title("Attention Concentration (Top-K)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Add prediction info
        pred_class = "Pneumonia" if prediction > 0.5 else "Normal"
        true_class = "Pneumonia" if label and label > 0.5 else "Normal" if label is not None else "Unknown"
        
        fig.suptitle(f"Attention Statistics | Patient: {patient_id} | Pred: {pred_class} ({prediction:.3f})")
        
        # Save
        save_path = self.save_dir / f"{patient_id}_attention_statistics.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _prepare_image_for_vis(self, image: torch.Tensor) -> np.ndarray:
        """Prepare image tensor for visualization"""
        # Denormalize image (reverse ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy and change to HWC format
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Convert to grayscale for medical image visualization
        if image_np.shape[2] == 3:
            image_np = np.dot(image_np, [0.299, 0.587, 0.114])
        
        return image_np
    
    def create_summary_visualization(self, 
                                   visualization_paths: Dict[str, List],
                                   summary_name: str = "attention_summary") -> str:
        """Create a summary visualization combining multiple samples"""
        
        # This would create a grid showing multiple samples
        # Implementation depends on specific requirements
        
        print(f"✅ Created {len(visualization_paths['attention_rollout'])} attention visualizations")
        print(f"📁 Saved to: {self.save_dir}")
        
        return str(self.save_dir / f"{summary_name}.png")


if __name__ == "__main__":
    print("✅ Attention visualization module loaded successfully!")
    print("Use ViTAttentionVisualizer to create interpretability visualizations")
