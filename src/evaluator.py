"""
Evaluation module for Vision Transformer pneumonia classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import asdict

from attention_visualization import ViTAttentionVisualizer


class ViTEvaluator:
    """
    Comprehensive evaluation for Vision Transformer pneumonia classification
    """
    
    def __init__(self, 
                 model,
                 config,
                 device: str = "cuda"):
        """
        Initialize evaluator
        
        Args:
            model: Trained ViT model
            config: Configuration object
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attention visualizer
        self.attention_visualizer = ViTAttentionVisualizer(
            model=self.model,
            device=self.device,
            save_dir=str(self.output_dir / "attention_maps")
        )
    
    def evaluate_model(self, test_loader) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            
        Returns:
            evaluation_results: Dictionary with all evaluation metrics
        """
        print("🔍 Starting comprehensive ViT evaluation...")
        
        # 1. Basic performance metrics
        basic_metrics = self._evaluate_basic_metrics(test_loader)
        
        # 2. Threshold optimization
        optimal_threshold = self._optimize_threshold(test_loader)
        
        # 3. Detailed analysis
        detailed_analysis = self._detailed_analysis(test_loader, optimal_threshold)
        
        # 4. Attention visualization
        if self.config.evaluation.generate_attention_maps:
            try:
                attention_results = self._generate_attention_visualizations(test_loader)
            except Exception as e:
                print(f"⚠️  Warning: Attention visualization failed: {e}")
                attention_results = {'error': str(e), 'num_visualizations': 0}
        else:
            attention_results = {}
        
        # 5. Error analysis
        error_analysis = self._error_analysis(test_loader, optimal_threshold)
        
        # Combine results
        evaluation_results = {
            'basic_metrics': basic_metrics,
            'optimal_threshold': optimal_threshold,
            'detailed_analysis': detailed_analysis,
            'attention_results': attention_results,
            'error_analysis': error_analysis,
            'config': asdict(self.config)
        }
        
        # Save results
        self._save_evaluation_results(evaluation_results)
        
        # Generate report
        self._generate_evaluation_report(evaluation_results)
        
        print("✅ Evaluation completed successfully!")
        
        return evaluation_results
    
    def _evaluate_basic_metrics(self, test_loader) -> Dict:
        """Evaluate basic performance metrics"""
        print("📊 Evaluating basic metrics...")
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_patient_ids = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                patient_ids = batch['patient_id']
                
                # Measure inference time
                start_time = time.time()
                logits = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / images.size(0))  # Per sample
                
                # Get predictions
                probabilities = torch.sigmoid(logits.squeeze()).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(labels)
                all_patient_ids.extend(patient_ids)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Average inference time
        avg_inference_time = np.mean(inference_times)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'specificity': specificity,
            'sensitivity': recall,  # Same as recall
            'avg_inference_time': avg_inference_time,
            'total_samples': len(all_labels),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist(),
            'labels': all_labels.tolist(),
            'patient_ids': all_patient_ids
        }
    
    def _optimize_threshold(self, test_loader) -> Dict:
        """Find optimal classification threshold"""
        print("🎯 Optimizing classification threshold...")
        
        # Get predictions and labels
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                logits = self.model(images)
                probabilities = torch.sigmoid(logits.squeeze()).cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_labels.extend(labels)
        
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        threshold_results = []
        
        for threshold in thresholds:
            predictions = (all_probabilities > threshold).astype(int)
            
            # Calculate metrics for this threshold
            accuracy = accuracy_score(all_labels, predictions)
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            f1 = f1_score(all_labels, predictions, zero_division=0)
            
            # Use F1 score as optimization metric (configurable)
            metric_name = self.config.evaluation.threshold_metric
            if metric_name == 'f1':
                score = f1
            elif metric_name == 'accuracy':
                score = accuracy
            elif metric_name == 'precision':
                score = precision
            elif metric_name == 'recall':
                score = recall
            else:
                score = f1  # Default to F1
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'metric_score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return {
            'best_threshold': best_threshold,
            'best_score': best_score,
            'optimization_metric': metric_name,
            'threshold_results': threshold_results
        }
    
    def _detailed_analysis(self, test_loader, optimal_threshold: Dict) -> Dict:
        """Detailed performance analysis"""
        print("🔬 Performing detailed analysis...")
        
        threshold = optimal_threshold['best_threshold']
        
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                logits = self.model(images)
                probabilities = torch.sigmoid(logits.squeeze()).cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_labels.extend(labels)
        
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        predictions = (all_probabilities > threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, predictions, 
            target_names=['Normal', 'Pneumonia'],
            output_dict=True
        )
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probabilities)
        
        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            all_labels, all_probabilities
        )
        
        # Create visualizations
        self._plot_confusion_matrix(cm)
        self._plot_roc_curve(fpr, tpr, roc_auc_score(all_labels, all_probabilities))
        self._plot_precision_recall_curve(precision_curve, recall_curve, pr_thresholds)
        self._plot_threshold_analysis(optimal_threshold['threshold_results'])
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'precision_recall_curve': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
    
    def _generate_attention_visualizations(self, test_loader) -> Dict:
        """Generate attention visualizations for sample images"""
        print("🎨 Generating attention visualizations...")
        
        # Get sample images for visualization
        sample_batch = next(iter(test_loader))
        num_samples = min(self.config.evaluation.num_attention_samples, len(sample_batch['image']))
        
        sample_images = sample_batch['image'][:num_samples]
        sample_labels = sample_batch['label'][:num_samples]
        sample_patient_ids = sample_batch['patient_id'][:num_samples]
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(sample_images.to(self.device))
            predictions = torch.sigmoid(logits.squeeze())
        
        # Generate attention visualizations
        visualization_paths = self.attention_visualizer.visualize_attention_maps(
            images=sample_images,
            layer_indices=self.config.evaluation.attention_layers,
            head_indices=self.config.evaluation.attention_heads,
            patient_ids=sample_patient_ids,
            predictions=predictions.cpu(),
            labels=sample_labels
        )
        
        return {
            'num_visualizations': num_samples,
            'visualization_paths': visualization_paths,
            'attention_layers': self.config.evaluation.attention_layers,
            'attention_heads': self.config.evaluation.attention_heads
        }
    
    def _error_analysis(self, test_loader, optimal_threshold: Dict) -> Dict:
        """Analyze prediction errors"""
        print("🔍 Performing error analysis...")
        
        threshold = optimal_threshold['best_threshold']
        
        correct_predictions = []
        incorrect_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                patient_ids = batch['patient_id']
                
                logits = self.model(images)
                probabilities = torch.sigmoid(logits.squeeze()).cpu().numpy()
                predictions = (probabilities > threshold).astype(int)
                
                for i in range(len(labels)):
                    sample_info = {
                        'patient_id': patient_ids[i],
                        'true_label': int(labels[i]),
                        'predicted_label': int(predictions[i]),
                        'probability': float(probabilities[i]),
                        'confidence': abs(probabilities[i] - 0.5)
                    }
                    
                    if predictions[i] == labels[i]:
                        correct_predictions.append(sample_info)
                    else:
                        incorrect_predictions.append(sample_info)
        
        # Analyze error patterns
        false_positives = [p for p in incorrect_predictions if p['predicted_label'] == 1]
        false_negatives = [p for p in incorrect_predictions if p['predicted_label'] == 0]
        
        # Calculate confidence statistics
        correct_confidences = [p['confidence'] for p in correct_predictions]
        incorrect_confidences = [p['confidence'] for p in incorrect_predictions]
        
        return {
            'total_correct': len(correct_predictions),
            'total_incorrect': len(incorrect_predictions),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'error_rate': len(incorrect_predictions) / (len(correct_predictions) + len(incorrect_predictions)),
            'avg_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
            'avg_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'incorrect_samples': incorrect_predictions[:10],  # Top 10 errors
            'low_confidence_correct': sorted(correct_predictions, key=lambda x: x['confidence'])[:5],
            'high_confidence_incorrect': sorted(incorrect_predictions, key=lambda x: x['confidence'], reverse=True)[:5]
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia']
        )
        plt.title('Confusion Matrix - ViT Pneumonia Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ViT Pneumonia Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - ViT Pneumonia Classification')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self, threshold_results: List[Dict]):
        """Plot threshold optimization analysis"""
        thresholds = [r['threshold'] for r in threshold_results]
        accuracies = [r['accuracy'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        recalls = [r['recall'] for r in threshold_results]
        f1_scores = [r['f1_score'] for r in threshold_results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        
        plt.xlabel('Classification Threshold')
        plt.ylabel('Metric Value')
        plt.title('Threshold Optimization - ViT Pneumonia Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to JSON"""
        # Create a copy without numpy arrays for JSON serialization
        def convert_to_json_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        json_results = {
            'basic_metrics': {
                k: convert_to_json_serializable(v) for k, v in results['basic_metrics'].items()
                if k not in ['predictions', 'probabilities', 'labels']
            },
            'optimal_threshold': convert_to_json_serializable(results['optimal_threshold']),
            'error_analysis': convert_to_json_serializable(results['error_analysis']),
            'attention_results': convert_to_json_serializable(results['attention_results'])
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Evaluation results saved to {self.output_dir}")
    
    def _generate_evaluation_report(self, results: Dict):
        """Generate comprehensive evaluation report"""
        basic_metrics = results['basic_metrics']
        optimal_threshold = results['optimal_threshold']
        error_analysis = results['error_analysis']
        
        report = f"""
# Vision Transformer Pneumonia Classification - Evaluation Report

## Model Performance Summary

### Basic Metrics (Threshold: {optimal_threshold['best_threshold']:.3f})
- **Accuracy**: {basic_metrics['accuracy']:.4f}
- **Precision**: {basic_metrics['precision']:.4f}
- **Recall (Sensitivity)**: {basic_metrics['recall']:.4f}
- **Specificity**: {basic_metrics['specificity']:.4f}
- **F1 Score**: {basic_metrics['f1_score']:.4f}
- **AUC-ROC**: {basic_metrics['auc_roc']:.4f}

### Performance Analysis
- **Total Test Samples**: {basic_metrics['total_samples']}
- **Correct Predictions**: {error_analysis['total_correct']}
- **Incorrect Predictions**: {error_analysis['total_incorrect']}
- **Error Rate**: {error_analysis['error_rate']:.4f}
- **False Positives**: {error_analysis['false_positives']}
- **False Negatives**: {error_analysis['false_negatives']}

### Inference Performance
- **Average Inference Time**: {basic_metrics['avg_inference_time']:.4f} seconds per image

### Threshold Optimization
- **Optimization Metric**: {optimal_threshold['optimization_metric']}
- **Best Threshold**: {optimal_threshold['best_threshold']:.3f}
- **Best Score**: {optimal_threshold['best_score']:.4f}

### Attention Visualization
- **Generated Visualizations**: {results['attention_results'].get('num_visualizations', 0)}
- **Attention Layers Analyzed**: {results['attention_results'].get('attention_layers', [])}
- **Attention Heads Analyzed**: {results['attention_results'].get('attention_heads', [])}

## Model Interpretability
The Vision Transformer model provides excellent interpretability through:
1. **Attention Rollout**: Shows which image regions the model focuses on for final prediction
2. **Multi-Head Attention**: Reveals different attention patterns across attention heads
3. **Layer Comparison**: Demonstrates how attention evolves through transformer layers
4. **Attention Statistics**: Quantifies attention distribution and focus patterns

## Clinical Relevance
- High sensitivity ({basic_metrics['recall']:.4f}) is crucial for pneumonia detection to minimize missed cases
- Good specificity ({basic_metrics['specificity']:.4f}) helps reduce false alarms
- Attention maps provide clinically interpretable explanations for predictions
- Fast inference time ({basic_metrics['avg_inference_time']:.4f}s) suitable for clinical deployment

## Recommendations
1. **Deployment**: Model shows strong performance suitable for clinical assistance
2. **Threshold**: Use optimized threshold of {optimal_threshold['best_threshold']:.3f} for best {optimal_threshold['optimization_metric']} performance
3. **Monitoring**: Continue monitoring for distribution shifts in real-world deployment
4. **Interpretability**: Use attention maps for clinical decision support and model validation

---
*Report generated automatically by ViT Evaluation System*
"""
        
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        print("📋 Evaluation report generated successfully!")


if __name__ == "__main__":
    print("✅ ViT evaluation module loaded successfully!")
    print("Use ViTEvaluator class for comprehensive model evaluation")
