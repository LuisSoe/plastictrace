"""Evaluation utilities for model performance tracking."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

from feedback.schema import ScanRecord
from feedback.dataset_store import DatasetStore
from ml.config import CLASSES


class ModelEvaluator:
    """Evaluates model performance from stored records."""
    
    def __init__(self, dataset_store: DatasetStore):
        """
        Initialize evaluator.
        
        Args:
            dataset_store: DatasetStore instance
        """
        self.dataset_store = dataset_store
    
    def compute_metrics(self, model_version: Optional[str] = None) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            model_version: Filter by model version (None for all)
            
        Returns:
            Dictionary with metrics
        """
        # Get all records (or filtered by version)
        all_records = self.dataset_store.get_all_records()
        if model_version:
            all_records = [r for r in all_records if r.model_version == model_version]
        
        # Only use confirmed or corrected records
        valid_records = [
            r for r in all_records
            if r.is_confirmed or (r.user_label is not None and r.user_label != r.pred_label)
        ]
        
        if not valid_records:
            return {
                "total_samples": 0,
                "error": "No valid records for evaluation"
            }
        
        # Build confusion matrix
        confusion_matrix = self._compute_confusion_matrix(valid_records)
        
        # Per-class metrics
        per_class_metrics = self._compute_per_class_metrics(confusion_matrix)
        
        # Overall accuracy
        accuracy = self._compute_accuracy(confusion_matrix)
        
        # Confidence calibration
        calibration = self._compute_calibration(valid_records)
        
        return {
            "model_version": model_version or "all",
            "total_samples": len(valid_records),
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix,
            "per_class_metrics": per_class_metrics,
            "calibration": calibration
        }
    
    def _compute_confusion_matrix(self, records: List[ScanRecord]) -> Dict[str, Dict[str, int]]:
        """Compute confusion matrix."""
        # Get ground truth labels (user_label if available, else pred_label if confirmed)
        # For corrected records, user_label is the ground truth
        matrix = defaultdict(lambda: defaultdict(int))
        
        for record in records:
            # Ground truth
            if record.user_label:
                gt = record.user_label
            elif record.is_confirmed:
                gt = record.pred_label
            else:
                continue  # Skip if no ground truth
            
            # Prediction
            pred = record.pred_label
            
            matrix[gt][pred] += 1
        
        return dict(matrix)
    
    def _compute_per_class_metrics(self, confusion_matrix: Dict) -> Dict[str, Dict]:
        """Compute precision, recall, F1 per class."""
        all_labels = set()
        for gt in confusion_matrix:
            all_labels.add(gt)
            for pred in confusion_matrix[gt]:
                all_labels.add(pred)
        
        metrics = {}
        for label in all_labels:
            tp = confusion_matrix.get(label, {}).get(label, 0)
            fp = sum(confusion_matrix.get(other, {}).get(label, 0) 
                    for other in all_labels if other != label)
            fn = sum(confusion_matrix.get(label, {}).get(other, 0) 
                    for other in all_labels if other != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
        
        return metrics
    
    def _compute_accuracy(self, confusion_matrix: Dict) -> float:
        """Compute overall accuracy."""
        correct = sum(confusion_matrix.get(label, {}).get(label, 0) 
                     for label in confusion_matrix)
        total = sum(sum(confusion_matrix.get(gt, {}).values()) 
                   for gt in confusion_matrix)
        
        return correct / total if total > 0 else 0.0
    
    def _compute_calibration(self, records: List[ScanRecord], bins: int = 10) -> Dict:
        """Compute confidence calibration."""
        # Group by confidence bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_data = {i: {"confidences": [], "correct": []} 
                    for i in range(bins)}
        
        for record in records:
            # Ground truth
            if record.user_label:
                gt = record.user_label
            elif record.is_confirmed:
                gt = record.pred_label
            else:
                continue
            
            pred = record.pred_label
            conf = record.pred_confidence
            
            # Find bin
            bin_idx = np.digitize(conf, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, bins - 1))
            
            bin_data[bin_idx]["confidences"].append(conf)
            bin_data[bin_idx]["correct"].append(1 if pred == gt else 0)
        
        # Compute calibration per bin
        calibration = []
        for i in range(bins):
            data = bin_data[i]
            if data["confidences"]:
                mean_conf = np.mean(data["confidences"])
                accuracy = np.mean(data["correct"])
                calibration.append({
                    "bin": i,
                    "mean_confidence": float(mean_conf),
                    "accuracy": float(accuracy),
                    "samples": len(data["confidences"])
                })
        
        return calibration
    
    def save_metrics(self, output_path: str, model_version: Optional[str] = None):
        """
        Compute and save metrics to file.
        
        Args:
            output_path: Path to save metrics JSON
            model_version: Filter by model version
        """
        metrics = self.compute_metrics(model_version)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

