"""Dataset export pipeline."""

import csv
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from feedback.schema import ScanRecord
from feedback.dataset_store import DatasetStore


class DatasetExporter:
    """Exports datasets in various formats."""
    
    def __init__(self, dataset_store: DatasetStore):
        """
        Initialize dataset exporter.
        
        Args:
            dataset_store: DatasetStore instance
        """
        self.dataset_store = dataset_store
    
    def export(self, output_dir: str, mode: str = "all",
               label_filter: Optional[str] = None) -> Path:
        """
        Export dataset.
        
        Args:
            output_dir: Output directory path
            mode: Export mode: "all", "high_value", "corrected", "by_label"
            label_filter: Label to filter by (for "by_label" mode)
            
        Returns:
            Path to exported dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get records based on mode
        if mode == "all":
            records = self.dataset_store.get_all_records()
        elif mode == "high_value":
            records = self.dataset_store.get_high_value_records()
        elif mode == "corrected":
            records = self.dataset_store.get_corrected_records()
        elif mode == "by_label":
            if label_filter is None:
                raise ValueError("label_filter required for 'by_label' mode")
            records = self.dataset_store.get_by_label(label_filter)
        else:
            raise ValueError(f"Unknown export mode: {mode}")
        
        # Create images directory
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Copy images and create labels CSV
        labels_data = []
        for record in records:
            # Determine label (user_label if available, else pred_label if confirmed)
            label = record.user_label if record.user_label else (
                record.pred_label if record.is_confirmed else record.pred_label
            )
            
            # Copy snapshot image
            snapshot_src = Path(record.image_ref.snapshot_path)
            if snapshot_src.exists():
                snapshot_dst = images_dir / f"{record.id}.jpg"
                shutil.copy2(snapshot_src, snapshot_dst)
                
                # Add to labels
                labels_data.append({
                    "filename": snapshot_dst.name,
                    "user_label": label,
                    "pred_label": record.pred_label,
                    "confidence": record.pred_confidence,
                    "stability": record.stability,
                    "clean": record.conditions.clean,
                    "label_present": record.conditions.label_present,
                    "crushed": record.conditions.crushed,
                    "mixed": record.conditions.mixed,
                    "is_confirmed": record.is_confirmed,
                    "priority_score": record.priority_score,
                    "high_value": record.high_value
                })
        
        # Write labels.csv
        if labels_data:
            csv_path = output_path / "labels.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=labels_data[0].keys())
                writer.writeheader()
                writer.writerows(labels_data)
        
        # Write metadata.json
        metadata = {
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "export_mode": mode,
            "label_filter": label_filter,
            "total_samples": len(labels_data),
            "app_version": records[0].app_version if records else "unknown",
            "model_version": records[0].model_version if records else "unknown",
            "schema_version": records[0].schema_version if records else "1.0.0"
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def export_all(self, output_dir: str) -> Path:
        """Export all records."""
        return self.export(output_dir, mode="all")
    
    def export_high_value(self, output_dir: str) -> Path:
        """Export high-value records only."""
        return self.export(output_dir, mode="high_value")
    
    def export_corrected(self, output_dir: str) -> Path:
        """Export corrected records only."""
        return self.export(output_dir, mode="corrected")
    
    def export_by_label(self, output_dir: str, label: str) -> Path:
        """Export records by label."""
        return self.export(output_dir, mode="by_label", label_filter=label)

