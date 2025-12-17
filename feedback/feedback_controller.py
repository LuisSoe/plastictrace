"""Feedback controller for confirm/correct/unsure actions."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import time

from feedback.schema import ScanRecord, FrameQualityData, ConditionsData, ImageRefData, DeviceData
from feedback.dataset_store import DatasetStore
from feedback.priority_scorer import PriorityScorer
from ml.config import CLASSES


class FeedbackController:
    """Handles user feedback and record creation."""
    
    def __init__(self, dataset_store: DatasetStore, priority_scorer: PriorityScorer,
                 app_version: str = "1.0.0", model_version: str = "1.0.0"):
        """
        Initialize feedback controller.
        
        Args:
            dataset_store: DatasetStore instance
            priority_scorer: PriorityScorer instance
            app_version: Application version
            model_version: Model version
        """
        self.dataset_store = dataset_store
        self.priority_scorer = priority_scorer
        self.app_version = app_version
        self.model_version = model_version
    
    def confirm(self, frame: np.ndarray, roi: Optional[np.ndarray],
                pred_label: str, pred_confidence: float, stability: float,
                vote_ratio: float, margin: float, entropy: float,
                frame_quality: FrameQualityData,
                conditions: Optional[ConditionsData] = None) -> ScanRecord:
        """
        Confirm the prediction as correct.
        
        Args:
            frame: Full frame image
            roi: ROI image (optional)
            pred_label: Predicted label
            pred_confidence: Prediction confidence
            stability: Stability score
            vote_ratio: Vote ratio
            margin: Margin value
            entropy: Entropy value
            frame_quality: Frame quality data
            conditions: Optional conditions
            
        Returns:
            Created ScanRecord
        """
        # Save images
        image_ref = self._save_images(frame, roi)
        
        # Create record
        record = ScanRecord.create(
            pred_label=pred_label,
            pred_confidence=pred_confidence,
            stability=stability,
            vote_ratio=vote_ratio,
            margin=margin,
            entropy=entropy,
            frame_quality=frame_quality,
            image_ref=image_ref,
            app_version=self.app_version,
            model_version=self.model_version
        )
        
        # Set as confirmed
        record.user_label = pred_label
        record.is_confirmed = True
        
        # Set conditions if provided
        if conditions:
            record.conditions = conditions
        
        # Compute priority and high-value
        record.priority_score = self.priority_scorer.compute_priority_score(record)
        record.high_value = self.priority_scorer.is_high_value(record)
        
        # Save to store
        self.dataset_store.save_record(record)
        
        return record
    
    def correct(self, frame: np.ndarray, roi: Optional[np.ndarray],
                pred_label: str, pred_confidence: float, stability: float,
                vote_ratio: float, margin: float, entropy: float,
                frame_quality: FrameQualityData,
                user_label: str,
                conditions: Optional[ConditionsData] = None) -> ScanRecord:
        """
        Correct the prediction with user label.
        
        Args:
            frame: Full frame image
            roi: ROI image (optional)
            pred_label: Predicted label
            pred_confidence: Prediction confidence
            stability: Stability score
            vote_ratio: Vote ratio
            margin: Margin value
            entropy: Entropy value
            frame_quality: Frame quality data
            user_label: User-provided correct label
            conditions: Optional conditions
            
        Returns:
            Created ScanRecord
        """
        # Save images
        image_ref = self._save_images(frame, roi)
        
        # Create record
        record = ScanRecord.create(
            pred_label=pred_label,
            pred_confidence=pred_confidence,
            stability=stability,
            vote_ratio=vote_ratio,
            margin=margin,
            entropy=entropy,
            frame_quality=frame_quality,
            image_ref=image_ref,
            app_version=self.app_version,
            model_version=self.model_version
        )
        
        # Set user label (correction)
        record.user_label = user_label
        record.is_confirmed = False  # Not confirmed since it was corrected
        
        # Set conditions if provided
        if conditions:
            record.conditions = conditions
        
        # Compute priority and high-value
        record.priority_score = self.priority_scorer.compute_priority_score(record)
        record.high_value = self.priority_scorer.is_high_value(record)
        
        # Save to store
        self.dataset_store.save_record(record)
        
        return record
    
    def unsure(self, frame: np.ndarray, roi: Optional[np.ndarray],
               pred_label: str, pred_confidence: float, stability: float,
               vote_ratio: float, margin: float, entropy: float,
               frame_quality: FrameQualityData,
               conditions: Optional[ConditionsData] = None) -> ScanRecord:
        """
        Mark as unsure/unknown.
        
        Args:
            frame: Full frame image
            roi: ROI image (optional)
            pred_label: Predicted label
            pred_confidence: Prediction confidence
            stability: Stability score
            vote_ratio: Vote ratio
            margin: Margin value
            entropy: Entropy value
            frame_quality: Frame quality data
            conditions: Optional conditions
            
        Returns:
            Created ScanRecord
        """
        # Save images
        image_ref = self._save_images(frame, roi)
        
        # Create record
        record = ScanRecord.create(
            pred_label=pred_label,
            pred_confidence=pred_confidence,
            stability=stability,
            vote_ratio=vote_ratio,
            margin=margin,
            entropy=entropy,
            frame_quality=frame_quality,
            image_ref=image_ref,
            app_version=self.app_version,
            model_version=self.model_version
        )
        
        # Set as UNKNOWN
        record.user_label = "UNKNOWN"
        record.is_confirmed = False
        
        # Set conditions if provided
        if conditions:
            record.conditions = conditions
        
        # Compute priority and high-value (should be high due to uncertainty)
        record.priority_score = self.priority_scorer.compute_priority_score(record)
        record.high_value = True  # Always high-value for unsure cases
        
        # Save to store
        self.dataset_store.save_record(record)
        
        return record
    
    def _save_images(self, frame: np.ndarray, roi: Optional[np.ndarray]) -> ImageRefData:
        """Save images and return references."""
        timestamp = int(time.time() * 1000)
        record_id = f"scan_{timestamp}"
        
        # Save original frame
        original_path = self.dataset_store.data_dir / "images" / "original" / f"{record_id}_original.jpg"
        cv2.imwrite(str(original_path), frame)
        
        # Save snapshot (same as original for now)
        snapshot_path = self.dataset_store.data_dir / "images" / "snapshot" / f"{record_id}_snapshot.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        
        # Save ROI if provided
        roi_path = None
        if roi is not None and roi.size > 0:
            roi_path_obj = self.dataset_store.data_dir / "images" / "roi" / f"{record_id}_roi.jpg"
            cv2.imwrite(str(roi_path_obj), roi)
            roi_path = str(roi_path_obj)
        
        return ImageRefData(
            original_path=str(original_path),
            snapshot_path=str(snapshot_path),
            roi_path=roi_path
        )

