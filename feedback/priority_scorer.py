"""Priority scoring for active learning."""

from typing import Optional
from feedback.schema import ScanRecord


class PriorityScorer:
    """Computes priority scores for active learning."""
    
    def __init__(self,
                 w1: float = 0.25,  # (1 - stability)
                 w2: float = 0.25,  # entropy
                 w3: float = 0.20,  # (1 - margin_normalized)
                 w4: float = 0.15,  # user_corrected
                 w5: float = 0.15):  # poor_quality
        """
        Initialize priority scorer.
        
        Args:
            w1: Weight for (1 - stability)
            w2: Weight for entropy
            w3: Weight for (1 - margin_normalized)
            w4: Weight for user_corrected flag
            w5: Weight for poor_quality flag
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
    
    def compute_priority_score(self, record: ScanRecord) -> float:
        """
        Compute priority score for a record.
        
        Args:
            record: ScanRecord to score
            
        Returns:
            Priority score in [0, 1]
        """
        # Component 1: (1 - stability)
        stability_component = (1.0 - record.stability) * self.w1
        
        # Component 2: entropy (normalize to [0, 1])
        # Assuming max entropy is around 2.0 for 4 classes
        max_entropy = 2.0
        entropy_normalized = min(record.entropy / max_entropy, 1.0)
        entropy_component = entropy_normalized * self.w2
        
        # Component 3: (1 - margin_normalized)
        # Normalize margin (assuming max margin is around 0.5)
        max_margin = 0.5
        margin_normalized = min(record.margin / max_margin, 1.0)
        margin_component = (1.0 - margin_normalized) * self.w3
        
        # Component 4: user_corrected
        user_corrected = 1.0 if (record.user_label is not None and 
                                record.user_label != record.pred_label) else 0.0
        corrected_component = user_corrected * self.w4
        
        # Component 5: poor_quality
        poor_quality = 1.0 if (record.frame_quality.is_blurry or 
                              record.frame_quality.is_too_dark) else 0.0
        quality_component = poor_quality * self.w5
        
        # Sum components
        score = (stability_component + entropy_component + margin_component +
                corrected_component + quality_component)
        
        # Normalize to [0, 1]
        return min(max(score, 0.0), 1.0)
    
    def is_high_value(self, record: ScanRecord, threshold: float = 0.6) -> bool:
        """
        Determine if record is high-value.
        
        High-value if:
        - User corrected label
        - OR priority_score >= threshold
        - OR pred_label is OTHER/UNKNOWN
        
        Args:
            record: ScanRecord to evaluate
            threshold: Priority score threshold
            
        Returns:
            True if high-value
        """
        # User corrected
        if record.user_label is not None and record.user_label != record.pred_label:
            return True
        
        # High priority score
        if record.priority_score >= threshold:
            return True
        
        # OTHER/UNKNOWN labels
        if record.pred_label in ["OTHER", "UNKNOWN"]:
            return True
        
        return False

