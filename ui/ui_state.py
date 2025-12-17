"""UI state machine for scan and review modes."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
from trust.decision_engine import DecisionState


class UIMode(Enum):
    """UI mode enum."""
    SCAN_MODE = "SCAN_MODE"
    REVIEW_MODE = "REVIEW_MODE"


@dataclass
class CapturedResult:
    """Captured result for review mode."""
    frame_image: np.ndarray
    plastic_type: str
    confidence: float
    confidence_band: str  # "High", "Medium", "Low"
    stability: float
    stability_text: str  # e.g., "18/20 frames agree"
    margin: float
    reason: Optional[str]
    timestamp: float
    frame_quality_notes: list


class UIStateMachine:
    """Manages UI state transitions."""
    
    def __init__(self):
        self.current_mode = UIMode.SCAN_MODE
        self.captured_result: Optional[CapturedResult] = None
        
    def can_capture(self, decision_state: DecisionState) -> bool:
        """Check if capture is allowed."""
        return decision_state == DecisionState.LOCKED
    
    def capture(self, frame: np.ndarray, decision_result, frame_quality) -> CapturedResult:
        """
        Capture current state for review mode.
        
        Args:
            frame: Current frame image
            decision_result: DecisionStateResult from trust layer
            frame_quality: FrameQuality object
            
        Returns:
            CapturedResult object
        """
        import time
        
        # Determine confidence band
        conf = decision_result.locked_confidence
        if conf >= 0.80:
            band = "High"
        elif conf >= 0.70:
            band = "Medium"
        else:
            band = "Low"
        
        # Create stability text
        vote_ratio = decision_result.vote_ratio
        frames_agree = int(vote_ratio * 20)  # Assuming window size of 20
        stability_text = f"{frames_agree}/20 frames agree"
        
        self.captured_result = CapturedResult(
            frame_image=frame.copy(),
            plastic_type=decision_result.locked_label or decision_result.current_label or "UNKNOWN",
            confidence=conf,
            confidence_band=band,
            stability=decision_result.stability,
            stability_text=stability_text,
            margin=decision_result.mean_margin,
            reason=decision_result.reason,
            timestamp=time.time(),
            frame_quality_notes=frame_quality.notes if frame_quality else []
        )
        
        self.current_mode = UIMode.REVIEW_MODE
        return self.captured_result
    
    def retake(self):
        """Return to scan mode."""
        self.current_mode = UIMode.SCAN_MODE
        self.captured_result = None
    
    def is_scan_mode(self) -> bool:
        """Check if in scan mode."""
        return self.current_mode == UIMode.SCAN_MODE
    
    def is_review_mode(self) -> bool:
        """Check if in review mode."""
        return self.current_mode == UIMode.REVIEW_MODE

