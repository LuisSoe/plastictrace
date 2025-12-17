"""Decision engine for lock/unlock logic with quality gates."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from trust.config import (
    LOCK_MIN_CONF, LOCK_MIN_MARGIN, LOCK_MIN_VOTE_RATIO,
    UNLOCK_VOTE_RATIO, UNKNOWN_MIN_CONF, MAX_BAD_QUALITY_FRAMES,
    STABILITY_VOTE_WEIGHT, STABILITY_MARGIN_WEIGHT
)
from trust.temporal_aggregator import TemporalAggregator
from trust.frame_quality import FrameQuality, assess_frame_quality
from ml.config import CLASSES
from utils.softmax import softmax


class DecisionState(Enum):
    """Decision state enum."""
    SCANNING = "SCANNING"
    UNSTABLE = "UNSTABLE"
    LOCKED = "LOCKED"
    UNKNOWN = "UNKNOWN"


@dataclass
class DecisionStateResult:
    """Result from decision engine."""
    state: DecisionState
    locked_label: Optional[str] = None
    locked_confidence: float = 0.0
    stability: float = 0.0
    frames_considered: int = 0
    reason: Optional[str] = None
    
    # Debug overlay fields
    current_label: Optional[str] = None
    ema_conf: float = 0.0
    vote_ratio: float = 0.0
    mean_margin: float = 0.0
    mean_entropy: float = 0.0
    frame_quality: Optional[FrameQuality] = None


class DecisionEngine:
    """Decision engine for lock/unlock logic."""
    
    def __init__(self, num_classes: int = len(CLASSES)):
        """
        Initialize decision engine.
        
        Args:
            num_classes: Number of classification classes
        """
        self.num_classes = num_classes
        self.aggregator = TemporalAggregator(num_classes)
        self.frames_considered = 0
        self.bad_quality_count = 0
        self.current_state = DecisionState.SCANNING
        self.locked_label: Optional[str] = None
        self.locked_confidence: float = 0.0
        
    def process(self, frame: np.ndarray, model_output, roi_area_ratio: float = 1.0) -> DecisionStateResult:
        """
        Process a frame and return decision state.
        
        Args:
            frame: Input frame (BGR or grayscale)
            model_output: Can be dict with 'probs' or 'logits', or array of logits/probs
            roi_area_ratio: Ratio of ROI area to full frame (default 1.0)
            
        Returns:
            DecisionStateResult with current state and debug info
        """
        # Extract probabilities from model output
        if isinstance(model_output, dict):
            if "probs" in model_output:
                probs = np.array(model_output["probs"], dtype=np.float32)
            elif "logits" in model_output:
                probs = softmax(np.array(model_output["logits"], dtype=np.float32))
            else:
                # Try to find any array-like value
                probs = np.array(list(model_output.values())[0], dtype=np.float32)
                if abs(probs.sum() - 1.0) > 0.1:
                    probs = softmax(probs)
        else:
            # Assume it's an array
            probs = np.array(model_output, dtype=np.float32)
            if abs(probs.sum() - 1.0) > 0.1:
                probs = softmax(probs)
        
        # Assess frame quality
        quality = assess_frame_quality(frame, roi_area_ratio)
        
        # Update bad quality counter
        if quality.is_blurry or quality.is_too_dark:
            self.bad_quality_count += 1
        else:
            # Decay bad quality count (allow recovery)
            self.bad_quality_count = max(0, self.bad_quality_count - 1)
        
        # Update aggregator
        agg_result = self.aggregator.update(probs)
        
        self.frames_considered += 1
        
        # Get aggregated metrics
        vote_label = agg_result["vote_label"]
        vote_ratio = agg_result["vote_ratio"]
        ema_label = agg_result["ema_label"]
        ema_conf = agg_result["ema_conf"]
        mean_margin = agg_result["mean_margin"]
        mean_entropy = agg_result["mean_entropy"]
        
        # Map label indices to strings
        current_label_str = CLASSES[ema_label] if ema_label < len(CLASSES) else "UNKNOWN"
        vote_label_str = CLASSES[vote_label] if vote_label < len(CLASSES) else "UNKNOWN"
        
        # Compute stability
        normalized_margin = np.clip(mean_margin / 0.5, 0.0, 1.0)  # Normalize to 0-1
        stability = np.clip(
            STABILITY_VOTE_WEIGHT * vote_ratio + STABILITY_MARGIN_WEIGHT * normalized_margin,
            0.0, 1.0
        )
        
        # Decision logic
        reason = None
        new_state = self.current_state
        
        # Rule 1: Bad quality persistence → UNKNOWN
        if self.bad_quality_count >= MAX_BAD_QUALITY_FRAMES:
            new_state = DecisionState.UNKNOWN
            reason = "bad_quality"
            if self.current_state == DecisionState.LOCKED:
                self.locked_label = None
                self.locked_confidence = 0.0
        
        # Rule 2: Lock conditions (only if quality is ok)
        elif (not quality.is_blurry and not quality.is_too_dark and
              ema_conf >= LOCK_MIN_CONF and
              mean_margin >= LOCK_MIN_MARGIN and
              vote_ratio >= LOCK_MIN_VOTE_RATIO):
            new_state = DecisionState.LOCKED
            self.locked_label = vote_label_str
            self.locked_confidence = ema_conf
            reason = "locked"
        
        # Rule 3: Unlock conditions (if currently locked)
        elif self.current_state == DecisionState.LOCKED:
            if (vote_ratio < UNLOCK_VOTE_RATIO or
                quality.is_blurry or quality.is_too_dark or
                self.bad_quality_count >= MAX_BAD_QUALITY_FRAMES):
                new_state = DecisionState.UNSTABLE
                self.locked_label = None
                self.locked_confidence = 0.0
                reason = "unlocked"
            else:
                # Stay locked, but update confidence
                self.locked_confidence = ema_conf
                reason = "locked"
        
        # Rule 4: UNKNOWN conditions
        elif ema_conf < UNKNOWN_MIN_CONF or mean_entropy > 1.5:  # High entropy threshold
            new_state = DecisionState.UNKNOWN
            if mean_entropy > 1.5:
                reason = "high_entropy"
            else:
                reason = "low_confidence"
        
        # Rule 5: SCANNING vs UNSTABLE
        else:
            if vote_ratio < 0.5 or quality.is_blurry or quality.is_too_dark:
                new_state = DecisionState.UNSTABLE
                reason = "unstable"
            else:
                new_state = DecisionState.SCANNING
                reason = "scanning"
        
        self.current_state = new_state
        
        # Build result
        result = DecisionStateResult(
            state=new_state,
            locked_label=self.locked_label,
            locked_confidence=self.locked_confidence,
            stability=stability,
            frames_considered=self.frames_considered,
            reason=reason,
            current_label=current_label_str,
            ema_conf=ema_conf,
            vote_ratio=vote_ratio,
            mean_margin=mean_margin,
            mean_entropy=mean_entropy,
            frame_quality=quality
        )
        
        return result
    
    def reset(self):
        """Reset decision engine state."""
        self.aggregator.reset()
        self.frames_considered = 0
        self.bad_quality_count = 0
        self.current_state = DecisionState.SCANNING
        self.locked_label = None
        self.locked_confidence = 0.0


# Convenience function for integration
def process_frame(frame: np.ndarray, model_output, roi_area_ratio: float = 1.0,
                  engine: Optional[DecisionEngine] = None) -> Tuple[DecisionStateResult, DecisionEngine]:
    """
    Process a frame and return decision state.
    
    Args:
        frame: Input frame (BGR or grayscale)
        model_output: Model output (dict with 'probs'/'logits' or array)
        roi_area_ratio: Ratio of ROI area to full frame (default 1.0)
        engine: Optional DecisionEngine instance (creates new one if None)
        
    Returns:
        Tuple of (DecisionStateResult, DecisionEngine)
    """
    if engine is None:
        engine = DecisionEngine()
    
    result = engine.process(frame, model_output, roi_area_ratio)
    return result, engine

