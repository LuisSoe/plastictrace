"""
Stability module for realtime prediction smoothing and hysteresis.
"""
import numpy as np
from ml.config import CLASSES


class ProbSmoother:
    """EMA smoothing of probability vectors with renormalization."""
    
    def __init__(self, alpha=0.6):
        """
        Args:
            alpha: EMA smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = float(alpha)
        self._probs = None
    
    def update(self, probs):
        """
        Update smoothed probabilities.
        
        Args:
            probs: numpy array of shape (4,) with probabilities for [HDPE, PET, PP, PS]
        
        Returns:
            numpy array of smoothed and renormalized probabilities
        """
        probs = np.asarray(probs, dtype=np.float32)
        if len(probs) != len(CLASSES):
            raise ValueError(f"Expected {len(CLASSES)} probabilities, got {len(probs)}")
        
        if self._probs is None:
            self._probs = probs.copy()
        else:
            self._probs = self.alpha * probs + (1.0 - self.alpha) * self._probs
        
        # Renormalize to ensure sum = 1.0
        total = np.sum(self._probs)
        if total > 0:
            self._probs = self._probs / total
        else:
            self._probs = np.ones(len(CLASSES), dtype=np.float32) / len(CLASSES)
        
        return self._probs.copy()
    
    def reset(self):
        """Reset internal state."""
        self._probs = None


class HysteresisLabel:
    """Hysteresis-based label switching to prevent flicker."""
    
    def __init__(self, min_conf=0.55, switch_margin=0.10):
        """
        Args:
            min_conf: Minimum confidence to consider a label valid
            switch_margin: Margin required to switch labels (prevents flicker)
        """
        self.min_conf = float(min_conf)
        self.switch_margin = float(switch_margin)
        self._current_label = None
        self._current_conf = 0.0
    
    def update(self, label, confidence):
        """
        Update label with hysteresis logic.
        
        Args:
            label: Current predicted label
            confidence: Current confidence
        
        Returns:
            (label, confidence) tuple - may be locked to previous label if switch_margin not met
        """
        if self._current_label is None:
            # First update: accept if above min_conf
            if confidence >= self.min_conf:
                self._current_label = label
                self._current_conf = confidence
        else:
            # Subsequent updates: check if we should switch
            if label == self._current_label:
                # Same label: update confidence
                self._current_conf = confidence
            else:
                # Different label: only switch if new confidence is significantly higher
                if confidence >= self._current_conf + self.switch_margin:
                    self._current_label = label
                    self._current_conf = confidence
                # Otherwise keep current label (hysteresis)
        
        return (self._current_label, self._current_conf)
    
    def reset(self):
        """Reset internal state."""
        self._current_label = None
        self._current_conf = 0.0


def apply_confidence_gating(label, conf, threshold):
    """
    Apply confidence gating: if confidence below threshold, return "Unknown".
    
    Args:
        label: Predicted label
        conf: Confidence value (0-1)
        threshold: Confidence threshold (default 0.65)
    
    Returns:
        (gated_label, raw_label, raw_conf) tuple
    """
    raw_label = label
    raw_conf = conf
    
    if conf < threshold:
        gated_label = "Unknown"
    else:
        gated_label = label
    
    return (gated_label, raw_label, raw_conf)

