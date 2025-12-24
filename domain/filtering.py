"""
Filtering and smoothing logic for predictions.
"""
import numpy as np
from typing import Tuple, Optional
from ml.config import CLASSES


class TemporalSmoother:
    """Enhanced temporal EMA smoothing with frame persistence."""
    
    def __init__(self, alpha=0.5, persistence_frames=5):
        """
        Args:
            alpha: EMA smoothing factor (0-1). Higher = more responsive.
            persistence_frames: Number of frames a label must persist before switching.
        """
        self.alpha = float(alpha)
        self.persistence_frames = int(persistence_frames)
        self._probs = None
        self._current_label = None
        self._label_persistence_count = 0
    
    def update(self, probs: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Update smoothed probabilities with persistence logic.
        
        Args:
            probs: numpy array of shape (n_classes,) with probabilities
        
        Returns:
            (smoothed_probs, label, confidence) tuple
        """
        probs = np.asarray(probs, dtype=np.float32)
        if len(probs) != len(CLASSES):
            raise ValueError(f"Expected {len(CLASSES)} probabilities, got {len(probs)}")
        
        # EMA smoothing
        if self._probs is None:
            self._probs = probs.copy()
        else:
            self._probs = self.alpha * probs + (1.0 - self.alpha) * self._probs
        
        # Renormalize
        total = np.sum(self._probs)
        if total > 0:
            self._probs = self._probs / total
        else:
            self._probs = np.ones(len(CLASSES), dtype=np.float32) / len(CLASSES)
        
        # Get current prediction
        idx = int(np.argmax(self._probs))
        new_label = CLASSES[idx]
        new_conf = float(self._probs[idx])
        
        # Persistence logic: only switch if label persists for N frames
        if new_label == self._current_label:
            self._label_persistence_count += 1
        else:
            if self._label_persistence_count >= self.persistence_frames:
                # Switch allowed
                self._current_label = new_label
                self._label_persistence_count = 1
            else:
                # Keep current label, increment persistence
                self._label_persistence_count += 1
        
        # If no current label, set it
        if self._current_label is None:
            self._current_label = new_label
            self._label_persistence_count = 1
        
        return (self._probs.copy(), self._current_label, new_conf)
    
    def reset(self):
        """Reset internal state."""
        self._probs = None
        self._current_label = None
        self._label_persistence_count = 0


class HysteresisGate:
    """Hysteresis + gating for label stability."""
    
    def __init__(self, min_conf=0.65, switch_margin=0.10):
        """
        Args:
            min_conf: Minimum confidence to accept a label (gating)
            switch_margin: Margin required to switch labels (hysteresis)
        """
        self.min_conf = float(min_conf)
        self.switch_margin = float(switch_margin)
        self._current_label = None
        self._current_conf = 0.0
    
    def update(self, label: str, confidence: float) -> Tuple[str, float]:
        """
        Update label with hysteresis and gating.
        
        Args:
            label: Current predicted label
            confidence: Current confidence
        
        Returns:
            (gated_label, confidence) tuple
        """
        # Gating: reject if below threshold
        if confidence < self.min_conf:
            return ("Unknown", confidence)
        
        # Hysteresis logic
        if self._current_label is None:
            # First update: accept if above threshold
            self._current_label = label
            self._current_conf = confidence
        else:
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

