"""Temporal aggregation for anti-flicker using rolling window and EMA."""

import numpy as np
from collections import deque
from typing import Tuple, Optional

from trust.config import N, EMA_ALPHA
from utils.softmax import softmax


class TemporalAggregator:
    """Maintains rolling window and EMA for temporal smoothing."""
    
    def __init__(self, num_classes: int, window_size: int = N, ema_alpha: float = EMA_ALPHA):
        """
        Initialize temporal aggregator.
        
        Args:
            num_classes: Number of classification classes
            window_size: Size of rolling window (default: N from config)
            ema_alpha: EMA smoothing factor (default: EMA_ALPHA from config)
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        
        # Rolling window: store (label_idx, p1, margin, entropy) for each frame
        self.window: deque = deque(maxlen=window_size)
        
        # EMA over probabilities
        self.ema_probs: Optional[np.ndarray] = None
        
    def update(self, probs: np.ndarray) -> dict:
        """
        Update aggregator with new frame probabilities.
        
        Args:
            probs: Probability array (can be logits, will be softmaxed if needed)
            
        Returns:
            Dictionary with aggregated metrics:
            - vote_label: most frequent label in window
            - vote_ratio: count(vote_label) / window_size
            - ema_label: argmax(ema_probs)
            - ema_conf: max(ema_probs)
            - mean_margin: average margin in window
            - mean_entropy: average entropy in window
        """
        # Ensure probs is a numpy array
        probs = np.array(probs, dtype=np.float32)
        
        # If probs don't sum to ~1.0, assume they're logits and softmax them
        if abs(probs.sum() - 1.0) > 0.1:
            probs = softmax(probs)
        
        # Ensure probabilities are valid
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        probs = probs / probs.sum()  # Renormalize
        
        # Compute metrics for this frame
        label_idx = int(np.argmax(probs))
        p1 = float(probs[label_idx])
        
        # Get second max
        sorted_probs = np.sort(probs)[::-1]
        p2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = p1 - p2
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs))
        
        # Add to rolling window
        self.window.append((label_idx, p1, margin, entropy))
        
        # Update EMA
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.ema_alpha * self.ema_probs + (1 - self.ema_alpha) * probs
            # Renormalize EMA probs
            self.ema_probs = self.ema_probs / self.ema_probs.sum()
        
        # Compute aggregated metrics
        if len(self.window) == 0:
            # Fallback if window is empty
            return {
                "vote_label": label_idx,
                "vote_ratio": 0.0,
                "ema_label": label_idx,
                "ema_conf": p1,
                "mean_margin": margin,
                "mean_entropy": entropy,
            }
        
        # Vote label: most frequent label in window
        label_counts = {}
        label_p1_sums = {}
        for lbl_idx, p1_val, _, _ in self.window:
            label_counts[lbl_idx] = label_counts.get(lbl_idx, 0) + 1
            label_p1_sums[lbl_idx] = label_p1_sums.get(lbl_idx, 0.0) + p1_val
        
        # Find most frequent label, break ties by higher mean p1
        vote_label = max(label_counts.keys(), 
                        key=lambda k: (label_counts[k], label_p1_sums[k] / label_counts[k]))
        vote_ratio = label_counts[vote_label] / len(self.window)
        
        # EMA metrics
        ema_label = int(np.argmax(self.ema_probs))
        ema_conf = float(self.ema_probs[ema_label])
        
        # Mean margin and entropy
        margins = [m for _, _, m, _ in self.window]
        entropies = [e for _, _, _, e in self.window]
        mean_margin = float(np.mean(margins))
        mean_entropy = float(np.mean(entropies))
        
        return {
            "vote_label": vote_label,
            "vote_ratio": vote_ratio,
            "ema_label": ema_label,
            "ema_conf": ema_conf,
            "mean_margin": mean_margin,
            "mean_entropy": mean_entropy,
        }
    
    def reset(self):
        """Reset aggregator state."""
        self.window.clear()
        self.ema_probs = None

