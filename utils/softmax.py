"""Softmax utility function."""

import numpy as np


def softmax(logits):
    """
    Compute softmax probabilities from logits.
    
    Args:
        logits: Array-like of logit values
        
    Returns:
        numpy array of probabilities
    """
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / exp_logits.sum()

