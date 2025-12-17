"""Frame quality assessment using blur and brightness heuristics."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List

from trust.config import BLUR_MIN, BRIGHTNESS_MIN


@dataclass
class FrameQuality:
    """Frame quality metrics."""
    is_blurry: bool = False
    blur_score: float = 0.0
    brightness: float = 0.0
    is_too_dark: bool = False
    roi_area_ratio: float = 1.0  # If ROI exists; else set to 1.0
    notes: List[str] = field(default_factory=list)


def compute_blur_score(frame: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        Variance of Laplacian (higher = sharper)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def compute_brightness(frame: np.ndarray) -> float:
    """
    Compute mean grayscale luminance.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        Mean brightness value (0-255)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    return float(np.mean(gray))


def assess_frame_quality(frame: np.ndarray, roi_area_ratio: float = 1.0) -> FrameQuality:
    """
    Assess frame quality using blur and brightness heuristics.
    
    Args:
        frame: Input frame (BGR or grayscale)
        roi_area_ratio: Ratio of ROI area to full frame (default 1.0)
        
    Returns:
        FrameQuality object with assessment results
    """
    blur_score = compute_blur_score(frame)
    brightness = compute_brightness(frame)
    
    is_blurry = blur_score < BLUR_MIN
    is_too_dark = brightness < BRIGHTNESS_MIN
    
    notes = []
    if is_blurry:
        notes.append("blurry")
    if is_too_dark:
        notes.append("low_light")
    
    return FrameQuality(
        is_blurry=is_blurry,
        blur_score=blur_score,
        brightness=brightness,
        is_too_dark=is_too_dark,
        roi_area_ratio=roi_area_ratio,
        notes=notes
    )

