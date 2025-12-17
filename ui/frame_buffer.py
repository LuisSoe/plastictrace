"""Recent frame buffer for best frame selection."""

import numpy as np
from collections import deque
from typing import Optional
from dataclasses import dataclass
import time

from trust.frame_quality import FrameQuality
from trust.decision_engine import DecisionState


@dataclass
class BufferedFrame:
    """Frame with metadata in buffer."""
    frame_image: np.ndarray
    timestamp: float
    frame_quality: FrameQuality
    decision_state: DecisionState
    locked_confidence: float
    stability: float
    is_locked: bool


class RecentFrameBuffer:
    """Ring buffer for recent frames with scoring."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def add(self, frame: np.ndarray, frame_quality: FrameQuality,
            decision_state: DecisionState, locked_confidence: float,
            stability: float):
        """Add a frame to the buffer."""
        buffered = BufferedFrame(
            frame_image=frame.copy(),
            timestamp=time.time(),
            frame_quality=frame_quality,
            decision_state=decision_state,
            locked_confidence=locked_confidence,
            stability=stability,
            is_locked=(decision_state == DecisionState.LOCKED)
        )
        self.buffer.append(buffered)
    
    def get_best_frame(self) -> Optional[BufferedFrame]:
        """
        Select best frame based on scoring.
        
        Score = w1*(not_blurry) + w2*(not_dark) + w3*(locked_confidence) + w4*(stability)
        
        Returns:
            Best BufferedFrame or None if buffer is empty
        """
        if len(self.buffer) == 0:
            return None
        
        best_frame = None
        best_score = -1.0
        
        # Weights
        w1 = 0.3  # not_blurry
        w2 = 0.2  # not_dark
        w3 = 0.3  # locked_confidence
        w4 = 0.2  # stability
        
        for buffered in self.buffer:
            # Only consider locked frames
            if not buffered.is_locked:
                continue
            
            score = 0.0
            score += w1 * (1.0 if not buffered.frame_quality.is_blurry else 0.0)
            score += w2 * (1.0 if not buffered.frame_quality.is_too_dark else 0.0)
            score += w3 * buffered.locked_confidence
            score += w4 * buffered.stability
            
            if score > best_score:
                best_score = score
                best_frame = buffered
        
        # If no locked frames, return most recent frame that's not blurry/dark
        if best_frame is None:
            for buffered in reversed(self.buffer):
                if (not buffered.frame_quality.is_blurry and 
                    not buffered.frame_quality.is_too_dark):
                    return buffered
        
        # Fallback: most recent frame
        if best_frame is None and len(self.buffer) > 0:
            return self.buffer[-1]
        
        return best_frame
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

