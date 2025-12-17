"""Overlay rendering for alignment box, stability, and quality warnings."""

import cv2
import numpy as np
from typing import Optional, List

from trust.decision_engine import DecisionState
from trust.frame_quality import FrameQuality


class OverlayRenderer:
    """Renders UI overlays on camera frames."""
    
    def __init__(self):
        self.show_alignment_box = True
        self.show_stability = True
        self.show_quality_warnings = True
    
    def render(self, frame: np.ndarray, decision_state: DecisionState,
               stability: float, frame_quality: Optional[FrameQuality],
               current_label: Optional[str] = None) -> np.ndarray:
        """
        Render all overlays on frame.
        
        Args:
            frame: Input frame (BGR)
            decision_state: Current decision state
            stability: Stability value (0..1)
            frame_quality: FrameQuality object or None
            current_label: Current predicted label
            
        Returns:
            Frame with overlays rendered
        """
        overlay = frame.copy()
        H, W = overlay.shape[:2]
        
        # Render alignment box
        if self.show_alignment_box:
            overlay = self._render_alignment_box(overlay, W, H)
        
        # Render stability indicator
        if self.show_stability:
            overlay = self._render_stability(overlay, W, H, decision_state, stability, current_label)
        
        # Render quality warnings
        if self.show_quality_warnings and frame_quality:
            overlay = self._render_quality_warnings(overlay, W, H, frame_quality, decision_state)
        
        return overlay
    
    def _render_alignment_box(self, frame: np.ndarray, W: int, H: int) -> np.ndarray:
        """Render alignment box in center."""
        # Box dimensions: 60-70% width, 45-55% height
        box_w = int(W * 0.65)
        box_h = int(H * 0.50)
        x1 = (W - box_w) // 2
        y1 = (H - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # Draw box with rounded corners effect
        color = (100, 255, 100)  # Light green
        thickness = 2
        
        # Main rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Corner accents
        corner_len = 30
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
        
        # Text hint
        text = "Align item inside the box"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        text_x = (W - tw) // 2
        text_y = y1 - 15
        if text_y < th:
            text_y = y2 + th + 15
        
        # Text background
        cv2.rectangle(frame, (text_x - 5, text_y - th - 5),
                     (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y),
                    font, 0.6, color, 1, cv2.LINE_AA)
        
        return frame
    
    def _render_stability(self, frame: np.ndarray, W: int, H: int,
                         decision_state: DecisionState, stability: float,
                         current_label: Optional[str]) -> np.ndarray:
        """Render stability indicator."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Position: top center
        y_pos = 50
        
        if decision_state == DecisionState.LOCKED:
            text = f"✓ Stable: {current_label or 'LOCKED'}"
            color = (0, 255, 0)  # Green
            icon_text = "🔒"
        elif decision_state == DecisionState.UNSTABLE:
            text = "Hold steady..."
            color = (0, 165, 255)  # Orange
            icon_text = "⚠"
        elif decision_state == DecisionState.UNKNOWN:
            text = "Uncertain — improve lighting / move closer"
            color = (0, 0, 255)  # Red
            icon_text = "❓"
        else:  # SCANNING
            text = "Scanning..."
            color = (255, 255, 0)  # Cyan
            icon_text = "🔍"
        
        # Draw stability progress bar
        bar_w = 200
        bar_h = 8
        bar_x = (W - bar_w) // 2
        bar_y = y_pos + 30
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (50, 50, 50), -1)
        # Progress
        progress_w = int(bar_w * stability)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h),
                     color, -1)
        
        # Text
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        text_x = (W - tw) // 2
        
        # Text background
        cv2.rectangle(frame, (text_x - 10, y_pos - th - 5),
                     (text_x + tw + 10, y_pos + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, y_pos),
                    font, 0.7, color, 2, cv2.LINE_AA)
        
        return frame
    
    def _render_quality_warnings(self, frame: np.ndarray, W: int, H: int,
                                 frame_quality: FrameQuality,
                                 decision_state: DecisionState) -> np.ndarray:
        """Render quality warnings with prioritization."""
        warnings = []
        
        # Priority 1: Low light
        if frame_quality.is_too_dark:
            warnings.append("Low light — turn on flashlight / move to brighter area")
        
        # Priority 2: Blur
        if frame_quality.is_blurry:
            warnings.append("Too blurry — hold still / clean lens")
        
        # Priority 3: Unstable (only if not already showing stability message)
        if decision_state == DecisionState.UNSTABLE and len(warnings) == 0:
            warnings.append("Move closer")
        
        # Show at most 2 warnings
        warnings = warnings[:2]
        
        if not warnings:
            return frame
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_start = H - 100
        
        for i, warning in enumerate(warnings):
            y_pos = y_start + i * 30
            
            # Determine color
            if "Low light" in warning:
                color = (0, 100, 255)  # Orange-red
            elif "blurry" in warning:
                color = (0, 165, 255)  # Orange
            else:
                color = (255, 255, 0)  # Yellow
            
            # Text size
            (tw, th), _ = cv2.getTextSize(warning, font, 0.6, 1)
            text_x = (W - tw) // 2
            
            # Background
            cv2.rectangle(frame, (text_x - 10, y_pos - th - 5),
                         (text_x + tw + 10, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, warning, (text_x, y_pos),
                       font, 0.6, color, 1, cv2.LINE_AA)
        
        return frame

