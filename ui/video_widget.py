"""
Video widget for displaying camera feed with overlays.
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtCore import Qt
from ui.overlay import (
    draw_bbox, draw_label_confidence, draw_top_panel,
    draw_fps, draw_status
)
from ml.config import RECOMMENDATION


class VideoWidget(QWidget):
    """Widget for displaying video frames with overlays."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Latest data
        self.latest_frame = None  # BGR numpy array
        self.latest_bbox = None  # (x1, y1, x2, y2) in frame coordinates or None
        self.latest_result = None  # dict with label, confidence, etc.
        self.latest_fps = 0.0
        self.tracker_active = False
        
        # Overlay settings
        self.overlay_alpha = 0.6
        
        # Frame dimensions (for coordinate mapping)
        self.frame_width = 640
        self.frame_height = 480
    
    def setFrame(self, frame_bgr, fps):
        """
        Update frame and FPS.
        
        Args:
            frame_bgr: BGR numpy array or None
            fps: FPS value
        """
        self.latest_frame = frame_bgr
        self.latest_fps = fps
        if frame_bgr is not None and frame_bgr.size > 0:
            self.frame_height, self.frame_width = frame_bgr.shape[:2]
        self.update()  # Trigger repaint
    
    def setBBox(self, bbox_xyxy_or_none, tracker_active):
        """
        Update bounding box.
        
        Args:
            bbox_xyxy_or_none: (x1, y1, x2, y2) in frame coordinates or None
            tracker_active: Whether tracker is active
        """
        self.latest_bbox = bbox_xyxy_or_none
        self.tracker_active = tracker_active
        self.update()  # Trigger repaint
    
    def setResult(self, result):
        """
        Update classification result.
        
        Args:
            result: dict with label, confidence, etc.
        """
        self.latest_result = result
        self.update()  # Trigger repaint
    
    def setOverlayAlpha(self, alpha):
        """Set overlay panel transparency (0-1)."""
        self.overlay_alpha = float(alpha)
        self.update()
    
    def _map_bbox_to_widget(self, bbox_frame, widget_width, widget_height):
        """
        Map bbox from frame coordinates to widget coordinates.
        
        Args:
            bbox_frame: (x1, y1, x2, y2) in frame coordinates
            widget_width: Widget width
            widget_height: Widget height
        
        Returns:
            (x1, y1, x2, y2) in widget coordinates
        """
        if bbox_frame is None:
            return None
        
        x1_f, y1_f, x2_f, y2_f = bbox_frame
        
        # Calculate scaling to fit frame in widget (keep aspect ratio)
        scale_x = widget_width / self.frame_width
        scale_y = widget_height / self.frame_height
        scale = min(scale_x, scale_y)
        
        # Calculate offset (centering)
        scaled_frame_w = self.frame_width * scale
        scaled_frame_h = self.frame_height * scale
        offset_x = (widget_width - scaled_frame_w) / 2
        offset_y = (widget_height - scaled_frame_h) / 2
        
        # Map coordinates
        x1_w = offset_x + x1_f * scale
        y1_w = offset_y + y1_f * scale
        x2_w = offset_x + x2_f * scale
        y2_w = offset_y + y2_f * scale
        
        return (x1_w, y1_w, x2_w, y2_w)
    
    def paintEvent(self, event):
        """Paint video frame and overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        widget_width = self.width()
        widget_height = self.height()
        
        # Draw background (black)
        painter.fillRect(0, 0, widget_width, widget_height, Qt.black)
        
        if self.latest_frame is None:
            # No frame: draw placeholder
            painter.setPen(Qt.white)
            painter.drawText(
                widget_width // 2 - 100,
                widget_height // 2,
                "No camera feed"
            )
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget (keep aspect ratio)
        scaled_image = qt_image.scaled(
            widget_width, widget_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Convert QImage to QPixmap
        scaled_pixmap = QPixmap.fromImage(scaled_image)
        
        # Center the image
        pixmap_x = (widget_width - scaled_pixmap.width()) // 2
        pixmap_y = (widget_height - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(pixmap_x, pixmap_y, scaled_pixmap)
        
        # Draw overlays
        # Map bbox to widget coordinates
        bbox_widget = self._map_bbox_to_widget(
            self.latest_bbox, widget_width, widget_height
        )
        
        # Draw bbox
        if bbox_widget is not None:
            draw_bbox(painter, bbox_widget)
            
            # Draw label + confidence near bbox
            if self.latest_result:
                label = self.latest_result.get("label", "Unknown")
                confidence = self.latest_result.get("confidence", 0.0)
                draw_label_confidence(painter, label, confidence, bbox_widget)
        
        # Draw top panel with recommendation
        if self.latest_result:
            label = self.latest_result.get("label", "")
            if label and label != "Unknown":
                recommendation = RECOMMENDATION.get(label, "")
                if recommendation:
                    draw_top_panel(
                        painter, recommendation, widget_width,
                        alpha=self.overlay_alpha, max_lines=3
                    )
        
        # Draw FPS (yellow, top-right)
        draw_fps(painter, self.latest_fps, widget_width, widget_height)
        
        # Draw status (cyan, bottom-left)
        draw_status(
            painter,
            bbox_exists=(self.latest_bbox is not None),
            result_exists=(self.latest_result is not None),
            tracker_active=self.tracker_active,
            widget_width=widget_width,
            widget_height=widget_height
        )

