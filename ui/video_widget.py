"""
VideoWidget: Displays camera feed only (no overlays - overlays in separate widget).
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtCore import Qt, QRect
from ui.overlay_widget import OverlayWidget
from domain.models import Detection


class VideoWidget(QWidget):
    """
    Widget for displaying video frames only.
    Overlays are rendered in separate OverlayWidget on top.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Latest data
        self.latest_frame = None  # BGR numpy array
        self.latest_bbox = None  # (x1, y1, x2, y2) in frame coordinates or None
        self.latest_detection: Detection = None
        self.latest_fps = 0.0
        self.tracker_active = False
        
        # Frame dimensions (for coordinate mapping)
        self.frame_width = 640
        self.frame_height = 480
        
        # Setup overlay widget
        self.overlay_widget = OverlayWidget(self)
        self.overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.overlay_widget.raise_()  # Ensure overlay is on top
    
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
            self.overlay_widget.set_frame_dimensions(self.frame_width, self.frame_height)
        self.update()  # Trigger repaint
        self.overlay_widget.set_fps(fps)
    
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
        
        # Update overlay
        if bbox_xyxy_or_none is not None:
            bbox_widget = self._map_bbox_to_widget(
                bbox_xyxy_or_none, self.width(), self.height()
            )
            self.overlay_widget.set_bbox(bbox_widget)
        else:
            self.overlay_widget.set_bbox(None)
        
        self.overlay_widget.set_tracker_active(tracker_active)
    
    def setResult(self, result_dict):
        """
        Update classification result.
        
        Args:
            result_dict: dict with label, confidence, etc.
        """
        if result_dict:
            self.latest_detection = Detection(
                label=result_dict.get("label", "Unknown"),
                confidence=result_dict.get("confidence", 0.0),
                probs=result_dict.get("probs", []),
                raw_label=result_dict.get("raw_label", "Unknown"),
                raw_conf=result_dict.get("raw_conf", 0.0),
                bbox=self.latest_bbox
            )
        else:
            self.latest_detection = None
        
        self.update()  # Trigger repaint
        self.overlay_widget.set_detection(self.latest_detection)
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        # Resize overlay to match
        self.overlay_widget.setGeometry(0, 0, self.width(), self.height())
        self.overlay_widget.raise_()  # Ensure overlay stays on top
        # Update bbox coordinates
        if self.latest_bbox is not None:
            bbox_widget = self._map_bbox_to_widget(
                self.latest_bbox, self.width(), self.height()
            )
            self.overlay_widget.set_bbox(bbox_widget)
    
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
        """Paint video frame only (no overlays)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        widget_width = self.width()
        widget_height = self.height()
        
        # Draw background (black)
        painter.fillRect(0, 0, widget_width, widget_height, Qt.black)
        
        if self.latest_frame is None:
            # No frame: draw placeholder
            painter.setPen(Qt.white)
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(
                QRect(0, 0, widget_width, widget_height),
                Qt.AlignCenter,
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

