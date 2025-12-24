"""
OverlayWidget: Transparent overlay for bbox, labels, and confidence bars.
Separate from video rendering for clean, anti-flicker display.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QRect
from domain.models import Detection


class OverlayWidget(QWidget):
    """
    Transparent overlay widget for drawing bbox, labels, and confidence bars.
    Positioned on top of VideoWidget.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        # State
        self.bbox = None  # (x1, y1, x2, y2) in widget coordinates
        self.detection: Detection = None
        self.fps = 0.0
        self.tracker_active = False
        
        # Frame dimensions (for coordinate mapping)
        self.frame_width = 640
        self.frame_height = 480
    
    def set_bbox(self, bbox_widget_coords):
        """Set bounding box in widget coordinates."""
        self.bbox = bbox_widget_coords
        self.update()
    
    def set_detection(self, detection: Detection):
        """Set detection result."""
        self.detection = detection
        self.update()
    
    def set_fps(self, fps: float):
        """Set FPS value."""
        self.fps = fps
        self.update()
    
    def set_tracker_active(self, active: bool):
        """Set tracker active state."""
        self.tracker_active = active
        self.update()
    
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions for coordinate mapping."""
        self.frame_width = width
        self.frame_height = height
    
    def paintEvent(self, event):
        """Paint overlay elements."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        widget_width = self.width()
        widget_height = self.height()
        
        # Draw bbox
        if self.bbox is not None:
            self._draw_bbox(painter, self.bbox)
            
            # Draw label + confidence near bbox
            if self.detection:
                self._draw_label_confidence(painter, self.detection, self.bbox)
        
        # Draw top panel with recommendation
        if self.detection and not self.detection.is_unknown:
            self._draw_top_panel(painter, self.detection, widget_width)
        
        # Draw FPS (top-right)
        self._draw_fps(painter, self.fps, widget_width)
        
        # Draw status (bottom-left)
        self._draw_status(painter, widget_width, widget_height)
    
    def _draw_bbox(self, painter: QPainter, bbox: tuple):
        """Draw bounding box with corner accents."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Main rectangle
        pen = QPen(QColor(16, 185, 129), 2)  # Green accent
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        
        # Corner accents
        corner_len = 20
        pen.setWidth(4)
        painter.setPen(pen)
        
        # Top-left
        painter.drawLine(x1, y1, x1 + corner_len, y1)
        painter.drawLine(x1, y1, x1, y1 + corner_len)
        
        # Top-right
        painter.drawLine(x2 - corner_len, y1, x2, y1)
        painter.drawLine(x2, y1, x2, y1 + corner_len)
        
        # Bottom-left
        painter.drawLine(x1, y2 - corner_len, x1, y2)
        painter.drawLine(x1, y2, x1 + corner_len, y2)
        
        # Bottom-right
        painter.drawLine(x2 - corner_len, y2, x2, y2)
        painter.drawLine(x2, y2 - corner_len, x2, y2)
    
    def _draw_label_confidence(self, painter: QPainter, detection: Detection, bbox: tuple):
        """Draw label and confidence near bbox."""
        x1, y1, x2, y2 = bbox
        
        # Position text above bbox
        text_x = int(x1)
        text_y = int(y1) - 10
        
        if text_y < 30:  # If too close to top, put below bbox
            text_y = int(y2) + 25
        
        label = detection.label
        conf = detection.confidence
        
        # Draw background rectangle
        text = f"{label} {conf*100:.1f}%"
        font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(text)
        
        bg_rect = QRect(
            text_x - 6, text_y - text_rect.height() - 4,
            text_rect.width() + 12, text_rect.height() + 8
        )
        
        painter.fillRect(bg_rect, QColor(0, 0, 0, 200))
        
        # Draw text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(text_x, text_y, text)
    
    def _draw_top_panel(self, painter: QPainter, detection: Detection, widget_width: int):
        """Draw top panel with recommendation."""
        from ml.config import RECOMMENDATION
        
        recommendation = RECOMMENDATION.get(detection.label, "")
        if not recommendation:
            return
        
        # Split recommendation into lines
        words = recommendation.split()
        lines = []
        current_line = ""
        
        font = QFont("Arial", 12)
        painter.setFont(font)
        fm = painter.fontMetrics()
        max_width = widget_width - 40
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if fm.width(test_line) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                if len(lines) >= 2:  # Max 3 lines
                    break
        
        if current_line:
            lines.append(current_line)
        
        lines = lines[:3]
        
        if not lines:
            return
        
        # Calculate panel height
        line_height = fm.height()
        padding = 12
        panel_height = len(lines) * line_height + padding * 2 + 30  # Extra for label
        
        # Draw semi-transparent background
        painter.fillRect(0, 0, widget_width, panel_height, QColor(0, 0, 0, 180))
        
        # Draw label
        label_font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(label_font)
        painter.setPen(QColor(16, 185, 129))  # Green accent
        painter.drawText(20, 25, f"{detection.label} ({detection.confidence*100:.1f}%)")
        
        # Draw recommendation lines
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        y_offset = padding + line_height + 30
        for line in lines:
            painter.drawText(20, y_offset, line)
            y_offset += line_height
    
    def _draw_fps(self, painter: QPainter, fps: float, widget_width: int):
        """Draw FPS counter (top-right)."""
        text = f"FPS: {fps:.1f}"
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(text)
        
        text_x = widget_width - text_rect.width() - 20
        text_y = 25
        
        # Background
        bg_rect = QRect(
            text_x - 6, text_y - text_rect.height() - 4,
            text_rect.width() + 12, text_rect.height() + 8
        )
        painter.fillRect(bg_rect, QColor(0, 0, 0, 200))
        
        # Text in yellow
        painter.setPen(QColor(255, 255, 0))
        painter.drawText(text_x, text_y, text)
    
    def _draw_status(self, painter: QPainter, widget_width: int, widget_height: int):
        """Draw status text (bottom-left)."""
        bbox_status = "YES" if self.bbox is not None else "NO"
        result_status = "YES" if self.detection is not None else "NO"
        tracker_status = "ACTIVE" if self.tracker_active else "INACTIVE"
        
        text = f"bbox {bbox_status} | result {result_status} | tracker {tracker_status}"
        
        font = QFont("Arial", 10)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(text)
        
        text_x = 10
        text_y = widget_height - 10
        
        # Background
        bg_rect = QRect(
            text_x - 6, text_y - text_rect.height() - 4,
            text_rect.width() + 12, text_rect.height() + 8
        )
        painter.fillRect(bg_rect, QColor(0, 0, 0, 200))
        
        # Text in cyan
        painter.setPen(QColor(0, 255, 255))
        painter.drawText(text_x, text_y, text)

