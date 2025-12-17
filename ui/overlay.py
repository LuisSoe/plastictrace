"""
Overlay drawing utilities for video widget.
"""
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt
from ml.config import RECOMMENDATION


def draw_bbox(painter, bbox_widget, pen_width=2):
    """
    Draw bounding box in green.
    
    Args:
        painter: QPainter
        bbox_widget: (x1, y1, x2, y2) in widget coordinates
        pen_width: Line width
    """
    x1, y1, x2, y2 = bbox_widget
    
    pen = QPen(QColor(0, 255, 0), pen_width)  # Green
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def draw_label_confidence(painter, label, confidence, bbox_widget, font_size=14):
    """
    Draw label and confidence near bbox.
    
    Args:
        painter: QPainter
        label: Label string
        confidence: Confidence value (0-1)
        bbox_widget: (x1, y1, x2, y2) in widget coordinates
        font_size: Font size
    """
    x1, y1, x2, y2 = bbox_widget
    
    # Position text above bbox
    text_x = int(x1)
    text_y = int(y1) - 5
    
    if text_y < 20:  # If too close to top, put below bbox
        text_y = int(y2) + 20
    
    # Draw background rectangle for text
    text = f"{label} {confidence:.2f}"
    font = QFont("Arial", font_size, QFont.Bold)
    painter.setFont(font)
    fm = painter.fontMetrics()
    text_rect = fm.boundingRect(text)
    
    bg_rect_x = text_x - 4
    bg_rect_y = text_y - text_rect.height() - 2
    bg_rect_w = text_rect.width() + 8
    bg_rect_h = text_rect.height() + 4
    
    # Semi-transparent background
    painter.fillRect(
        bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h,
        QColor(0, 0, 0, 180)
    )
    
    # Draw text
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(text_x, text_y, text)


def draw_top_panel(painter, recommendation, widget_width, alpha=0.6, max_lines=3):
    """
    Draw translucent top panel with recommendation.
    
    Args:
        painter: QPainter
        recommendation: Recommendation text (max 3 lines)
        widget_width: Widget width
        alpha: Transparency (0-1)
        max_lines: Maximum lines to display
    """
    if not recommendation:
        return
    
    # Split recommendation into lines
    words = recommendation.split()
    lines = []
    current_line = ""
    
    font = QFont("Arial", 12)
    painter.setFont(font)
    fm = painter.fontMetrics()
    max_width = widget_width - 40  # Margins
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if fm.width(test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
            if len(lines) >= max_lines - 1:
                break
    
    if current_line:
        lines.append(current_line)
    
    # Limit to max_lines
    lines = lines[:max_lines]
    
    if not lines:
        return
    
    # Calculate panel height
    line_height = fm.height()
    padding = 12
    panel_height = len(lines) * line_height + padding * 2
    
    # Draw semi-transparent background
    bg_color = QColor(0, 0, 0, int(255 * alpha))
    painter.fillRect(0, 0, widget_width, panel_height, bg_color)
    
    # Draw text
    painter.setPen(QColor(255, 255, 255))
    y_offset = padding + line_height
    for line in lines:
        painter.drawText(20, y_offset, line)
        y_offset += line_height


def draw_fps(painter, fps, widget_width, widget_height, font_size=14):
    """
    Draw FPS counter in yellow (top-right).
    
    Args:
        painter: QPainter
        fps: FPS value
        widget_width: Widget width
        widget_height: Widget height
        font_size: Font size
    """
    text = f"FPS: {fps:.1f}"
    font = QFont("Arial", font_size, QFont.Bold)
    painter.setFont(font)
    fm = painter.fontMetrics()
    text_rect = fm.boundingRect(text)
    
    # Position top-right
    text_x = widget_width - text_rect.width() - 20
    text_y = 25
    
    # Draw background
    bg_rect_x = text_x - 4
    bg_rect_y = text_y - text_rect.height() - 2
    bg_rect_w = text_rect.width() + 8
    bg_rect_h = text_rect.height() + 4
    
    painter.fillRect(
        bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h,
        QColor(0, 0, 0, 180)
    )
    
    # Draw text in yellow
    painter.setPen(QColor(255, 255, 0))  # Yellow
    painter.drawText(text_x, text_y, text)


def draw_status(painter, bbox_exists, result_exists, tracker_active, widget_width, widget_height, font_size=12):
    """
    Draw status text in cyan (bottom-left).
    
    Args:
        painter: QPainter
        bbox_exists: Whether bbox exists
        result_exists: Whether result exists
        tracker_active: Whether tracker is active
        widget_width: Widget width
        widget_height: Widget height
        font_size: Font size
    """
    bbox_status = "YES" if bbox_exists else "NO"
    result_status = "YES" if result_exists else "NO"
    tracker_status = "ACTIVE" if tracker_active else "INACTIVE"
    
    text = f"bbox {bbox_status} | result {result_status} | tracker {tracker_status}"
    
    font = QFont("Arial", font_size)
    painter.setFont(font)
    fm = painter.fontMetrics()
    text_rect = fm.boundingRect(text)
    
    # Position bottom-left
    text_x = 10
    text_y = widget_height - 10
    
    # Draw background
    bg_rect_x = text_x - 4
    bg_rect_y = text_y - text_rect.height() - 2
    bg_rect_w = text_rect.width() + 8
    bg_rect_h = text_rect.height() + 4
    
    painter.fillRect(
        bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h,
        QColor(0, 0, 0, 180)
    )
    
    # Draw text in cyan
    painter.setPen(QColor(0, 255, 255))  # Cyan
    painter.drawText(text_x, text_y, text)

