import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from ml.classifier import PlastiTraceClassifier
from ml.config import RECOMMENDATION
from vision.bbox_detector import detect_bbox, clamp_bbox_xyxy
from vision.bbox_tracker import BBoxTracker
from vision.smoothing import EMASmoother
import time

# --- Konfigurasi Tema ---
COLORS = {
    "bg_dark": "#0f172a",
    "card_bg": "#1e293b",
    "accent": "#10b981", # Emerald 500
    "accent_hover": "#059669",
    "text_main": "#f8fafc",
    "text_dim": "#94a3b8",
    "danger": "#ef4444"
}

class VideoThread(QThread):
    change_pixmap = pyqtSignal(np.ndarray, tuple, dict)
    
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.running = True
        self.tracker = BBoxTracker()
        self.smoother = EMASmoother(alpha=0.7)
        self.frame_count = 0
        self.last_result = None
        self.inference_interval = 3
        self.redetect_interval = 30
        
    def run(self):
        cap = cv2.VideoCapture(0)
        bbox = None
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break
                
            H, W = frame.shape[:2]
            self.frame_count += 1
            
            should_redetect = (not self.tracker.is_active()) or (self.frame_count % self.redetect_interval == 0)
            
            if not should_redetect and self.tracker.is_active():
                tracked = self.tracker.update(frame)
                bbox = tracked if tracked is not None else None
            
            if should_redetect:
                detected = detect_bbox(frame)
                if detected is not None:
                    clamped = clamp_bbox_xyxy(detected, W, H)
                    if clamped is not None:
                        bbox = clamped
                        try: self.tracker.init(frame, bbox)
                        except: pass
            
            if bbox is None:
                x1, y1, x2, y2 = int(W*0.25), int(H*0.25), int(W*0.75), int(H*0.75)
                bbox = (x1, y1, x2, y2)
            
            bbox = self.smoother.update_bbox(bbox)
            
            if self.frame_count % self.inference_interval == 0:
                x1, y1, x2, y2 = bbox
                roi = frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    result = self.classifier.predict_from_bgr(roi)
                    if result:
                        result["confidence"] = self.smoother.update_confidence(result.get("confidence", 0.0))
                        self.last_result = result
            
            self.change_pixmap.emit(frame, bbox, self.last_result if self.last_result else {})
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class PlastiTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlastiTrace AI Vision")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        
        self.classifier = PlastiTraceClassifier("models/plastitrace.pth")
        self.setup_ui()
        
        self.thread = VideoThread(self.classifier)
        self.thread.change_pixmap.connect(self.update_frame)
        self.thread.start()
        
        self.fps_start = time.time()
        self.fps_count = 0

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- LEFT COLUMN: Video Stream ---
        left_panel = QVBoxLayout()
        
        # Video Container with Rounded Border
        self.video_container = QFrame()
        self.video_container.setStyleSheet(f"""
            QFrame {{
                background-color: #000000;
                border-radius: 15px;
                border: 2px solid {COLORS['card_bg']};
            }}
        """)
        video_inner_layout = QVBoxLayout(self.video_container)
        video_inner_layout.setContentsMargins(5, 5, 5, 5)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 12px;")
        video_inner_layout.addWidget(self.video_label)
        
        left_panel.addWidget(self.video_container, 5)
        
        # Stats Bar
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-weight: bold;")
        stats_layout.addWidget(self.fps_label)
        
        self.status_dot = QLabel("● SYSTEM ACTIVE")
        self.status_dot.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        stats_layout.addStretch()
        stats_layout.addWidget(self.status_dot)
        
        left_panel.addLayout(stats_layout)
        main_layout.addLayout(left_panel, 7)

        # --- RIGHT COLUMN: Analysis Panel ---
        right_panel = QFrame()
        right_panel.setFixedWidth(380)
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['card_bg']};
                border-radius: 20px;
            }}
        """)
        analysis_layout = QVBoxLayout(right_panel)
        analysis_layout.setContentsMargins(25, 30, 25, 30)
        analysis_layout.setSpacing(15)

        # Header
        brand_title = QLabel("PLASTITRACE")
        brand_title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 24px; font-weight: 900; letter-spacing: 2px;")
        analysis_layout.addWidget(brand_title)

        tagline = QLabel("AI Real-time Plastic Analysis")
        tagline.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px; margin-bottom: 20px;")
        analysis_layout.addWidget(tagline)

        # Detection Result Card
        result_card = QFrame()
        result_card.setStyleSheet("background-color: rgba(15, 23, 42, 0.5); border-radius: 15px; padding: 15px;")
        result_inner = QVBoxLayout(result_card)
        
        self.result_label = QLabel("Scanning...")
        self.result_label.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 32px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_inner.addWidget(self.result_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(8)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #334155; border-radius: 4px; }}
            QProgressBar::chunk {{ background-color: {COLORS['accent']}; border-radius: 4px; }}
        """)
        result_inner.addWidget(self.conf_bar)

        self.conf_text = QLabel("Confidence: 0%")
        self.conf_text.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        self.conf_text.setAlignment(Qt.AlignRight)
        result_inner.addWidget(self.conf_text)
        
        analysis_layout.addWidget(result_card)

        # Recommendations
        rec_header = QLabel("♻️ RECYCLING GUIDE")
        rec_header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 14px; font-weight: bold; margin-top: 10px;")
        analysis_layout.addWidget(rec_header)

        self.rec_label = QLabel("Please point the camera at a plastic object to begin classification.")
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet(f"color: {COLORS['text_dim']}; line-height: 150%; font-size: 13px; background: transparent; border: none;")
        analysis_layout.addWidget(self.rec_label)

        analysis_layout.addStretch()

        # Action Buttons
        quit_btn = QPushButton("Terminating Session")
        quit_btn.setCursor(Qt.PointingHandCursor)
        quit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['danger']};
                border: 2px solid {COLORS['danger']};
                padding: 12px;
                border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']};
                color: white;
            }}
        """)
        quit_btn.clicked.connect(self.close)
        analysis_layout.addWidget(quit_btn)

        main_layout.addWidget(right_panel)

    def update_frame(self, frame, bbox, result):
        # Update FPS Calculation
        self.fps_count += 1
        now = time.time()
        if now - self.fps_start >= 1.0:
            self.fps_label.setText(f"FPS: {self.fps_count / (now - self.fps_start):.1f}")
            self.fps_count = 0
            self.fps_start = now

        # Draw Clean Bounding Box
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
            # Add subtle corner accents
            l = 20
            cv2.line(frame, (x1, y1), (x1+l, y1), (16, 185, 129), 5)
            cv2.line(frame, (x1, y1), (x1, y1+l), (16, 185, 129), 5)

        # Render Frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qt_img.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatioByExpanding)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

        # Update Info Panel
        if result and 'label' in result:
            label = result.get('label', 'Unknown')
            conf = result.get('confidence', 0.0)
            
            self.result_label.setText(label.upper())
            self.conf_bar.setValue(int(conf * 100))
            self.conf_text.setText(f"Match Confidence: {conf*100:.1f}%")
            
            rec = RECOMMENDATION.get(label, "No recommendation available for this material.")
            self.rec_label.setText(rec)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Set global font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = PlastiTraceGUI()
    window.show()
    sys.exit(app.exec_())