import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from ml.classifier import PlastiTraceClassifier
from ml.config import RECOMMENDATION
from vision.bbox_detector import detect_bbox, clamp_bbox_xyxy
from vision.bbox_tracker import BBoxTracker
from vision.smoothing import EMASmoother
import time


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
            if not ret:
                break
                
            H, W = frame.shape[:2]
            self.frame_count += 1
            
            # Tracking/Detection logic (sama kayak camera_loop.py)
            should_redetect = (not self.tracker.is_active()) or (self.frame_count % self.redetect_interval == 0)
            
            if not should_redetect and self.tracker.is_active():
                tracked = self.tracker.update(frame)
                if tracked is not None:
                    bbox = tracked
                else:
                    bbox = None
            
            if should_redetect:
                detected = detect_bbox(frame)
                if detected is not None:
                    clamped = clamp_bbox_xyxy(detected, W, H)
                    if clamped is not None:
                        bbox = clamped
                        try:
                            self.tracker.init(frame, bbox)
                        except RuntimeError:
                            pass
            
            # Fallback bbox
            if bbox is None:
                x1 = int(W * 0.2)
                y1 = int(H * 0.2)
                x2 = int(W * 0.8)
                y2 = int(H * 0.8)
                bbox = (x1, y1, x2, y2)
            
            # Smooth bbox
            if bbox is not None:
                bbox = self.smoother.update_bbox(bbox)
            
            # ROI extraction
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    roi = frame
            else:
                roi = frame
            
            # Inference
            if self.frame_count % self.inference_interval == 0:
                try:
                    result = self.classifier.predict_from_bgr(roi)
                    if result is not None:
                        conf = result.get("confidence", 0.0)
                        result["confidence"] = self.smoother.update_confidence(conf)
                        self.last_result = result
                except Exception as e:
                    print(f"Inference error: {e}")
            
            # Emit frame
            self.change_pixmap.emit(frame, bbox, self.last_result if self.last_result else {})
            
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


class PlastiTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlastiTrace Desktop - Realtime Detection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Load classifier
        self.classifier = PlastiTraceClassifier("models/plastitrace.pth")
        
        # Setup UI
        self.setup_ui()
        
        # Start video thread
        self.thread = VideoThread(self.classifier)
        self.thread.change_pixmap.connect(self.update_frame)
        self.thread.start()
        
        # FPS counter
        self.fps = 0
        self.fps_count = 0
        self.fps_start = time.time()
        
    def setup_ui(self):
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout
        layout = QHBoxLayout()
        main_widget.setLayout(layout)
        
        # Left side - Video
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #10b981;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        # FPS label
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.fps_label.setStyleSheet("color: #10b981; padding: 10px;")
        left_layout.addWidget(self.fps_label)
        
        layout.addWidget(left_widget, 2)
        
        # Right side - Info panel
        right_widget = QFrame()
        right_widget.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #14b8a6);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Title
        title = QLabel("PlastiTrace")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)
        
        subtitle = QLabel("AI Plastic Classifier")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setStyleSheet("color: #d1fae5;")
        subtitle.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(subtitle)
        
        right_layout.addSpacing(30)
        
        # Result label
        self.result_label = QLabel("Menunggu deteksi...")
        self.result_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.result_label.setStyleSheet("color: white; background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        right_layout.addWidget(self.result_label)
        
        # Confidence
        self.conf_label = QLabel("Confidence: -")
        self.conf_label.setFont(QFont("Arial", 16))
        self.conf_label.setStyleSheet("color: white; padding: 10px;")
        self.conf_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.conf_label)
        
        right_layout.addSpacing(20)
        
        # Recommendation
        rec_title = QLabel("♻️ Rekomendasi Daur Ulang")
        rec_title.setFont(QFont("Arial", 14, QFont.Bold))
        rec_title.setStyleSheet("color: white;")
        right_layout.addWidget(rec_title)
        
        self.rec_label = QLabel("Arahkan kamera ke plastik untuk memulai...")
        self.rec_label.setFont(QFont("Arial", 11))
        self.rec_label.setStyleSheet("color: white; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;")
        self.rec_label.setWordWrap(True)
        right_layout.addWidget(self.rec_label)
        
        right_layout.addStretch()
        
        # Quit button
        quit_btn = QPushButton("Keluar (ESC)")
        quit_btn.setFont(QFont("Arial", 12, QFont.Bold))
        quit_btn.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.9);
                color: white;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: rgba(220, 38, 38, 1);
            }
        """)
        quit_btn.clicked.connect(self.close)
        right_layout.addWidget(quit_btn)
        
        layout.addWidget(right_widget, 1)
    
    def update_frame(self, frame, bbox, result):
        # Update FPS
        self.fps_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.fps = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_start = time.time()
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
        
        # Draw bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 3)
        
        # Convert to Qt format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qt_image.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))
        
        # Update result
        if result and 'label' in result:
            label = result.get('label', '-')
            conf = result.get('confidence', 0.0)
            
            self.result_label.setText(label)
            self.conf_label.setText(f"Confidence: {conf*100:.1f}%")
            
            rec = RECOMMENDATION.get(label, "Tidak ada rekomendasi.")
            self.rec_label.setText(rec)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlastiTraceGUI()
    window.show()
    sys.exit(app.exec_())