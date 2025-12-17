"""PlastiTrace GUI with Phase 2: Enhanced UX with trust layer integration."""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar,
                             QScrollArea, QTextEdit, QGroupBox, QCheckBox, QDialog,
                             QListWidget, QListWidgetItem, QDialogButtonBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from ml.classifier import PlastiTraceClassifier
from ml.config import RECOMMENDATION
from ml.action_guidance import get_action_guidance
from vision.bbox_detector import detect_bbox, clamp_bbox_xyxy
from vision.bbox_tracker import BBoxTracker
from vision.smoothing import EMASmoother
from trust.decision_engine import DecisionEngine, DecisionState
from ui.ui_state import UIStateMachine, UIMode
from ui.overlay_renderer import OverlayRenderer
from ui.frame_buffer import RecentFrameBuffer
from ui.history import HistoryManager
from feedback.feedback_controller import FeedbackController
from feedback.dataset_store import DatasetStore
from feedback.priority_scorer import PriorityScorer
from feedback.schema import ConditionsData
from location.rules_engine import RulesEngine
from location.dropoff_store import DropOffStore
from location.location_filter import LocationFilterRanker
from location.region_manager import RegionManager
from location.event_logger import MapEventLogger
from ml.config import CLASSES
import time
import threading

# --- Theme Configuration ---
COLORS = {
    "bg_dark": "#0f172a",
    "card_bg": "#1e293b",
    "accent": "#10b981",  # Emerald 500
    "accent_hover": "#059669",
    "text_main": "#f8fafc",
    "text_dim": "#94a3b8",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "success": "#10b981"
}


class VideoThread(QThread):
    """Video processing thread with trust layer integration."""
    change_pixmap = pyqtSignal(np.ndarray, object, object)  # frame, decision_result, frame_quality
    
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.running = True
        self.tracker = BBoxTracker()
        self.smoother = EMASmoother(alpha=0.7)
        self.decision_engine = DecisionEngine()
        self.frame_count = 0
        self.last_result = None
        self.last_decision = None
        self.inference_interval = 3
        self.redetect_interval = 30
        self._infer_running = False
        self._infer_lock = threading.Lock()
        self._last_frame = None
        
    def _infer_async(self, roi_bgr, full_frame):
        """Async inference with trust layer."""
        with self._infer_lock:
            if self._infer_running:
                return
            self._infer_running = True
        
        def job():
            try:
                result = self.classifier.predict_from_bgr(roi_bgr)
                if result is not None:
                    # Calculate ROI area ratio
                    roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]
                    frame_area = full_frame.shape[0] * full_frame.shape[1]
                    roi_area_ratio = roi_area / frame_area if frame_area > 0 else 1.0
                    
                    # Process through decision engine
                    decision = self.decision_engine.process(full_frame, result, roi_area_ratio)
                    
                    # Update result with trust layer output
                    if decision.state == DecisionState.LOCKED and decision.locked_label:
                        result["label"] = decision.locked_label
                        result["confidence"] = decision.locked_confidence
                    else:
                        result["label"] = decision.current_label or result.get("label", "UNKNOWN")
                        result["confidence"] = decision.ema_conf
                    
                    self.last_result = result
                    self.last_decision = decision
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                with self._infer_lock:
                    self._infer_running = False
        
        threading.Thread(target=job, daemon=True).start()
        
    def run(self):
        cap = cv2.VideoCapture(0)
        bbox = None
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            H, W = frame.shape[:2]
            self.frame_count += 1
            self._last_frame = frame.copy()
            
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
                        try:
                            self.tracker.init(frame, bbox)
                        except:
                            pass
            
            if bbox is None:
                x1, y1, x2, y2 = int(W*0.25), int(H*0.25), int(W*0.75), int(H*0.75)
                bbox = (x1, y1, x2, y2)
            
            bbox = self.smoother.update_bbox(bbox)
            
            # Inference
            if self.frame_count % self.inference_interval == 0:
                x1, y1, x2, y2 = bbox
                roi = frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    self._infer_async(roi, frame)
            
            # Emit frame with decision result
            decision = self.last_decision
            frame_quality = decision.frame_quality if decision else None
            
            self.change_pixmap.emit(frame, decision, frame_quality)
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class PlastiTraceGUI(QMainWindow):
    """Main GUI window with Phase 2 features."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlastiTrace AI Vision - Phase 2")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        
        self.classifier = PlastiTraceClassifier("models/plastitrace.pth")
        self.ui_state = UIStateMachine()
        self.overlay_renderer = OverlayRenderer()
        self.frame_buffer = RecentFrameBuffer(max_size=10)
        self.history_manager = HistoryManager()
        
        # Phase 3: Feedback system
        self.dataset_store = DatasetStore()
        self.priority_scorer = PriorityScorer()
        self.feedback_controller = FeedbackController(
            self.dataset_store,
            self.priority_scorer,
            app_version="1.0.0",
            model_version="1.0.0"
        )
        
        # Phase 4: Location services
        self.rules_engine = RulesEngine()
        self.dropoff_store = DropOffStore()
        self.location_filter = LocationFilterRanker()
        self.region_manager = RegionManager()
        self.map_event_logger = MapEventLogger()
        self.current_scan_record_id = None  # Track current scan for event logging
        
        self.last_decision = None
        self.last_frame_quality = None
        self.current_frame = None
        self.current_roi = None
        self.top3_expanded = False
        self.ema_probs = None
        self.captured_frame = None
        self.captured_roi = None
        
        self.setup_ui()
        
        self.thread = VideoThread(self.classifier)
        self.thread.change_pixmap.connect(self.update_frame)
        self.thread.start()
        
        self.fps_start = time.time()
        self.fps_count = 0
        
        # UI update throttling (15 FPS)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_elements)
        self.ui_timer.start(66)  # ~15 FPS
        
        # Initial UI update
        self.update_ui_elements()

    def setup_ui(self):
        """Setup the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- LEFT COLUMN: Video Stream ---
        left_panel = QVBoxLayout()
        
        # Video Container
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
        
        self.status_label = QLabel("● SCANNING")
        self.status_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        stats_layout.addStretch()
        stats_layout.addWidget(self.status_label)
        
        left_panel.addLayout(stats_layout)
        main_layout.addLayout(left_panel, 7)

        # --- RIGHT COLUMN: Analysis Panel ---
        self.right_panel = QFrame()
        self.right_panel.setFixedWidth(400)
        self.right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['card_bg']};
                border-radius: 20px;
            }}
        """)
        self.analysis_layout = QVBoxLayout(self.right_panel)
        self.analysis_layout.setContentsMargins(25, 30, 25, 30)
        self.analysis_layout.setSpacing(15)

        # Header
        brand_title = QLabel("PLASTITRACE")
        brand_title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 24px; font-weight: 900; letter-spacing: 2px;")
        self.analysis_layout.addWidget(brand_title)

        tagline = QLabel("AI Real-time Plastic Analysis")
        tagline.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px; margin-bottom: 20px;")
        self.analysis_layout.addWidget(tagline)

        # Detection Result Card (Scan Mode)
        self.scan_result_card = QFrame()
        self.scan_result_card.setStyleSheet("background-color: rgba(15, 23, 42, 0.5); border-radius: 15px; padding: 15px;")
        scan_inner = QVBoxLayout(self.scan_result_card)
        
        self.result_label = QLabel("Scanning...")
        self.result_label.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 32px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        scan_inner.addWidget(self.result_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(8)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #334155; border-radius: 4px; }}
            QProgressBar::chunk {{ background-color: {COLORS['accent']}; border-radius: 4px; }}
        """)
        scan_inner.addWidget(self.conf_bar)

        self.conf_text = QLabel("Confidence: 0%")
        self.conf_text.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        self.conf_text.setAlignment(Qt.AlignRight)
        scan_inner.addWidget(self.conf_text)
        
        self.analysis_layout.addWidget(self.scan_result_card)
        
        # Top-3 Predictions Panel (Collapsible)
        self.top3_group = QGroupBox("Top Predictions")
        self.top3_group.setCheckable(True)
        self.top3_group.setChecked(False)
        self.top3_group.toggled.connect(self.toggle_top3)
        self.top3_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['text_main']};
                font-weight: bold;
                border: 1px solid {COLORS['card_bg']};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        top3_layout = QVBoxLayout()
        self.top3_list = QLabel("")
        self.top3_list.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        self.top3_list.setWordWrap(True)
        top3_layout.addWidget(self.top3_list)
        self.top3_group.setLayout(top3_layout)
        self.top3_group.hide()  # Hidden by default
        self.analysis_layout.addWidget(self.top3_group)
        
        # Capture Button (only shown when locked)
        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.setCursor(Qt.PointingHandCursor)
        self.capture_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                padding: 15px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        self.capture_btn.clicked.connect(self.capture_result)
        self.capture_btn.hide()
        self.analysis_layout.addWidget(self.capture_btn)
        
        # Review Mode Card (hidden initially)
        self.review_card = self.create_review_card()
        self.review_card.hide()
        self.analysis_layout.addWidget(self.review_card)

        # Recommendations
        rec_header = QLabel("♻️ RECYCLING GUIDE")
        rec_header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.analysis_layout.addWidget(rec_header)

        self.rec_label = QLabel("Please point the camera at a plastic object to begin classification.")
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet(f"color: {COLORS['text_dim']}; line-height: 150%; font-size: 13px;")
        self.analysis_layout.addWidget(self.rec_label)
        
        self.analysis_layout.addStretch()
        
        # Phase 3: History button
        history_btn = QPushButton("📋 History")
        history_btn.setCursor(Qt.PointingHandCursor)
        history_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_main']};
                border: 2px solid {COLORS['text_dim']};
                padding: 12px;
                border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['text_dim']};
                color: white;
            }}
        """)
        history_btn.clicked.connect(self.show_history)
        self.analysis_layout.addWidget(history_btn)
        
        # Phase 3: Export Dataset button (dev menu)
        export_btn = QPushButton("📤 Export Dataset")
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_main']};
                border: 2px solid {COLORS['text_dim']};
                padding: 12px;
                border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['text_dim']};
                color: white;
            }}
        """)
        export_btn.clicked.connect(self.show_export_dialog)
        self.analysis_layout.addWidget(export_btn)

        # Action Buttons
        quit_btn = QPushButton("Exit")
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
        self.analysis_layout.addWidget(quit_btn)

        main_layout.addWidget(self.right_panel)

    def create_review_card(self):
        """Create review mode result card."""
        card = QFrame()
        card.setStyleSheet("background-color: rgba(15, 23, 42, 0.7); border-radius: 15px; padding: 20px;")
        layout = QVBoxLayout(card)
        
        self.review_title = QLabel("")
        self.review_title.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 28px; font-weight: bold;")
        self.review_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.review_title)
        
        self.review_confidence_band = QLabel("")
        self.review_confidence_band.setStyleSheet(f"color: {COLORS['accent']}; font-size: 14px; font-weight: bold;")
        self.review_confidence_band.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.review_confidence_band)
        
        self.review_stability = QLabel("")
        self.review_stability.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        self.review_stability.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.review_stability)
        
        self.review_caution = QLabel("")
        self.review_caution.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
        self.review_caution.setWordWrap(True)
        self.review_caution.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.review_caution)
        
        # Phase 4: Action section (recycling recommendation)
        action_header = QLabel("Action:")
        action_header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 12px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(action_header)
        
        self.review_recommendation = QLabel("")
        self.review_recommendation.setStyleSheet(f"color: {COLORS['accent']}; font-size: 11px; font-weight: bold;")
        self.review_recommendation.setWordWrap(True)
        layout.addWidget(self.review_recommendation)
        
        self.review_instructions = QLabel("")
        self.review_instructions.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; line-height: 150%;")
        self.review_instructions.setWordWrap(True)
        layout.addWidget(self.review_instructions)
        
        self.review_warnings = QLabel("")
        self.review_warnings.setStyleSheet(f"color: {COLORS['warning']}; font-size: 10px; line-height: 150%;")
        self.review_warnings.setWordWrap(True)
        layout.addWidget(self.review_warnings)
        
        # Find Drop-off button
        self.find_dropoff_btn = QPushButton("📍 Find Drop-off Locations")
        self.find_dropoff_btn.setCursor(Qt.PointingHandCursor)
        self.find_dropoff_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        self.find_dropoff_btn.clicked.connect(self.show_dropoff_map)
        layout.addWidget(self.find_dropoff_btn)
        
        # Action guidance (legacy, keep for compatibility)
        guidance_header = QLabel("What to do next:")
        guidance_header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 12px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(guidance_header)
        
        self.review_guidance = QLabel("")
        self.review_guidance.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; line-height: 150%;")
        self.review_guidance.setWordWrap(True)
        layout.addWidget(self.review_guidance)
        
        # Phase 3: Condition toggles
        conditions_header = QLabel("Conditions (optional):")
        conditions_header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 11px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(conditions_header)
        
        conditions_layout = QVBoxLayout()
        self.condition_clean = QCheckBox("Clean")
        self.condition_label_present = QCheckBox("Label present")
        self.condition_crushed = QCheckBox("Crushed")
        self.condition_mixed = QCheckBox("Mixed material")
        
        for checkbox in [self.condition_clean, self.condition_label_present, 
                         self.condition_crushed, self.condition_mixed]:
            checkbox.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
            conditions_layout.addWidget(checkbox)
        
        layout.addLayout(conditions_layout)
        
        # Phase 3: Feedback buttons
        feedback_header = QLabel("Feedback:")
        feedback_header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 12px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(feedback_header)
        
        feedback_layout = QHBoxLayout()
        
        self.confirm_btn = QPushButton("✅ Correct")
        self.confirm_btn.setCursor(Qt.PointingHandCursor)
        self.confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
        """)
        self.confirm_btn.clicked.connect(self.confirm_feedback)
        feedback_layout.addWidget(self.confirm_btn)
        
        self.change_btn = QPushButton("✏️ Change")
        self.change_btn.setCursor(Qt.PointingHandCursor)
        self.change_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
        """)
        self.change_btn.clicked.connect(self.change_feedback)
        feedback_layout.addWidget(self.change_btn)
        
        self.unsure_btn = QPushButton("❌ Unsure")
        self.unsure_btn.setCursor(Qt.PointingHandCursor)
        self.unsure_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
        """)
        self.unsure_btn.clicked.connect(self.unsure_feedback)
        feedback_layout.addWidget(self.unsure_btn)
        
        layout.addLayout(feedback_layout)
        
        # Original buttons (Retake/Save)
        btn_layout = QHBoxLayout()
        
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.setCursor(Qt.PointingHandCursor)
        self.retake_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_main']};
                border: 2px solid {COLORS['text_dim']};
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['text_dim']};
                color: white;
            }}
        """)
        self.retake_btn.clicked.connect(self.retake_scan)
        btn_layout.addWidget(self.retake_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        self.save_btn.clicked.connect(self.save_result)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)
        
        return card
    
    def toggle_top3(self, checked):
        """Toggle top-3 predictions panel."""
        self.top3_expanded = checked
    
    def update_frame(self, frame, decision_result, frame_quality):
        """Update frame from video thread."""
        self.current_frame = frame
        self.last_decision = decision_result
        self.last_frame_quality = frame_quality
        
        # Update FPS
        self.fps_count += 1
        now = time.time()
        if now - self.fps_start >= 1.0:
            self.fps_label.setText(f"FPS: {self.fps_count / (now - self.fps_start):.1f}")
            self.fps_count = 0
            self.fps_start = now

        # Add to frame buffer if locked
        if decision_result and decision_result.state == DecisionState.LOCKED and frame_quality:
            self.frame_buffer.add(
                frame,
                frame_quality,
                decision_result.state,
                decision_result.locked_confidence,
                decision_result.stability
            )
    
    def update_ui_elements(self):
        """Update UI elements (throttled)."""
        if not self.current_frame is None:
            self.render_frame()
        
        if self.ui_state.is_scan_mode():
            self.update_scan_mode_ui()
        elif self.ui_state.is_review_mode():
            self.update_review_mode_ui()
    
    def render_frame(self):
        """Render frame with overlays."""
        if self.current_frame is None:
            return
        
        # Use default state if no decision yet
        if self.last_decision is None:
            from trust.decision_engine import DecisionState
            state = DecisionState.SCANNING
            stability = 0.0
            current_label = None
        else:
            state = self.last_decision.state
            stability = self.last_decision.stability
            current_label = self.last_decision.current_label
        
        # Apply overlays
        overlay_frame = self.overlay_renderer.render(
            self.current_frame.copy(),
            state,
            stability,
            self.last_frame_quality,
            current_label
        )

        # Convert to QImage and display
        rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qt_img.scaled(self.video_label.width(), self.video_label.height(), 
                              Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def update_scan_mode_ui(self):
        """Update UI for scan mode."""
        if self.last_decision is None:
            return
        
        # Update result display
        label = self.last_decision.current_label or "UNKNOWN"
        conf = self.last_decision.ema_conf
            
            self.result_label.setText(label.upper())
            self.conf_bar.setValue(int(conf * 100))
        self.conf_text.setText(f"Confidence: {conf*100:.1f}%")
            
        # Update status
        state = self.last_decision.state
        if state == DecisionState.LOCKED:
            self.status_label.setText("● LOCKED")
            self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold;")
        elif state == DecisionState.UNSTABLE:
            self.status_label.setText("● UNSTABLE")
            self.status_label.setStyleSheet(f"color: {COLORS['warning']}; font-weight: bold;")
        elif state == DecisionState.UNKNOWN:
            self.status_label.setText("● UNKNOWN")
            self.status_label.setStyleSheet(f"color: {COLORS['danger']}; font-weight: bold;")
        else:
            self.status_label.setText("● SCANNING")
            self.status_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        
        # Show/hide capture button
        if state == DecisionState.LOCKED:
            self.capture_btn.show()
        else:
            self.capture_btn.hide()
        
        # Update top-3 predictions
        if hasattr(self.thread, 'decision_engine') and hasattr(self.thread.decision_engine, 'aggregator'):
            agg = self.thread.decision_engine.aggregator
            if agg.ema_probs is not None:
                self.update_top3_predictions(agg.ema_probs)
        
        # Update recommendations
        rec = RECOMMENDATION.get(label, "No recommendation available.")
            self.rec_label.setText(rec)

    def update_top3_predictions(self, probs):
        """Update top-3 predictions display."""
        if probs is None:
            return
        
        from ml.config import CLASSES
        
        # Get top 3
        top3_indices = np.argsort(probs)[::-1][:3]
        lines = []
        for i, idx in enumerate(top3_indices):
            label = CLASSES[idx] if idx < len(CLASSES) else "UNKNOWN"
            prob = probs[idx]
            lines.append(f"{i+1}. {label}: {prob*100:.1f}%")
        
        self.top3_list.setText("\n".join(lines))
        if self.top3_expanded:
            self.top3_group.show()
    
    def update_review_mode_ui(self):
        """Update UI for review mode."""
        captured = self.ui_state.captured_result
        if captured is None:
            return
        
        self.review_title.setText(captured.plastic_type.upper())
        
        # Confidence band
        band_color = COLORS['success'] if captured.confidence_band == "High" else \
                    COLORS['warning'] if captured.confidence_band == "Medium" else COLORS['danger']
        self.review_confidence_band.setText(f"Confidence: {captured.confidence_band}")
        self.review_confidence_band.setStyleSheet(f"color: {band_color}; font-size: 14px; font-weight: bold;")
        
        self.review_stability.setText(f"Stability: {captured.stability:.2f} ({captured.stability_text})")
        
        # Caution if needed
        if captured.confidence_band != "High":
            margin_text = f"Margin: {captured.margin:.3f}"
            if captured.reason:
                margin_text += f" ({captured.reason})"
            self.review_caution.setText(margin_text)
            self.review_caution.show()
        else:
            self.review_caution.hide()
        
        # Action guidance
        guidance = get_action_guidance(captured.plastic_type)
        guidance_text = "• " + "\n• ".join(guidance)
        self.review_guidance.setText(guidance_text)
    
    def capture_result(self):
        """Capture current result and enter review mode."""
        if self.last_decision is None or self.last_frame_quality is None:
            return
        
        # Get best frame from buffer
        best_frame_data = self.frame_buffer.get_best_frame()
        if best_frame_data:
            frame = best_frame_data.frame_image
            frame_quality = best_frame_data.frame_quality
        else:
            frame = self.current_frame
            frame_quality = self.last_frame_quality
        
        # Store captured frame and ROI for feedback
        self.captured_frame = frame.copy() if frame is not None else None
        # Extract ROI if bbox exists (simplified - use full frame for now)
        self.captured_roi = None  # Could extract from bbox if needed
        
        # Capture
        captured = self.ui_state.capture(frame, self.last_decision, frame_quality)
        
        # Phase 4: Store scan record ID for event logging (will be set when feedback is provided)
        self.current_scan_record_id = None
        
        # Switch UI
        self.scan_result_card.hide()
        self.capture_btn.hide()
        self.top3_group.hide()
        self.review_card.show()
        
        # Freeze video (show captured frame)
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled = qt_img.scaled(self.video_label.width(), self.video_label.height(),
                                  Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.video_label.setPixmap(QPixmap.fromImage(scaled))
        
        self.update_review_mode_ui()
    
    def retake_scan(self):
        """Return to scan mode."""
        self.ui_state.retake()
        self.review_card.hide()
        self.scan_result_card.show()
        self.frame_buffer.clear()
        self.captured_frame = None
        self.captured_roi = None
        # Reset condition checkboxes
        self.condition_clean.setChecked(False)
        self.condition_label_present.setChecked(False)
        self.condition_crushed.setChecked(False)
        self.condition_mixed.setChecked(False)
    
    def save_result(self):
        """Save result to history (legacy method, kept for compatibility)."""
        captured = self.ui_state.captured_result
        if captured is None:
            return
        
        self.history_manager.add_entry(
            captured.frame_image,
            captured.plastic_type,
            captured.confidence,
            captured.confidence_band,
            captured.stability,
            captured.stability_text,
            captured.margin,
            captured.reason
        )
        
        # Show confirmation (simple for now)
        self.save_btn.setText("✓ Saved")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }}
        """)
        
        # Reset after 2 seconds
        QTimer.singleShot(2000, lambda: self.save_btn.setText("Save"))
    
    def get_conditions(self) -> ConditionsData:
        """Get condition flags from UI."""
        return ConditionsData(
            clean=self.condition_clean.isChecked() if self.condition_clean.isChecked() else None,
            label_present=self.condition_label_present.isChecked() if self.condition_label_present.isChecked() else None,
            crushed=self.condition_crushed.isChecked() if self.condition_crushed.isChecked() else None,
            mixed=self.condition_mixed.isChecked() if self.condition_mixed.isChecked() else None
        )
    
    def confirm_feedback(self):
        """Confirm prediction as correct."""
        if self.captured_frame is None or self.last_decision is None or self.last_frame_quality is None:
            return
        
        from feedback.schema import FrameQualityData
        
        # Create frame quality data
        fq = FrameQualityData(
            blur_score=self.last_frame_quality.blur_score,
            brightness=self.last_frame_quality.brightness,
            is_blurry=self.last_frame_quality.is_blurry,
            is_too_dark=self.last_frame_quality.is_too_dark
        )
        
        # Get conditions
        conditions = self.get_conditions()
        
        # Confirm
        record = self.feedback_controller.confirm(
            self.captured_frame,
            self.captured_roi,
            self.last_decision.locked_label or self.last_decision.current_label or "UNKNOWN",
            self.last_decision.locked_confidence,
            self.last_decision.stability,
            self.last_decision.vote_ratio,
            self.last_decision.mean_margin,
            self.last_decision.mean_entropy,
            fq,
            conditions
        )
        
        # Phase 4: Store record ID for event logging
        self.current_scan_record_id = record.id
        
        # Show confirmation
        self.confirm_btn.setText("✓ Confirmed")
        QTimer.singleShot(2000, lambda: self.confirm_btn.setText("✅ Correct"))
    
    def change_feedback(self):
        """Change/correct the prediction."""
        if self.captured_frame is None or self.last_decision is None or self.last_frame_quality is None:
            return
        
        # Show label selector dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Correct Label")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        
        label = QLabel("What is the correct label?")
        label.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 14px;")
        layout.addWidget(label)
        
        combo = QComboBox()
        combo.addItems(["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER", "UNKNOWN"])
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_main']};
                padding: 8px;
                border-radius: 5px;
            }}
        """)
        layout.addWidget(combo)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            user_label = combo.currentText()
            
            from feedback.schema import FrameQualityData
            
            # Create frame quality data
            fq = FrameQualityData(
                blur_score=self.last_frame_quality.blur_score,
                brightness=self.last_frame_quality.brightness,
                is_blurry=self.last_frame_quality.is_blurry,
                is_too_dark=self.last_frame_quality.is_too_dark
            )
            
            # Get conditions
            conditions = self.get_conditions()
            
            # Correct
            record = self.feedback_controller.correct(
                self.captured_frame,
                self.captured_roi,
                self.last_decision.locked_label or self.last_decision.current_label or "UNKNOWN",
                self.last_decision.locked_confidence,
                self.last_decision.stability,
                self.last_decision.vote_ratio,
                self.last_decision.mean_margin,
                self.last_decision.mean_entropy,
                fq,
                user_label,
                conditions
            )
            
            # Show confirmation
            self.change_btn.setText(f"✓ Changed to {user_label}")
            QTimer.singleShot(2000, lambda: self.change_btn.setText("✏️ Change"))
    
    def unsure_feedback(self):
        """Mark as unsure/unknown."""
        if self.captured_frame is None or self.last_decision is None or self.last_frame_quality is None:
            return
        
        from feedback.schema import FrameQualityData
        
        # Create frame quality data
        fq = FrameQualityData(
            blur_score=self.last_frame_quality.blur_score,
            brightness=self.last_frame_quality.brightness,
            is_blurry=self.last_frame_quality.is_blurry,
            is_too_dark=self.last_frame_quality.is_too_dark
        )
        
        # Get conditions
        conditions = self.get_conditions()
        
        # Mark as unsure
        record = self.feedback_controller.unsure(
            self.captured_frame,
            self.captured_roi,
            self.last_decision.locked_label or self.last_decision.current_label or "UNKNOWN",
            self.last_decision.locked_confidence,
            self.last_decision.stability,
            self.last_decision.vote_ratio,
            self.last_decision.mean_margin,
            self.last_decision.mean_entropy,
            fq,
            conditions
        )
        
        # Phase 4: Store record ID for event logging
        self.current_scan_record_id = record.id
        
        # Show confirmation
        self.unsure_btn.setText("✓ Marked Unsure")
        QTimer.singleShot(2000, lambda: self.unsure_btn.setText("❌ Unsure"))
    
    def show_history(self):
        """Show history screen."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Scan History")
        dialog.setMinimumSize(600, 500)
        dialog.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QVBoxLayout(dialog)
        
        # Header
        header = QLabel("Recent Scans")
        header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # List widget
        list_widget = QListWidget()
        list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_main']};
                border-radius: 10px;
            }}
        """)
        
        # Get recent records
        records = self.dataset_store.get_all_records(limit=50)
        for record in records:
            # Determine display label
            label = record.user_label if record.user_label else record.pred_label
            status = "✓ Confirmed" if record.is_confirmed else (
                "✏️ Corrected" if record.user_label and record.user_label != record.pred_label else
                "❌ Unsure" if record.user_label == "UNKNOWN" else "—"
            )
            
            item_text = f"{label} ({record.pred_confidence*100:.1f}%) - {status}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, record.id)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # View button
        view_btn = QPushButton("View Details")
        view_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }}
        """)
        
        def view_details():
            current = list_widget.currentItem()
            if current:
                record_id = current.data(Qt.UserRole)
                record = self.dataset_store.get_record(record_id)
                if record:
                    self.show_record_details(record)
        
        view_btn.clicked.connect(view_details)
        layout.addWidget(view_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_main']};
                border: 2px solid {COLORS['text_dim']};
                padding: 10px;
                border-radius: 8px;
            }}
        """)
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def show_record_details(self, record):
        """Show detailed record information."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Record Details")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel(f"Scan Record: {record.id[:8]}...")
        title.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Details text
        details = QTextEdit()
        details.setReadOnly(True)
        details.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_main']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        
        details_text = f"""
Predicted: {record.pred_label} ({record.pred_confidence*100:.1f}%)
User Label: {record.user_label or 'N/A'}
Confirmed: {'Yes' if record.is_confirmed else 'No'}
Stability: {record.stability:.2f}
Priority Score: {record.priority_score:.2f}
High Value: {'Yes' if record.high_value else 'No'}

Conditions:
  Clean: {record.conditions.clean or 'N/A'}
  Label Present: {record.conditions.label_present or 'N/A'}
  Crushed: {record.conditions.crushed or 'N/A'}
  Mixed: {record.conditions.mixed or 'N/A'}

Timestamp: {record.timestamp}
        """
        details.setPlainText(details_text)
        layout.addWidget(details)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def show_export_dialog(self):
        """Show export dataset dialog."""
        from feedback.dataset_exporter import DatasetExporter
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Dataset")
        dialog.setMinimumSize(400, 300)
        dialog.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QVBoxLayout(dialog)
        
        # Header
        header = QLabel("Export Dataset")
        header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Mode selection
        mode_label = QLabel("Export Mode:")
        mode_label.setStyleSheet(f"color: {COLORS['text_main']};")
        layout.addWidget(mode_label)
        
        mode_combo = QComboBox()
        mode_combo.addItems(["All samples", "High-value only", "Corrected only"])
        mode_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_main']};
                padding: 8px;
                border-radius: 5px;
            }}
        """)
        layout.addWidget(mode_combo)
        
        # Output directory
        dir_label = QLabel("Output Directory:")
        dir_label.setStyleSheet(f"color: {COLORS['text_main']};")
        layout.addWidget(dir_label)
        
        dir_input = QTextEdit()
        dir_input.setMaximumHeight(30)
        dir_input.setPlainText("exports/dataset_export")
        dir_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_main']};
                border-radius: 5px;
            }}
        """)
        layout.addWidget(dir_input)
        
        # Export button
        export_btn = QPushButton("Export")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
            }}
        """)
        
        def do_export():
            mode_map = {
                "All samples": "all",
                "High-value only": "high_value",
                "Corrected only": "corrected"
            }
            mode = mode_map.get(mode_combo.currentText(), "all")
            output_dir = dir_input.toPlainText().strip()
            
            exporter = DatasetExporter(self.dataset_store)
            try:
                export_path = exporter.export(output_dir, mode=mode)
                # Show success
                success_label = QLabel(f"✓ Exported to: {export_path}")
                success_label.setStyleSheet(f"color: {COLORS['success']};")
                layout.addWidget(success_label)
            except Exception as e:
                error_label = QLabel(f"✗ Error: {str(e)}")
                error_label.setStyleSheet(f"color: {COLORS['danger']};")
                layout.addWidget(error_label)
        
        export_btn.clicked.connect(do_export)
        layout.addWidget(export_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def show_dropoff_map(self):
        """Show drop-off locations map/list."""
        if not hasattr(self, 'current_recommendation') or not hasattr(self, 'current_plastic_type'):
            return
        
        # Log event
        if self.current_scan_record_id:
            self.map_event_logger.log_event(
                self.current_scan_record_id,
                "opened_map"
            )
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Drop-off Locations")
        dialog.setMinimumSize(600, 500)
        dialog.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QVBoxLayout(dialog)
        
        # Header
        header = QLabel("📍 Nearby Drop-off Locations")
        header.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Warning if uncertain
        if self.current_plastic_type == "UNKNOWN" or (hasattr(self, 'last_decision') and self.last_decision and self.last_decision.stability < 0.5):
            warning = QLabel("⚠ Result uncertain—verify resin code manually")
            warning.setStyleSheet(f"color: {COLORS['warning']}; font-size: 12px; font-weight: bold;")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        
        # Get all locations
        all_locations = self.dropoff_store.get_all_locations()
        
        # Get current region
        region = self.region_manager.get_current_region()
        
        # Get conditions
        conditions = self.get_conditions()
        
        # Filter and rank
        ranked = self.location_filter.filter_and_rank(
            all_locations,
            self.current_plastic_type,
            self.current_recommendation,
            conditions,
            user_lat=None,  # No GPS for now
            user_lng=None
        )
        
        # Filter out excluded locations
        ranked = [r for r in ranked if not r.excluded]
        
        if not ranked:
            # No locations found
            no_locations = QLabel("No drop-off locations found for this type.\n\nYou can:\n• Add a location manually\n• Search externally")
            no_locations.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
            no_locations.setWordWrap(True)
            layout.addWidget(no_locations)
            
            # Search externally button
            search_btn = QPushButton("🔍 Search Externally")
            search_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['accent']};
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                }}
            """)
            
            def open_search():
                import webbrowser
                query = f"recycling center {self.current_plastic_type} {region.get('city', '')}"
                webbrowser.open(f"https://www.google.com/search?q={query}")
            
            search_btn.clicked.connect(open_search)
            layout.addWidget(search_btn)
        else:
            # List widget
            list_widget = QListWidget()
            list_widget.setStyleSheet(f"""
                QListWidget {{
                    background-color: {COLORS['card_bg']};
                    color: {COLORS['text_main']};
                    border-radius: 10px;
                }}
                QListWidgetItem {{
                    padding: 10px;
                    border-bottom: 1px solid {COLORS['bg_dark']};
                }}
            """)
            
            for ranked_loc in ranked[:10]:  # Show top 10
                loc = ranked_loc.location
                item_text = f"{loc.name}\n{loc.address}"
                if ranked_loc.distance is not None:
                    item_text += f"\n{ranked_loc.distance:.1f} km away"
                item_text += f"\n{ranked_loc.reason}"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, loc.id)
                list_widget.addItem(item)
            
            layout.addWidget(list_widget)
            
            # Open in Maps button
            open_maps_btn = QPushButton("🗺️ Open in Maps")
            open_maps_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['accent']};
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                }}
            """)
            
            def open_in_maps():
                current = list_widget.currentItem()
                if current:
                    loc_id = current.data(Qt.UserRole)
                    location = self.dropoff_store.get_location(loc_id)
                    if location:
                        # Open in system maps
                        import webbrowser
                        # Use Google Maps URL
                        url = f"https://www.google.com/maps/search/?api=1&query={location.lat},{location.lng}"
                        webbrowser.open(url)
                        
                        # Log event
                        if self.current_scan_record_id:
                            self.map_event_logger.log_event(
                                self.current_scan_record_id,
                                "opened_external_maps",
                                location_id=loc_id
                            )
            
            open_maps_btn.clicked.connect(open_in_maps)
            layout.addWidget(open_maps_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_main']};
                border: 2px solid {COLORS['text_dim']};
                padding: 10px;
                border-radius: 8px;
            }}
        """)
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def closeEvent(self, event):
        """Handle window close."""
        self.thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = PlastiTraceGUI()
    window.show()
    sys.exit(app.exec_())
