"""
Main window for PlastiTrace desktop application with 3-panel layout.
Left: Detection results + filters
Center: Video realtime
Right: Map + location list
"""
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QSpinBox,
    QGroupBox, QFrame, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from ui.video_widget import VideoWidget
from workers.capture_worker import CaptureWorker
from workers.inference_worker import InferenceWorker
from ui.map_view import MapView
from ml.classifier import PlastiTraceClassifier
from location.dropoff_store import DropOffStore
from domain.models import Location
from domain.geo import filter_locations


class MainWindow(QMainWindow):
    """Main application window with 3-panel layout."""
    
    def __init__(self, classifier: PlastiTraceClassifier):
        super().__init__()
        self.classifier = classifier
        
        # Workers
        self.capture_worker = None
        self.inference_worker = None
        self.inference_thread = None
        
        # Location store
        self.location_store = DropOffStore()
        self.all_locations = self._load_locations()
        
        # State
        self.running = False
        
        self.setWindowTitle("PlastiTrace Desktop")
        self.setMinimumSize(1600, 900)
        
        self.setup_ui()
    
    def _load_locations(self) -> list[Location]:
        """Load locations from store."""
        dropoff_locs = self.location_store.get_all_locations()
        locations = []
        for loc in dropoff_locs:
            locations.append(Location(
                id=loc.id,
                name=loc.name,
                lat=loc.lat,
                lon=loc.lng,
                address=loc.address,
                hours=loc.hours,
                phone=loc.contact,
                types=loc.accepted_types,
                source=loc.source
            ))
        return locations
    
    def setup_ui(self):
        """Setup UI components with 3-panel layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # LEFT PANEL: Detection results + filters
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # CENTER PANEL: Video realtime
        center_panel = QFrame()
        center_panel.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border-radius: 10px;
            }
        """)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        self.video_widget = VideoWidget()
        center_layout.addWidget(self.video_widget)
        
        main_layout.addWidget(center_panel, 2)
        
        # RIGHT PANEL: Map + location list
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border-radius: 10px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.map_view = MapView()
        self.map_view.set_locations(self.all_locations)
        self.map_view.locationSelected.connect(self.on_location_selected)
        right_layout.addWidget(self.map_view)
        
        main_layout.addWidget(right_panel, 1)
    
    def create_left_panel(self):
        """Create left panel with detection results and controls."""
        panel = QFrame()
        panel.setFixedWidth(350)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("PlastiTrace")
        title.setStyleSheet("color: #10b981; font-size: 24px; font-weight: 900; letter-spacing: 2px;")
        layout.addWidget(title)
        
        tagline = QLabel("AI Real-time Plastic Analysis")
        tagline.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(tagline)
        
        # Detection Result Card
        result_card = QFrame()
        result_card.setStyleSheet("background-color: rgba(15, 23, 42, 0.5); border-radius: 15px; padding: 15px;")
        result_inner = QVBoxLayout(result_card)
        
        self.result_label = QLabel("Scanning...")
        self.result_label.setStyleSheet("color: #f8fafc; font-size: 32px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_inner.addWidget(self.result_label)
        
        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(8)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet("""
            QProgressBar { background-color: #334155; border-radius: 4px; }
            QProgressBar::chunk { background-color: #10b981; border-radius: 4px; }
        """)
        result_inner.addWidget(self.conf_bar)
        
        self.conf_text = QLabel("Confidence: 0%")
        self.conf_text.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.conf_text.setAlignment(Qt.AlignRight)
        result_inner.addWidget(self.conf_text)
        
        layout.addWidget(result_card)
        
        # Recommendations
        rec_header = QLabel("♻️ RECYCLING GUIDE")
        rec_header.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(rec_header)
        
        self.rec_label = QLabel("Please point the camera at a plastic object to begin classification.")
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet("color: #94a3b8; line-height: 150%; font-size: 13px;")
        layout.addWidget(self.rec_label)
        
        # Start/Stop button
        self.start_stop_btn = QPushButton("Start")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.start_stop_btn.clicked.connect(self.toggle_start_stop)
        layout.addWidget(self.start_stop_btn)
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_group.setStyleSheet("""
            QGroupBox {
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        camera_layout = QVBoxLayout(camera_group)
        
        camera_idx_layout = QHBoxLayout()
        camera_idx_layout.addWidget(QLabel("Camera Index:"))
        self.camera_idx_spin = QSpinBox()
        self.camera_idx_spin.setMinimum(0)
        self.camera_idx_spin.setMaximum(5)
        self.camera_idx_spin.setValue(0)
        camera_idx_layout.addWidget(self.camera_idx_spin)
        camera_layout.addLayout(camera_idx_layout)
        
        layout.addWidget(camera_group)
        
        # Inference settings
        inference_group = QGroupBox("Inference Settings")
        inference_group.setStyleSheet(camera_group.styleSheet())
        inference_layout = QVBoxLayout(inference_group)
        
        inf_interval_layout = QHBoxLayout()
        inf_interval_layout.addWidget(QLabel("Inference Interval:"))
        self.inference_interval_spin = QSpinBox()
        self.inference_interval_spin.setMinimum(1)
        self.inference_interval_spin.setMaximum(10)
        self.inference_interval_spin.setValue(3)
        inf_interval_layout.addWidget(self.inference_interval_spin)
        inference_layout.addLayout(inf_interval_layout)
        
        layout.addWidget(inference_group)
        
        # Stability settings
        stability_group = QGroupBox("Stability Settings")
        stability_group.setStyleSheet(camera_group.styleSheet())
        stability_layout = QVBoxLayout(stability_group)
        
        conf_thresh_layout = QVBoxLayout()
        conf_thresh_layout.addWidget(QLabel("Confidence Threshold:"))
        conf_thresh_slider_layout = QHBoxLayout()
        self.conf_thresh_slider = QSlider(Qt.Horizontal)
        self.conf_thresh_slider.setMinimum(40)
        self.conf_thresh_slider.setMaximum(90)
        self.conf_thresh_slider.setValue(65)
        self.conf_thresh_label = QLabel("0.65")
        conf_thresh_slider_layout.addWidget(self.conf_thresh_slider)
        conf_thresh_slider_layout.addWidget(self.conf_thresh_label)
        conf_thresh_layout.addLayout(conf_thresh_slider_layout)
        self.conf_thresh_slider.valueChanged.connect(
            lambda v: self.conf_thresh_label.setText(f"{v/100:.2f}")
        )
        stability_layout.addLayout(conf_thresh_layout)
        
        self.stabilize_check = QCheckBox("Enable Stabilization")
        self.stabilize_check.setChecked(True)
        stability_layout.addWidget(self.stabilize_check)
        
        alpha_layout = QVBoxLayout()
        alpha_layout.addWidget(QLabel("Smoothing Alpha:"))
        alpha_slider_layout = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(10)
        self.alpha_slider.setMaximum(90)
        self.alpha_slider.setValue(50)
        self.alpha_label = QLabel("0.50")
        alpha_slider_layout.addWidget(self.alpha_slider)
        alpha_slider_layout.addWidget(self.alpha_label)
        alpha_layout.addLayout(alpha_slider_layout)
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_label.setText(f"{v/100:.2f}")
        )
        stability_layout.addLayout(alpha_layout)
        
        layout.addWidget(stability_group)
        
        layout.addStretch()
        
        return panel
    
    def toggle_start_stop(self):
        """Toggle camera start/stop."""
        if not self.running:
            self.start()
        else:
            self.stop()
    
    def start(self):
        """Start camera and inference workers."""
        if self.running:
            return
        
        camera_index = self.camera_idx_spin.value()
        
        # Create capture worker
        self.capture_worker = CaptureWorker(camera_index)
        self.capture_worker.frameReady.connect(self.on_frame_received)
        self.capture_worker.fpsReady.connect(self.on_fps_received)
        self.capture_worker.start()
        
        # Create inference worker
        self.inference_worker = InferenceWorker(self.classifier)
        self.inference_worker.bboxReady.connect(self.on_bbox_received)
        self.inference_worker.resultReady.connect(self.on_inference_result)
        
        # Update inference worker settings
        self.inference_worker.set_inference_interval(self.inference_interval_spin.value())
        self.inference_worker.set_confidence_threshold(self.conf_thresh_slider.value() / 100.0)
        self.inference_worker.set_stabilize_enabled(self.stabilize_check.isChecked())
        self.inference_worker.set_alpha(self.alpha_slider.value() / 100.0)
        
        # Connect settings changes
        self.inference_interval_spin.valueChanged.connect(
            self.inference_worker.set_inference_interval
        )
        self.conf_thresh_slider.valueChanged.connect(
            lambda v: self.inference_worker.set_confidence_threshold(v / 100.0)
        )
        self.stabilize_check.toggled.connect(
            self.inference_worker.set_stabilize_enabled
        )
        self.alpha_slider.valueChanged.connect(
            lambda v: self.inference_worker.set_alpha(v / 100.0)
        )
        
        # Move inference worker to thread
        self.inference_thread = QThread()
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.start()
        
        self.running = True
        self.start_stop_btn.setText("Stop")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
    
    def stop(self):
        """Stop camera and inference workers."""
        if not self.running:
            return
        
        # Stop capture worker
        if self.capture_worker:
            self.capture_worker.stop()
            self.capture_worker = None
        
        # Stop inference thread
        if self.inference_thread:
            self.inference_thread.quit()
            self.inference_thread.wait()
            self.inference_thread = None
        
        if self.inference_worker:
            self.inference_worker = None
        
        self.running = False
        self.start_stop_btn.setText("Start")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        
        # Clear video widget
        self.video_widget.setFrame(None, 0.0)
        self.video_widget.setBBox(None, False)
        self.video_widget.setResult(None)
        self.result_label.setText("Scanning...")
        self.conf_text.setText("Confidence: 0%")
        self.conf_bar.setValue(0)
    
    def on_frame_received(self, frame):
        """Handle frame from capture worker."""
        fps = self.video_widget.latest_fps
        self.video_widget.setFrame(frame, fps)
        # Send to inference worker
        if self.inference_worker:
            self.inference_worker.on_frame_received(frame)
    
    def on_fps_received(self, fps):
        """Handle FPS update."""
        if self.video_widget.latest_frame is not None:
            self.video_widget.setFrame(self.video_widget.latest_frame, fps)
    
    def on_bbox_received(self, bbox, tracker_active):
        """Handle bbox update."""
        self.video_widget.setBBox(bbox, tracker_active)
    
    def on_inference_result(self, result):
        """Handle inference result."""
        self.video_widget.setResult(result)
        
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        # Update result display
        self.result_label.setText(label.upper() if label != "Unknown" else "UNKNOWN")
        self.conf_bar.setValue(int(confidence * 100))
        self.conf_text.setText(f"Confidence: {confidence*100:.1f}%")
        
        # Update recommendation
        from ml.config import RECOMMENDATION
        if label != "Unknown":
            rec = RECOMMENDATION.get(label, "No recommendation available.")
            self.rec_label.setText(rec)
        else:
            self.rec_label.setText("Please point the camera at a plastic object.")
        
        # Update map with filtered locations
        self.map_view.set_selected_plastic_type(label)
    
    def on_location_selected(self, location_id: str):
        """Handle location selection."""
        # TODO: Highlight location in map
        pass
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop()
        event.accept()

