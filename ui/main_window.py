"""
Main window for PlastiTrace desktop application.
"""
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QSpinBox,
    QGroupBox, QFrame, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from ui.video_widget import VideoWidget
from ui.workers import CameraWorker, InferenceWorker
from ui.map_widget import MapWidget, LocationListWidget, SAMPLE_LOCATIONS
from ml.classifier import PlastiTraceClassifier


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, classifier: PlastiTraceClassifier):
        super().__init__()
        self.classifier = classifier
        
        # Workers
        self.camera_worker = None
        self.inference_worker = None
        self.inference_thread = None
        
        # State
        self.running = False
        
        self.setWindowTitle("PlastiTrace Desktop")
        self.setMinimumSize(1200, 800)
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Left: Video widget
        self.video_widget = VideoWidget()
        main_layout.addWidget(self.video_widget, 3)
        
        # Right: Tab widget with Controls and Map
        right_tabs = QTabWidget()
        right_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #334155;
                background-color: #1e293b;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #0f172a;
                color: #94a3b8;
                padding: 8px 16px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #f8fafc;
            }
        """)
        
        # Tab 1: Controls
        controls_panel = self.create_controls_panel()
        right_tabs.addTab(controls_panel, "Kontrol")
        
        # Tab 2: Map
        map_panel = self.create_map_panel()
        right_tabs.addTab(map_panel, "Peta Lokasi")
        
        main_layout.addWidget(right_tabs, 1)
    
    def create_controls_panel(self):
        """Create controls panel."""
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
        title = QLabel("PlastiTrace Controls")
        title.setStyleSheet("color: #f8fafc; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
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
            QPushButton:pressed {
                background-color: #047857;
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
        
        # Camera index
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
        
        # Inference interval
        inf_interval_layout = QHBoxLayout()
        inf_interval_layout.addWidget(QLabel("Inference Interval:"))
        self.inference_interval_spin = QSpinBox()
        self.inference_interval_spin.setMinimum(1)
        self.inference_interval_spin.setMaximum(10)
        self.inference_interval_spin.setValue(3)
        inf_interval_layout.addWidget(self.inference_interval_spin)
        inference_layout.addLayout(inf_interval_layout)
        
        # Redetect interval
        redetect_layout = QHBoxLayout()
        redetect_layout.addWidget(QLabel("Redetect Interval:"))
        self.redetect_interval_spin = QSpinBox()
        self.redetect_interval_spin.setMinimum(10)
        self.redetect_interval_spin.setMaximum(120)
        self.redetect_interval_spin.setValue(30)
        redetect_layout.addWidget(self.redetect_interval_spin)
        inference_layout.addLayout(redetect_layout)
        
        layout.addWidget(inference_group)
        
        # Stability settings
        stability_group = QGroupBox("Stability Settings")
        stability_group.setStyleSheet(camera_group.styleSheet())
        stability_layout = QVBoxLayout(stability_group)
        
        # Confidence threshold
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
        
        # Stabilize toggle
        self.stabilize_check = QCheckBox("Enable Stabilization")
        self.stabilize_check.setChecked(True)
        stability_layout.addWidget(self.stabilize_check)
        
        # Alpha slider
        alpha_layout = QVBoxLayout()
        alpha_layout.addWidget(QLabel("Smoothing Alpha:"))
        alpha_slider_layout = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(10)
        self.alpha_slider.setMaximum(90)
        self.alpha_slider.setValue(60)
        self.alpha_label = QLabel("0.60")
        alpha_slider_layout.addWidget(self.alpha_slider)
        alpha_slider_layout.addWidget(self.alpha_label)
        alpha_layout.addLayout(alpha_slider_layout)
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_label.setText(f"{v/100:.2f}")
        )
        stability_layout.addLayout(alpha_layout)
        
        # Hysteresis margin
        hyst_layout = QVBoxLayout()
        hyst_layout.addWidget(QLabel("Hysteresis Margin:"))
        hyst_slider_layout = QHBoxLayout()
        self.hyst_slider = QSlider(Qt.Horizontal)
        self.hyst_slider.setMinimum(0)
        self.hyst_slider.setMaximum(30)
        self.hyst_slider.setValue(10)
        self.hyst_label = QLabel("0.10")
        hyst_slider_layout.addWidget(self.hyst_slider)
        hyst_slider_layout.addWidget(self.hyst_label)
        hyst_layout.addLayout(hyst_slider_layout)
        self.hyst_slider.valueChanged.connect(
            lambda v: self.hyst_label.setText(f"{v/100:.2f}")
        )
        stability_layout.addLayout(hyst_layout)
        
        layout.addWidget(stability_group)
        
        # Display settings
        display_group = QGroupBox("Display Settings")
        display_group.setStyleSheet(camera_group.styleSheet())
        display_layout = QVBoxLayout(display_group)
        
        # Alpha slider for overlay
        overlay_alpha_layout = QVBoxLayout()
        overlay_alpha_layout.addWidget(QLabel("Overlay Alpha:"))
        overlay_alpha_slider_layout = QHBoxLayout()
        self.overlay_alpha_slider = QSlider(Qt.Horizontal)
        self.overlay_alpha_slider.setMinimum(10)
        self.overlay_alpha_slider.setMaximum(90)
        self.overlay_alpha_slider.setValue(60)
        self.overlay_alpha_label = QLabel("0.60")
        overlay_alpha_slider_layout.addWidget(self.overlay_alpha_slider)
        overlay_alpha_slider_layout.addWidget(self.overlay_alpha_label)
        overlay_alpha_layout.addLayout(overlay_alpha_slider_layout)
        self.overlay_alpha_slider.valueChanged.connect(
            lambda v: (self.overlay_alpha_label.setText(f"{v/100:.2f}"),
                      self.video_widget.setOverlayAlpha(v/100))
        )
        display_layout.addLayout(overlay_alpha_layout)
        
        layout.addWidget(display_group)
        
        # Tracker settings
        tracker_group = QGroupBox("Tracker Settings")
        tracker_group.setStyleSheet(camera_group.styleSheet())
        tracker_layout = QVBoxLayout(tracker_group)
        
        self.tracker_check = QCheckBox("Enable Tracker")
        self.tracker_check.setChecked(True)
        tracker_layout.addWidget(self.tracker_check)
        
        layout.addWidget(tracker_group)
        
        # Results display
        results_group = QGroupBox("Last Result")
        results_group.setStyleSheet(camera_group.styleSheet())
        results_layout = QVBoxLayout(results_group)
        
        self.result_label = QLabel("N/A")
        self.result_label.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.result_label)
        
        self.conf_label = QLabel("Confidence: N/A")
        self.conf_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.conf_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.conf_label)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        
        return panel
    
    def create_map_panel(self):
        """Create map panel with location map and list."""
        panel = QFrame()
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
        title = QLabel("Peta Lokasi Pembuangan Sampah")
        title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Category label
        self.map_category_label = QLabel("Kategori: Pilih kategori plastik")
        self.map_category_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(self.map_category_label)
        
        # Map widget
        self.map_widget = MapWidget()
        self.map_widget.setMinimumHeight(250)
        layout.addWidget(self.map_widget)
        
        # Location list
        self.location_list = LocationListWidget()
        self.location_list.setMinimumHeight(200)
        layout.addWidget(self.location_list)
        
        # Info label
        info_label = QLabel("Lokasi akan diperbarui otomatis berdasarkan kategori plastik yang terdeteksi")
        info_label.setStyleSheet("color: #64748b; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections."""
        # Connect control changes to workers (will be connected when workers are created)
        pass
    
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
        
        # Create camera worker
        self.camera_worker = CameraWorker(camera_index)
        
        # Connect frame updates (FPS is updated separately)
        def on_frame_ready(frame):
            self.video_widget.setFrame(frame, self.video_widget.latest_fps)
        
        def on_fps_ready(fps):
            if self.video_widget.latest_frame is not None:
                self.video_widget.setFrame(self.video_widget.latest_frame, fps)
        
        self.camera_worker.frameReady.connect(on_frame_ready)
        self.camera_worker.fpsReady.connect(on_fps_ready)
        self.camera_worker.bboxReady.connect(self.video_widget.setBBox)
        
        # Update camera worker settings
        self.camera_worker.set_redetect_interval(self.redetect_interval_spin.value())
        self.camera_worker.set_tracker_enabled(self.tracker_check.isChecked())
        
        # Connect redetect interval changes
        self.redetect_interval_spin.valueChanged.connect(
            self.camera_worker.set_redetect_interval
        )
        self.tracker_check.toggled.connect(
            self.camera_worker.set_tracker_enabled
        )
        
        # Create inference worker
        self.inference_worker = InferenceWorker(self.classifier)
        self.inference_worker.resultReady.connect(self.on_inference_result)
        
        # Update inference worker settings
        self.inference_worker.set_inference_interval(self.inference_interval_spin.value())
        self.inference_worker.set_confidence_threshold(self.conf_thresh_slider.value() / 100.0)
        self.inference_worker.set_stabilize_enabled(self.stabilize_check.isChecked())
        self.inference_worker.set_alpha(self.alpha_slider.value() / 100.0)
        self.inference_worker.set_hysteresis_margin(self.hyst_slider.value() / 100.0)
        
        # Connect inference worker settings
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
        self.hyst_slider.valueChanged.connect(
            lambda v: self.inference_worker.set_hysteresis_margin(v / 100.0)
        )
        
        # Store latest frame and bbox for inference
        self._latest_frame_for_inference = None
        self._latest_bbox_for_inference = None
        
        # Connect camera frames to inference worker
        def on_frame_ready(frame):
            self._latest_frame_for_inference = frame
            if self._latest_frame_for_inference is not None:
                self.inference_worker.inferenceRequested.emit(
                    self._latest_frame_for_inference,
                    self._latest_bbox_for_inference
                )
        
        def on_bbox_ready(bbox, tracker_active):
            self._latest_bbox_for_inference = bbox
            if self._latest_frame_for_inference is not None:
                self.inference_worker.inferenceRequested.emit(
                    self._latest_frame_for_inference,
                    self._latest_bbox_for_inference
                )
        
        self.camera_worker.frameReady.connect(on_frame_ready)
        self.camera_worker.bboxReady.connect(on_bbox_ready)
        
        # Move inference worker to thread
        self.inference_thread = QThread()
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.start()
        
        # Start camera worker
        self.camera_worker.start()
        
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
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        """)
    
    def stop(self):
        """Stop camera and inference workers."""
        if not self.running:
            return
        
        # Stop camera worker
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        
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
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        
        # Clear video widget
        self.video_widget.setFrame(None, 0.0)
        self.video_widget.setBBox(None, False)
        self.video_widget.setResult(None)
        self.result_label.setText("N/A")
        self.conf_label.setText("Confidence: N/A")
    
    def on_inference_result(self, result):
        """Handle inference result."""
        self.video_widget.setResult(result)
        
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0.0)
        raw_label = result.get("raw_label", "Unknown")
        raw_conf = result.get("raw_conf", 0.0)
        
        # Update result display
        if label == "Unknown":
            self.result_label.setText(f"Unknown (raw: {raw_label})")
            self.conf_label.setText(f"Confidence: {raw_conf:.2f} (below threshold)")
        else:
            self.result_label.setText(label)
            self.conf_label.setText(f"Confidence: {confidence:.2f}")
        
        # Update map with locations for detected category
        self.update_map_locations(label)
    
    def update_map_locations(self, category):
        """Update map widget with locations for the given category."""
        if category == "Unknown" or category not in SAMPLE_LOCATIONS:
            # Clear map
            self.map_widget.set_locations([])
            self.location_list.set_locations([])
            self.map_category_label.setText("Kategori: Tidak terdeteksi")
            return
        
        # Get locations for this category
        locations = SAMPLE_LOCATIONS.get(category, [])
        
        # Update map widget
        self.map_widget.set_locations(locations)
        
        # Update location list
        self.location_list.set_locations(locations)
        
        # Update category label
        self.map_category_label.setText(f"Kategori: {category} ({len(locations)} lokasi ditemukan)")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop()
        event.accept()

