"""
Worker threads for camera capture and inference.
"""
import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt
from ml.classifier import PlastiTraceClassifier
from vision.bbox_detector import detect_bbox, clamp_bbox_xyxy
from vision.bbox_tracker import BBoxTracker
from vision.smoothing import EMASmoother
from realtime.stability import ProbSmoother, HysteresisLabel, apply_confidence_gating
from ml.config import CLASSES


class CameraWorker(QThread):
    """Worker thread for camera capture and bbox detection/tracking."""
    
    frameReady = pyqtSignal(object)  # frame_bgr (numpy array as object)
    fpsReady = pyqtSignal(float)  # fps value
    bboxReady = pyqtSignal(object, bool)  # (bbox_xyxy_or_none, tracker_active)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        
        # Bbox pipeline
        self.tracker = None
        try:
            self.tracker = BBoxTracker()
        except:
            pass  # Tracker optional
        
        self.bbox_smoother = EMASmoother(alpha=0.7)
        self.frame_count = 0
        self.redetect_interval = 30
        self.current_bbox = None
        self.tracker_active = False
        
        # FPS calculation
        self.fps_start_time = None
        self.fps_frame_count = 0
    
    def set_redetect_interval(self, interval):
        """Set redetection interval (frames)."""
        self.redetect_interval = max(10, min(120, int(interval)))
    
    def set_tracker_enabled(self, enabled):
        """Enable/disable tracker."""
        if not enabled:
            self.tracker = None
            self.tracker_active = False
        else:
            try:
                self.tracker = BBoxTracker()
            except:
                self.tracker = None
                self.tracker_active = False
    
    def run(self):
        """Main camera loop."""
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        self.fps_start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            self.frame_count += 1
            self.fps_frame_count += 1
            
            # Calculate and emit FPS periodically
            now = time.time()
            if now - self.fps_start_time >= 1.0:
                fps = self.fps_frame_count / (now - self.fps_start_time)
                self.fpsReady.emit(fps)
                self.fps_frame_count = 0
                self.fps_start_time = now
            
            # Bbox pipeline
            should_redetect = (
                (self.frame_count % self.redetect_interval == 0) or
                (self.tracker is None) or
                (not self.tracker_active)
            )
            
            if not should_redetect and self.tracker is not None and self.tracker.is_active():
                # Update tracker
                tracked = self.tracker.update(frame)
                if tracked is not None:
                    self.current_bbox = tracked
                    self.tracker_active = True
                else:
                    self.tracker_active = False
                    should_redetect = True
            
            if should_redetect:
                # Detect new bbox
                detected = detect_bbox(frame)
                if detected is not None:
                    clamped = clamp_bbox_xyxy(detected, W, H)
                    if clamped is not None:
                        self.current_bbox = clamped
                        # Initialize tracker if available
                        if self.tracker is not None:
                            try:
                                self.tracker.init(frame, self.current_bbox)
                                self.tracker_active = True
                            except:
                                self.tracker_active = False
                else:
                    self.current_bbox = None
                    self.tracker_active = False
            
            # Smooth bbox
            if self.current_bbox is not None:
                self.current_bbox = self.bbox_smoother.update_bbox(self.current_bbox)
            
            # Emit signals
            self.frameReady.emit(frame.copy())
            self.bboxReady.emit(self.current_bbox, self.tracker_active)
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.wait()


class InferenceWorker(QObject):
    """Worker object for inference (moved to QThread)."""
    
    resultReady = pyqtSignal(dict)  # {label, confidence, probs, raw_label, raw_conf}
    inferenceRequested = pyqtSignal(object, object)  # (frame_bgr, bbox_xyxy_or_none) - using object for numpy arrays
    
    def __init__(self, classifier: PlastiTraceClassifier):
        super().__init__()
        self.classifier = classifier
        
        # Drop-frame strategy: only store latest request
        self.latest_frame = None
        self.latest_bbox = None
        self.busy = False
        
        # Stability components
        self.prob_smoother = ProbSmoother(alpha=0.6)
        self.hysteresis = HysteresisLabel(min_conf=0.55, switch_margin=0.10)
        self.confidence_threshold = 0.65
        self.stabilize_enabled = True
        
        # Frame counting for throttling
        self.frame_count = 0
        self.inference_interval = 3
        
        # Connect signal to slot
        self.inferenceRequested.connect(self.request)
    
    def set_inference_interval(self, interval):
        """Set inference interval (frames)."""
        self.inference_interval = max(1, min(10, int(interval)))
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for gating."""
        self.confidence_threshold = float(threshold)
    
    def set_stabilize_enabled(self, enabled):
        """Enable/disable stabilization."""
        self.stabilize_enabled = bool(enabled)
        if not enabled:
            self.prob_smoother.reset()
            self.hysteresis.reset()
    
    def set_alpha(self, alpha):
        """Set EMA alpha for probability smoothing."""
        self.prob_smoother.alpha = float(alpha)
    
    def set_hysteresis_margin(self, margin):
        """Set hysteresis switch margin."""
        self.hysteresis.switch_margin = float(margin)
    
    @pyqtSlot(object, object)
    def request(self, frame_bgr, bbox_xyxy_or_none):
        """
        Request inference (drop-frame strategy).
        
        Args:
            frame_bgr: BGR frame (numpy array) or None
            bbox_xyxy_or_none: Bounding box (x1, y1, x2, y2) or None
        """
        # Drop-frame: overwrite latest request
        if frame_bgr is not None:
            try:
                self.latest_frame = frame_bgr.copy()
            except:
                self.latest_frame = None
        else:
            self.latest_frame = None
        self.latest_bbox = bbox_xyxy_or_none
        
        # Throttle by inference_interval
        self.frame_count += 1
        if self.frame_count % self.inference_interval != 0:
            return
        
        # If busy, drop this frame
        if self.busy:
            return
        
        # Process in this thread (will be moved to worker thread)
        self._process_inference()
    
    def _process_inference(self):
        """Process inference request."""
        if self.latest_frame is None:
            return
        
        self.busy = True
        
        try:
            frame = self.latest_frame
            bbox = self.latest_bbox
            
            # Extract ROI
            H, W = frame.shape[:2]
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                # Expand ROI slightly (1.2x)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = (x2 - x1) * 1.2, (y2 - y1) * 1.2
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(W, int(cx + w / 2))
                y2 = min(H, int(cy + h / 2))
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame
            
            if roi.size == 0:
                self.busy = False
                return
            
            # Run inference
            result = self.classifier.predict_from_bgr(roi)
            if not result:
                self.busy = False
                return
            
            probs = np.array(result.get("probs", [0.25] * len(CLASSES)), dtype=np.float32)
            raw_label = result.get("label", CLASSES[0])
            raw_conf = result.get("confidence", 0.0)
            
            # Apply stability
            if self.stabilize_enabled:
                # EMA smooth probabilities
                probs = self.prob_smoother.update(probs)
                
                # Get label from smoothed probs
                idx = int(np.argmax(probs))
                label = CLASSES[idx]
                conf = float(probs[idx])
                
                # Apply hysteresis
                label, conf = self.hysteresis.update(label, conf)
            else:
                label = raw_label
                conf = raw_conf
            
            # Apply confidence gating
            gated_label, final_raw_label, final_raw_conf = apply_confidence_gating(
                label, conf, self.confidence_threshold
            )
            
            # Emit result
            self.resultReady.emit({
                "label": gated_label,
                "confidence": conf if gated_label != "Unknown" else 0.0,
                "probs": probs.tolist(),
                "raw_label": final_raw_label,
                "raw_conf": final_raw_conf
            })
        
        except Exception as e:
            print(f"Inference error: {e}")
        
        finally:
            self.busy = False

