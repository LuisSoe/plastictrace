"""
InferenceWorker: Processes frames with latest-frame-wins queue strategy.
"""
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QMutex
from ml.classifier import PlastiTraceClassifier
from vision.bbox_detector import detect_bbox, clamp_bbox_xyxy
from vision.bbox_tracker import BBoxTracker
from vision.smoothing import EMASmoother
from realtime.stability import ProbSmoother, HysteresisLabel, apply_confidence_gating
from ml.config import CLASSES


class InferenceWorker(QObject):
    """
    Worker object for inference (moved to QThread).
    Uses latest-frame-wins strategy: only processes the most recent frame.
    """
    
    resultReady = pyqtSignal(dict)  # {label, confidence, probs, raw_label, raw_conf, bbox}
    bboxReady = pyqtSignal(object, bool)  # (bbox_xyxy_or_none, tracker_active)
    
    def __init__(self, classifier: PlastiTraceClassifier):
        super().__init__()
        self.classifier = classifier
        
        # Latest-frame buffer (overwrite strategy)
        self._latest_frame = None
        self._latest_bbox = None
        self._mutex = QMutex()
        self._busy = False
        
        # Bbox pipeline
        self.tracker = None
        try:
            self.tracker = BBoxTracker()
        except:
            pass
        
        self.bbox_smoother = EMASmoother(alpha=0.7)
        self.frame_count = 0
        self.redetect_interval = 30
        self.current_bbox = None
        self.tracker_active = False
        
        # Stability components
        self.prob_smoother = ProbSmoother(alpha=0.5)
        self.hysteresis = HysteresisLabel(min_conf=0.55, switch_margin=0.10)
        self.confidence_threshold = 0.65
        self.stabilize_enabled = True
        
        # Frame counting for throttling
        self.inference_interval = 3
        
        # Timer for periodic processing
        self._process_timer = None
    
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
    
    @pyqtSlot(object)
    def on_frame_received(self, frame_bgr):
        """
        Receive latest frame (latest-frame-wins strategy).
        
        Args:
            frame_bgr: BGR frame (numpy array)
        """
        self._mutex.lock()
        try:
            if frame_bgr is not None:
                self._latest_frame = frame_bgr.copy()
            else:
                self._latest_frame = None
        finally:
            self._mutex.unlock()
        
        # Process bbox detection/tracking
        self._process_bbox()
        
        # Throttle inference
        self.frame_count += 1
        if self.frame_count % self.inference_interval == 0:
            self._process_inference()
    
    def _process_bbox(self):
        """Process bbox detection and tracking."""
        if self._latest_frame is None:
            return
        
        frame = self._latest_frame
        H, W = frame.shape[:2]
        
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
        
        # Emit bbox
        self.bboxReady.emit(self.current_bbox, self.tracker_active)
        self._latest_bbox = self.current_bbox
    
    def _process_inference(self):
        """Process inference on latest frame."""
        if self._busy:
            return  # Skip if busy
        
        self._mutex.lock()
        try:
            frame = self._latest_frame
            bbox = self._latest_bbox
        finally:
            self._mutex.unlock()
        
        if frame is None:
            return
        
        self._busy = True
        
        try:
            H, W = frame.shape[:2]
            
            # Extract ROI
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
                # Fallback: use center region
                x1, y1 = int(W * 0.2), int(H * 0.2)
                x2, y2 = int(W * 0.8), int(H * 0.8)
                roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # Run inference
            result = self.classifier.predict_from_bgr(roi)
            if not result:
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
                "raw_conf": final_raw_conf,
                "bbox": bbox
            })
        
        except Exception as e:
            print(f"Inference error: {e}")
        
        finally:
            self._busy = False

