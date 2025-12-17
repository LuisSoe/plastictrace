import cv2
import numpy as np
import time
import threading

from ml.config import RECOMMENDATION
from vision.bbox_detector import detect_bbox, clamp_bbox_xyxy
from vision.bbox_tracker import BBoxTracker
from vision.smoothing import EMASmoother


class CameraLoop:
    def __init__(self, classifier, camera_index=0,
                 inference_interval=3, redetect_interval=30):
        self.classifier = classifier
        self.camera_index = camera_index
        self.inference_interval = inference_interval
        self.redetect_interval = redetect_interval

        self.tracker = BBoxTracker()      # auto-fallback
        self.smoother = EMASmoother(alpha=0.7)

        self.frame_count = 0
        self.last_result = None

        self.fps_start = time.time()
        self.fps_n = 0
        self.fps = 0.0

        self._infer_running = False
        self._infer_lock = threading.Lock()

    def _update_fps(self):
        self.fps_n += 1
        dt = time.time() - self.fps_start
        if dt >= 1.0:
            self.fps = self.fps_n / dt
            self.fps_n = 0
            self.fps_start = time.time()

    def _wrap_text(self, text, font, font_scale, max_width):
        words = text.split()
        lines = []
        cur = ""
        for word in words:
            test = f"{cur} {word}".strip()
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, 1)
            if tw > max_width and cur:
                lines.append(cur)
                cur = word
            else:
                cur = test
        if cur:
            lines.append(cur)
        return lines

    def _infer_async(self, roi_bgr):
        with self._infer_lock:
            if self._infer_running:
                return
            self._infer_running = True

        def job():
            try:
                result = self.classifier.predict_from_bgr(roi_bgr)
                if result is not None:
                    conf = result.get("confidence", 0.0)
                    result["confidence"] = self.smoother.update_confidence(conf)
                    self.last_result = result
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                with self._infer_lock:
                    self._infer_running = False

        threading.Thread(target=job, daemon=True).start()

    def _draw_overlay(self, frame, bbox, result):
        H, W = frame.shape[:2]

        # bbox + label near bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if result is not None:
                label = result.get("label", "-")
                conf = float(result.get("confidence", 0.0)) * 100.0
                text = f"{label} {conf:.1f}%"

                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)

                y_text = y1 - 10
                if y_text - th - 8 < 0:
                    y_text = y1 + th + 10

                x_text = max(0, min(x1, W - tw - 8))

                cv2.rectangle(frame,
                              (x_text, y_text - th - 8),
                              (x_text + tw + 8, y_text + 4),
                              (0, 0, 0), -1)
                cv2.putText(frame, text, (x_text + 4, y_text),
                            font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # top panel
        if result is not None:
            panel_h = min(180, H)
            alpha = 0.65
            panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
            frame[:panel_h, :] = cv2.addWeighted(frame[:panel_h, :], 1 - alpha, panel, alpha, 0)

            label = result.get("label", "-")
            confidence = float(result.get("confidence", 0.0))
            rec = RECOMMENDATION.get(label, "")

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                        (12, 40), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            if rec:
                lines = self._wrap_text(rec, font, 0.6, max_width=W - 24)
                y0 = 80
                for i, line in enumerate(lines[:3]):
                    cv2.putText(frame, line, (12, y0 + i * 28),
                                font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (W - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return

        print("Camera started. Press ESC to exit.")
        cv2.namedWindow("PlastiTrace Desktop - Realtime", cv2.WINDOW_NORMAL)
        # DEBUG: uncomment to show edges window
        # cv2.namedWindow("edges", cv2.WINDOW_NORMAL)

        bbox = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                H, W = frame.shape[:2]
                self.frame_count += 1
                self._update_fps()

                # tracking update (cheap) unless redetect frame
                should_redetect = (not self.tracker.is_active()) or (self.frame_count % self.redetect_interval == 0)

                if not should_redetect and self.tracker.is_active():
                    tracked = self.tracker.update(frame)
                    if tracked is not None:
                        bbox = tracked
                    else:
                        bbox = None

                # redetect
                if should_redetect:
                    detected = detect_bbox(frame)
                    if detected is not None:
                        clamped = clamp_bbox_xyxy(detected, W, H)
                        if clamped is not None:
                            bbox = clamped
                            try:
                                self.tracker.init(frame, bbox)
                            except RuntimeError:
                                # tracker unavailable -> keep bbox, no tracking
                                pass

                # Guaranteed bbox fallback (if contour detector fails, use center box)
                if bbox is None:
                    # fallback: center box 60% frame (always valid)
                    x1 = int(W * 0.2)
                    y1 = int(H * 0.2)
                    x2 = int(W * 0.8)
                    y2 = int(H * 0.8)
                    bbox = (x1, y1, x2, y2)

                # smooth bbox
                if bbox is not None:
                    bbox = self.smoother.update_bbox(bbox)

                # choose ROI
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        roi = frame
                else:
                    roi = frame

                # async inference throttled
                if self.frame_count % self.inference_interval == 0:
                    self._infer_async(roi)

                # draw + show
                out = self._draw_overlay(frame, bbox, self.last_result)
                
                # DEBUG overlay: show bbox status
                cv2.putText(out, f"bbox: {'YES' if bbox is not None else 'NO'}",
                            (10, out.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow("PlastiTrace Desktop - Realtime", out)

                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released")
