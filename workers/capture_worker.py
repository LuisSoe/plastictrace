"""
CaptureWorker: Reads frames from camera and emits latest frame (queue size 1).
"""
import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal


class CaptureWorker(QThread):
    """
    Worker thread for camera capture.
    Emits latest frame only (drops old frames).
    """
    
    frameReady = pyqtSignal(object)  # frame_bgr (numpy array)
    fpsReady = pyqtSignal(float)  # fps value
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        
        # FPS calculation
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.fps = 0.0
    
    def run(self):
        """Main capture loop."""
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.fps_start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.fps_frame_count += 1
            
            # Calculate and emit FPS periodically
            now = time.time()
            if now - self.fps_start_time >= 1.0:
                self.fps = self.fps_frame_count / (now - self.fps_start_time)
                self.fpsReady.emit(self.fps)
                self.fps_frame_count = 0
                self.fps_start_time = now
            
            # Emit latest frame (drops old frames automatically via signal queue)
            # Use direct connection to ensure latest frame wins
            self.frameReady.emit(frame.copy())
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.wait()

