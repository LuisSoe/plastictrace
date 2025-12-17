import cv2

def _create_tracker_any():
    legacy = getattr(cv2, "legacy", None)

    def mk(attr):
        fn = getattr(cv2, attr, None)
        if fn:
            return fn()
        if legacy:
            fn2 = getattr(legacy, attr, None)
            if fn2:
                return fn2()
        return None

    for attr in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
        t = mk(attr)
        if t is not None:
            return t
    return None

class BBoxTracker:
    def __init__(self):
        self.tracker = _create_tracker_any()
        self.active = False

    def is_active(self):
        return self.active and self.tracker is not None

    def init(self, frame, bbox_xyxy):
        if self.tracker is None:
            raise RuntimeError("No OpenCV tracker available. Install opencv-contrib-python.")
        x1, y1, x2, y2 = bbox_xyxy
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ok = self.tracker.init(frame, (int(x1), int(y1), int(w), int(h)))
        self.active = bool(ok)

    def update(self, frame):
        if not self.is_active():
            return None
        ok, box = self.tracker.update(frame)
        if not ok:
            self.active = False
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)
