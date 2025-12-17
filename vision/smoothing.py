class EMASmoother:
    def __init__(self, alpha=0.7):
        self.alpha = float(alpha)
        self._bbox = None
        self._conf = None

    def update_bbox(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        v = (float(x1), float(y1), float(x2), float(y2))
        if self._bbox is None:
            self._bbox = v
        else:
            self._bbox = tuple(self.alpha * nv + (1 - self.alpha) * ov for nv, ov in zip(v, self._bbox))
        x1, y1, x2, y2 = map(int, self._bbox)
        return (x1, y1, x2, y2)

    def update_confidence(self, conf):
        c = float(conf)
        if self._conf is None:
            self._conf = c
        else:
            self._conf = self.alpha * c + (1 - self.alpha) * self._conf
        return float(self._conf)
