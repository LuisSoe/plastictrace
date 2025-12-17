import cv2
import numpy as np

def clamp_bbox_xyxy(bbox, W, H, pad=12):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def detect_bbox(frame, min_area=2000, min_area_ratio=0.08, center_weight=3.0, use_sharpness=True):
    """
    Pick a 'near object in front' via heuristics:
      - large contour
      - near center
      - optionally sharper
    """
    H, W = frame.shape[:2]
    min_area = max(min_area, int(min_area_ratio * (H * W)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    # DEBUG: show edges (optional - uncomment to debug)
    # cv2.imshow("edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cx0, cy0 = W / 2.0, H / 2.0
    best, best_score = None, -1e18

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        x1, y1, x2, y2 = x, y, x + w, y + h

        cx = x + w / 2.0
        cy = y + h / 2.0
        dist = np.hypot(cx - cx0, cy - cy0) / np.hypot(cx0, cy0)

        score = area - center_weight * dist * (H * W)

        if use_sharpness:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                score += 200.0 * _sharpness(roi)

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    return best
