# Rebuild PlastiTrace Desktop UI with PyQt5 and Add Location Map Feature

## Summary

This PR refactors the PlastiTrace desktop application UI from PySide6 to PyQt5 and adds a new location map feature that displays nearest waste disposal locations based on detected plastic categories.

## Changes

### 🔄 UI Refactoring (PySide6 → PyQt5)

- **Refactored from PySide6 to PyQt5** for better compatibility and stability
- **Modular architecture** with separate components for better maintainability
- **Worker thread architecture** prevents GUI blocking during camera capture and inference

### ✨ New Features

#### 1. Realtime Webcam Preview

- Real-time video preview in PyQt5 window (no cv2.imshow)
- Smooth frame rendering with proper coordinate mapping
- FPS counter display

#### 2. Stability Module (`realtime/stability.py`)

- **ProbSmoother**: EMA smoothing of probability vectors with renormalization
- **HysteresisLabel**: Hysteresis-based label switching to prevent flicker
- **Confidence gating**: Returns "Unknown" if confidence below threshold (default 0.65)

#### 3. Worker Threads (`ui/workers.py`)

- **CameraWorker**: Camera capture thread with bbox detection/tracking and FPS calculation
- **InferenceWorker**: Inference thread with drop-frame strategy (never queues infinitely)
- Both workers use signals for thread-safe communication

#### 4. UI Components

- **`ui/main_window.py`**: Main window with controls panel and map tabs
- **`ui/video_widget.py`**: Video display widget with overlay rendering
- **`ui/overlay.py`**: Overlay drawing utilities (bbox, labels, FPS, status)
- **`ui/map_widget.py`**: Map widget showing nearest waste disposal locations

#### 5. Location Map Feature 🗺️

- **Automatic updates**: Map updates automatically when plastic category is detected
- **Categorized locations**: Shows nearest waste disposal locations by plastic type (HDPE, PET, PP, PS)
- **Visual indicators**: Color-coded markers based on distance (< 3km green, 3-5km yellow, > 5km red)
- **Location list**: Detailed list with name, address, and distance
- **User location marker**: Shows user's current location on map

### 🎯 Overlay Features

- Green bbox with label and confidence near bbox
- Top translucent panel with recycling recommendation (max 3 lines)
- FPS counter (yellow, top-right)
- Status display (cyan, bottom-left): `bbox YES/NO | result YES/NO | tracker ACTIVE/INACTIVE`

### ⚡ Performance Improvements

- **Drop-frame strategy**: Inference worker drops frames if busy (never queues infinitely)
- **Latest frame only**: UI updates with latest frame/result only (prevents lag)
- **Non-blocking**: All heavy operations in worker threads
- **Throttled inference**: Configurable inference interval (1-10 frames)

### 🔧 Technical Details

#### New Files

- `realtime/stability.py` - Stability and smoothing modules
- `ui/main_window.py` - Main application window
- `ui/video_widget.py` - Video display widget
- `ui/workers.py` - Worker threads
- `ui/overlay.py` - Overlay drawing utilities
- `ui/map_widget.py` - Map widget for location display

#### Modified Files

- `app.py` - Updated to use new PyQt5 UI
- `requirements.txt` - Added PyQt5, updated numpy>=2.3.5 for Python 3.14 compatibility

#### Architecture

```
app.py
├── ui/
│   ├── main_window.py      # Main window with tabs
│   ├── video_widget.py     # Video display
│   ├── workers.py          # Camera & inference workers
│   ├── overlay.py          # Overlay drawing
│   └── map_widget.py       # Location map
├── realtime/
│   └── stability.py        # Stability & smoothing
└── ml/                     # Existing classifier
```

## Testing

- ✅ Tested on Python 3.14 with PyQt5
- ✅ Camera capture works with worker threads
- ✅ Inference with stability features (EMA, hysteresis, gating)
- ✅ Map widget displays locations correctly
- ✅ UI remains responsive during inference
- ✅ Graceful degradation when tracker unavailable

## Configuration

The app includes comprehensive controls:

- Camera index (0-5)
- Inference interval (1-10 frames)
- Redetect interval (10-120 frames)
- Confidence threshold (0.40-0.90, default 0.65)
- Stabilization toggle
- Smoothing alpha (0.1-0.9, default 0.6)
- Hysteresis margin (0.00-0.30, default 0.10)
- Tracker enable/disable
- Overlay alpha (0.1-0.9, default 0.6)

## Sample Location Data

The map widget includes sample location data for Jakarta, Indonesia. In production, this can be:

- Loaded from a database
- Fetched from a geocoding API
- Integrated with GPS for user location

## Breaking Changes

- **Entry point changed**: `app.py` now uses PyQt5 GUI instead of CLI
- **Dependencies**: Requires PyQt5 (added to requirements.txt)
- **Python version**: Tested with Python 3.14, numpy>=2.3.5 required

## Screenshots

(Add screenshots of the new UI and map feature)

## Checklist

- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] Comments added for complex code
- [x] Documentation updated (README if needed)
- [x] No new warnings generated
- [x] Tests pass (if applicable)
- [x] Dependencies updated in requirements.txt

## Related Issues

(Link to any related issues if applicable)
