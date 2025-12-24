# PlastiTrace Desktop Refactoring Summary

## Overview

Comprehensive refactoring of the PyQt desktop application to separate realtime pipeline from UI, improve performance, and add modern mapping capabilities.

## Key Changes

### 1. Architecture: Separated Realtime Pipeline from UI ✅

**Problem**: UI lagging because inference + camera running in GUI thread.

**Solution**: Implemented proper worker architecture with latest-frame-wins strategy.

**Files Created**:

- `workers/capture_worker.py`: QThread for camera capture, emits latest frame only
- `workers/inference_worker.py`: QObject worker for inference with queue size 1 (overwrite buffer)

**Key Features**:

- `CaptureWorker`: Reads frames from camera → emits frame (drops old frames automatically)
- `InferenceWorker`: Receives latest frame → preprocess → model → result
- UI thread: Only renders QImage/QPixmap + updates overlay + updates map panel
- Throttling: UI update 20-30 FPS, inference 5-15 FPS
- Pre-allocated buffers, avoids repeated RGB conversion

### 2. Camera View: Clean Overlay + Anti-Flicker ✅

**Problem**: Drawing bbox on numpy image is costly and causes flicker.

**Solution**: Separate overlay widget for clean rendering.

**Files Created**:

- `ui/overlay_widget.py`: Transparent overlay widget for bbox, labels, confidence bars
- Updated `ui/video_widget.py`: Only renders video frame, overlays handled separately

**Key Features**:

- `VideoWidget`: QLabel/PaintEvent for video only
- `OverlayWidget`: Transparent QWidget on top for bbox + label + confidence bar
- No numpy drawing - all rendering via QPainter
- Smooth, anti-flicker display

### 3. Stabilization: Enhanced Temporal Smoothing ✅

**Problem**: Label jumping around, predictions unstable.

**Solution**: Two-layer smoothing with persistence.

**Files Created**:

- `domain/filtering.py`: `TemporalSmoother` and `HysteresisGate`

**Key Features**:

- **Temporal EMA**: `p_t = α p_t + (1-α) p_{t-1}` with α ~ 0.35–0.6
- **Hysteresis + Gating**:
  - "Unknown" if confidence < 0.65
  - Switch to new label only if persists for N frames (default 5) or margin difference large
- Result: UI feels "mature" (no label jumping)

### 4. Mapping UI: QWebEngine + Leaflet ✅

**Problem**: Simple map widget not suitable for workflow panel.

**Solution**: Full-featured mapping with QWebEngine + Leaflet.

**Files Created**:

- `ui/map_view.py`: Complete mapping widget with search, filters, location list
- `assets/map/leaflet.html`: Leaflet map with marker clustering

**Key Features**:

- **3-Panel Layout**:
  - Left: Detection results + filter material
  - Center: Video realtime
  - Right: Map + list drop-off (synchronized)
- **Map Panel Components**:
  - Search bar + radius slider (1-20 km)
  - Material chips (PET/HDPE/PP/PS) with state: selected/disabled
  - Map with marker clustering
  - Location list cards with details
- **QWebChannel Bridge**: 2-way communication between PyQt and Leaflet
- **Features**:
  - Marker clustering (many locations stay lightweight)
  - Selected marker highlight + pan/zoom auto
  - Card list ↔ marker sync (click card: map focus; click marker: card select)
  - Navigate button: opens Google Maps directions URL

### 5. Data Model & Filtering ✅

**Files Created**:

- `domain/models.py`: `Detection` and `Location` dataclasses
- `domain/geo.py`: Haversine distance, radius filtering

**Key Features**:

- In-memory structure: `Location(id, name, lat, lon, address, hours, phone, types:set[str], source, updated_at)`
- Spatial filtering: Haversine distance calculation
- Query flow: filter by types → compute distance → sort by distance → push top N to map + list
- For small scale (<= 5k points): brute force Haversine OK
- For large scale: ready for sklearn.neighbors.BallTree or rtree

### 6. Modern UX Details ✅

**Features Added**:

- Material chips with state: selected/disabled (when model output "Unknown")
- Confidence bar + explanation
- Empty states: "No centers found within 3 km for PET"
- Debounce search (300 ms) so map doesn't spam update
- Persist preferences via QSettings (last radius, last type filter, last location)

### 7. Folder Structure Reorganization ✅

**New Structure**:

```
plastitrace/
  app.py
  ui/
    main_window_new.py  # New 3-panel layout
    video_widget.py      # Video only (no overlays)
    overlay_widget.py   # Separate overlay
    map_view.py          # QWebEngine + Leaflet
  workers/
    capture_worker.py    # Camera capture
    inference_worker.py  # ML inference
  domain/
    models.py            # Detection, Location dataclasses
    filtering.py         # Gating + smoothing + hysteresis
    geo.py               # Haversine, radius filter
  assets/
    map/
      leaflet.html      # Leaflet map HTML
  ml/
    model.onnx / tflite
    preprocess.py
```

### 8. Main Window: 3-Panel Layout ✅

**File**: `ui/main_window_new.py`

**Layout**:

- **Left Panel**: Detection results + controls
  - Result label + confidence bar
  - Recycling guide
  - Start/Stop button
  - Camera settings
  - Inference settings
  - Stability settings
- **Center Panel**: Video realtime
  - VideoWidget (video only)
  - OverlayWidget (transparent overlay on top)
- **Right Panel**: Map + location list
  - MapView with search, filters, map, list

## Performance Improvements

1. **UI Thread**: No longer blocked by inference/camera
2. **Latest-Frame-Wins**: Queue size 1, drops old frames automatically
3. **Throttling**: UI 20-30 FPS, inference 5-15 FPS
4. **Pre-allocated Buffers**: Avoids repeated RGB conversion
5. **Separate Overlay**: No numpy drawing, all QPainter

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run application:

   ```bash
   python app.py
   ```

3. The new main window (`main_window_new.py`) is used by default in `app.py`

## Migration Notes

- Old `ui/main_window.py` is kept for reference
- Old `ui/workers.py` is kept for reference
- New workers are in `workers/` directory
- New domain models in `domain/` directory
- Map widget now uses QWebEngine + Leaflet instead of simple QPainter

## Next Steps (Optional Enhancements)

1. Add GPS location detection
2. Implement search filtering in map view
3. Add routing engine integration (OSRM)
4. Add more location sources (API integration)
5. Add heat map visualization
6. Add layer toggles (Bank Sampah, TPS 3R, Recycler)
