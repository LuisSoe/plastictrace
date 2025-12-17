# PlastiTrace

Aplikasi desktop Python untuk deteksi jenis plastik secara realtime dari webcam menggunakan PyTorch dan OpenCV dengan bbox tracking stabil, trust & stability layer, enhanced UX, feedback system, dan location-aware recycling guidance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Phases Overview](#phases-overview)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Installation

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note for Python 3.14+ users:** If you encounter numpy build errors (missing `g++-14`), numpy 2.3.5+ is already installed and compatible. You may see dependency warnings from opencv-contrib-python, but they can be safely ignored. All packages work correctly with numpy 2.3.5+.

## Usage

### Desktop Application (CLI - Original)

Run the desktop application with camera loop (command line interface):

```bash
python app.py
```

**Note:** Make sure the virtual environment is activated before running the application.

Tekan **ESC** untuk keluar dari aplikasi.

### Desktop Application (GUI - Recommended)

Run the desktop application with PyQt5 GUI (realtime detection with modern interface):

```bash
python app_gui.py
```

**Features:**

- 🎨 Modern GUI dengan PyQt5
- 📹 Realtime video preview
- 🤖 Live plastic classification
- ♻️ Recycling recommendations
- 📊 FPS counter
- 🎯 Bounding box visualization
- 🔒 Trust & stability layer
- 📍 Location-aware guidance

Tekan **ESC** atau tombol "Exit" untuk menutup aplikasi.

### Web Application

Run the web-based UI with API backend:

1. **Start the Flask API backend:**

```bash
python api.py
```

The API will run on `http://localhost:5001` (port 5001 karena 5000 conflict dengan AirPlay di macOS)

2. **Open the web interface:**

Open `web/index.html` in your browser, or serve it with:

```bash
cd web
python -m http.server 8000
```

Then visit `http://localhost:8000/index.html`

**Web Features:**

- 📸 Upload images or use webcam for classification
- 🤖 AI classification with confidence scores
- ♻️ Recycling recommendations
- 🎨 Modern, responsive UI (Bahasa Indonesia)
- 📱 Mobile-friendly

**Note:** Web app menggunakan "capture photo" mode, bukan realtime detection. Untuk realtime detection, gunakan Desktop GUI (app_gui.py).

## Features

### Core Features

- **Realtime bbox detection**: Deteksi objek plastik menggunakan OpenCV contours (tanpa YOLO)
- **Stable tracking**: CSRT tracker untuk bbox yang stabil dan smooth
- **ROI classification**: Crop ROI dari bbox → klasifikasi dengan ResNet18
- **Inference throttling**: Inference hanya setiap N frames (default: setiap 3 frames)
- **Smooth overlay**: Bbox, label, confidence, dan rekomendasi ditampilkan realtime
- **FPS counter**: Monitor performa aplikasi
- **CPU-friendly**: Optimized untuk CPU, CUDA opsional

### Phase 1: Trust & Stability Layer

- **Frame Quality Assessment**: Blur and brightness detection
- **Temporal Aggregation**: Anti-flicker with rolling window and EMA
- **Decision Engine**: Lock/unlock logic with stability gating
- **State Management**: SCANNING → UNSTABLE → LOCKED → UNKNOWN transitions
- **Stability Metrics**: Vote ratio, margin, entropy tracking

### Phase 2: Enhanced UX

- **UI State Machine**: SCAN_MODE and REVIEW_MODE
- **Overlay Guidance**: Alignment box, stability indicator, quality warnings
- **Top-3 Predictions**: Collapsible panel showing top predictions
- **Frame Buffer**: Best frame selection for capture
- **Review Mode**: Frozen frame with result card
- **History Management**: Local storage of scan results

### Phase 3: Feedback & Data Flywheel

- **Feedback Actions**: Confirm, Correct, Unsure buttons
- **Condition Flags**: Clean, label_present, crushed, mixed toggles
- **Structured Logging**: Versioned ScanRecord schema
- **Priority Scoring**: Active learning prioritization
- **Dataset Export**: Multiple export modes (all, high-value, corrected)
- **Evaluation Metrics**: Confusion matrix, precision/recall, calibration
- **History Screen**: Browse and view scan records

### Phase 4: Location-Aware Guidance

- **Rules Engine**: Region-based recycling rules
- **Drop-off Locations**: Local database with seed data
- **Location Filtering**: Type-based and condition-based filtering
- **Location Ranking**: Distance, source, condition factors
- **Action Recommendations**: Recyclable status, instructions, warnings
- **Map Integration**: Open in external maps apps
- **Event Logging**: Track map interactions

## Architecture

### Desktop App (CLI & GUI)

```
Webcam → Bbox Detection (contours) → CSRT Tracker → ROI Crop →
ResNet18 Classifier (throttled) → Trust Layer (Phase 1) →
Decision Engine → UI State Machine (Phase 2) →
Feedback System (Phase 3) → Location Guidance (Phase 4) →
Overlay/Display (bbox + label + recommendation + actions)
```

### Web App

```
Browser (Upload/Camera) → Flask API → ResNet18 Classifier → JSON Response →
Web UI Display (Label + Confidence + Recommendations)
```

### Key Components

#### Desktop App

- **app.py**: Original CLI-based realtime detection (OpenCV display)
- **app_gui.py**: PyQt5 GUI-based realtime detection with all phases
- **vision/bbox_detector.py**: Deteksi bbox menggunakan Canny edges + contours
- **vision/bbox_tracker.py**: CSRT/KCF tracker untuk stabilisasi bbox
- **vision/smoothing.py**: EMA smoothing untuk bbox dan confidence
- **ml/classifier.py**: ResNet18 classifier dengan FP32 enforcement
- **ui/camera_loop.py**: Main camera loop logic (used by app.py)

#### Trust Layer (Phase 1)

- **trust/frame_quality.py**: Frame quality assessment (blur, brightness)
- **trust/temporal_aggregator.py**: Rolling window and EMA aggregation
- **trust/decision_engine.py**: Lock/unlock decision logic
- **trust/config.py**: Configuration constants

#### Enhanced UX (Phase 2)

- **ui/ui_state.py**: UI state machine (SCAN_MODE/REVIEW_MODE)
- **ui/overlay_renderer.py**: Overlay rendering (alignment, stability, warnings)
- **ui/frame_buffer.py**: Recent frame buffer with best frame selection
- **ui/history.py**: History management
- **ml/action_guidance.py**: Action guidance per plastic type

#### Feedback System (Phase 3)

- **feedback/schema.py**: ScanRecord data schema
- **feedback/dataset_store.py**: SQLite storage for records
- **feedback/priority_scorer.py**: Active learning priority scoring
- **feedback/feedback_controller.py**: Feedback handling (confirm/correct/unsure)
- **feedback/dataset_exporter.py**: Dataset export pipeline
- **feedback/evaluation.py**: Model evaluation utilities

#### Location Services (Phase 4)

- **location/rules_engine.py**: Recycling rules engine
- **location/dropoff_schema.py**: Drop-off location schema
- **location/dropoff_store.py**: Location storage
- **location/location_filter.py**: Location filtering and ranking
- **location/region_manager.py**: Region selection and persistence
- **location/event_logger.py**: Map interaction event logging

#### Web App

- **api.py**: Flask REST API endpoint untuk image classification
- **web/index.html**: React-based web UI (single file, no build required)

## Project Structure

```
plastictrace/
├── app.py                     # Entry point (Desktop CLI)
├── app_gui.py                 # Entry point (Desktop GUI - PyQt5) ⭐
├── api.py                     # Flask API backend (Web App)
├── requirements.txt
├── models/
│   └── plastitrace.pth       # ResNet18 model
├── ml/
│   ├── config.py             # Constants & recommendations
│   ├── classifier.py        # PyTorch ResNet18 classifier
│   ├── preprocess.py         # Image preprocessing
│   └── action_guidance.py    # Action guidance (Phase 2)
├── vision/
│   ├── bbox_detector.py      # Contour-based bbox detection
│   ├── bbox_tracker.py       # CSRT/KCF tracker wrapper
│   └── smoothing.py          # EMA smoothing
├── trust/                     # Phase 1: Trust & Stability
│   ├── __init__.py
│   ├── config.py
│   ├── frame_quality.py
│   ├── temporal_aggregator.py
│   └── decision_engine.py
├── ui/                        # Phase 2: Enhanced UX
│   ├── __init__.py
│   ├── camera_loop.py        # Camera loop logic (for app.py)
│   ├── ui_state.py           # UI state machine
│   ├── overlay_renderer.py   # Overlay rendering
│   ├── frame_buffer.py       # Frame buffer
│   └── history.py            # History management
├── feedback/                  # Phase 3: Feedback & Data Flywheel
│   ├── __init__.py
│   ├── schema.py             # ScanRecord schema
│   ├── dataset_store.py      # SQLite storage
│   ├── priority_scorer.py    # Priority scoring
│   ├── feedback_controller.py # Feedback handling
│   ├── dataset_exporter.py   # Dataset export
│   └── evaluation.py          # Evaluation utilities
├── location/                  # Phase 4: Location-Aware Guidance
│   ├── __init__.py
│   ├── rules_engine.py       # Rules engine
│   ├── dropoff_schema.py     # Location schema
│   ├── dropoff_store.py      # Location storage
│   ├── location_filter.py    # Filter & ranker
│   ├── region_manager.py     # Region management
│   └── event_logger.py        # Event logging
├── rules/                     # Phase 4: Rulesets
│   └── default.json           # Default ruleset (auto-created)
├── data/                      # Runtime data
│   ├── records.db             # Phase 3: Scan records
│   ├── dropoff_locations.db   # Phase 4: Location database
│   ├── map_events.db          # Phase 4: Event log
│   ├── dropoff_seed.json      # Phase 4: Seed locations
│   ├── region_config.json     # Phase 4: Region config
│   └── images/                # Phase 3: Saved images
│       ├── original/
│       ├── snapshot/
│       └── roi/
├── history/                   # Phase 2: History (legacy)
│   ├── history.json
│   └── snapshot_*.jpg
├── utils/
│   └── softmax.py            # Softmax utility
└── web/
    └── index.html            # Web UI (React + Tailwind)
```

## Phases Overview

### Phase 1: Trust & Stability Layer

Eliminates label flicker and provides stable "Locked" results.

**Key Features:**

- Frame quality gates (blur, brightness)
- Temporal aggregation (rolling window + EMA)
- Decision engine with lock/unlock logic
- Stability metrics and state management

**Files:** `trust/` directory

### Phase 2: Enhanced UX

Improves real-time scanning UX with guidance overlays and review mode.

**Key Features:**

- Alignment box and quality hints
- Stability indicators
- Top-3 predictions panel
- Review mode with result card
- Frame buffer for best frame selection

**Files:** `ui/` directory (except `camera_loop.py`)

### Phase 3: Feedback & Data Flywheel

Turns real-world usage into structured training data.

**Key Features:**

- Confirm/Correct/Unsure feedback
- Structured logging with versioned schema
- Priority scoring for active learning
- Dataset export (multiple modes)
- Evaluation metrics tracking
- History screen

**Files:** `feedback/` directory

### Phase 4: Location-Aware Guidance

Provides location-aware recycling guidance and drop-off map integration.

**Key Features:**

- Rules engine with region-based rulesets
- Drop-off location database
- Location filtering and ranking
- Action recommendations
- Map integration (external apps)
- Event logging

**Files:** `location/` directory, `rules/` directory

## Configuration

### Desktop App Settings (CLI & GUI)

Default settings in `app.py` and `app_gui.py`:

- `min_area=2000`: Minimum bbox area untuk detection
- `inference_interval=3`: Run inference setiap 3 frames
- `redetect_interval=30`: Re-detect bbox setiap 30 frames (untuk koreksi drift)

### Trust Layer Settings

Edit `trust/config.py`:

- `BLUR_MIN = 80.0`: Blur threshold
- `BRIGHTNESS_MIN = 40.0`: Brightness threshold
- `N = 20`: Rolling window size
- `LOCK_MIN_CONF = 0.70`: Minimum confidence to lock
- `LOCK_MIN_MARGIN = 0.15`: Minimum margin to lock

### Priority Scorer Settings

Edit `feedback/priority_scorer.py`:

- Weights for priority calculation (w1-w5)
- High-value threshold (default: 0.6)

### Rules Engine Settings

Edit `rules/default.json` or create region-specific rules:

- Per-plastic-type rules
- Condition overrides
- Eligible drop-off tags

### Web App Settings

Default settings in `api.py`:

- Host: `0.0.0.0`
- Port: `5001` (changed from 5000 to avoid AirPlay conflict on macOS)
- Debug: `True` (set to `False` for production)
- CORS: Enabled for all origins (restrict in production)

**Web UI (`index.html`):**

- API URL: `http://localhost:5001/api/classify`

## API Endpoints

### POST /api/classify

Classify a plastic item from an uploaded image.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**

```json
{
  "label": "PET",
  "confidence": 0.95
}
```

### GET /api/health

Check if the API is running.

**Response:**

```json
{
  "status": "ok"
}
```

## Model

The application uses a ResNet18 model trained for 4-class plastic classification. The model file should be located at `models/plastitrace.pth`.

**Classes:** HDPE, PET, PP, PS

## Recommendations

Aplikasi menampilkan rekomendasi daur ulang dalam Bahasa Indonesia:

- **HDPE**: Umumnya bisa didaur ulang. Bilas dan masukkan ke sampah daur ulang plastik keras.
- **PET**: Botol minum plastik. Bilas, lepas label bila memungkinkan, buang ke sampah daur ulang.
- **PP**: Wadah makanan/kantong tertentu. Bila bersih, daur ulang; jika tidak ada fasilitas, buang sebagai residu.
- **PS**: Styrofoam/foam. Sulit didaur ulang; hindari pembakaran, buang ke sampah residu.

## Troubleshooting

### Desktop App (CLI & GUI)

**Error: "No OpenCV tracker available"**

- Install opencv-contrib-python: `pip install opencv-contrib-python`

**Error: "No module named 'PyQt5'"** (GUI only)

- Install PyQt5: `pip install PyQt5`

**Low FPS or slow inference**

- Reduce `inference_interval` in `app.py` or `app_gui.py`
- Use smaller input images
- Enable CUDA if available

**Camera not detected**

- Check if camera is already in use by another application
- Try changing `camera_index` from 0 to 1 in the code
- Check camera permissions in System Preferences (macOS)

**Trust layer not locking**

- Check frame quality (blur, brightness)
- Verify stability threshold settings
- Ensure consistent predictions (20+ frames)

### Web App

**Error: "Port 5001 is in use"**

- Change port in `api.py`: `app.run(host='0.0.0.0', port=5002, debug=True)`
- Update API URL in `web/index.html` to match new port

**Error: "Cannot access camera"**

- Grant browser permission to access camera
- Use HTTPS (required for webcam on some browsers)
- Check if camera is already in use

**Error: "Failed to classify image"**

- Make sure Flask API is running on `http://localhost:5001`
- Check API logs for errors
- Verify model file exists at `models/plastitrace.pth`

**CORS errors**

- Ensure flask-cors is installed: `pip install flask-cors`
- Check browser console for specific CORS errors

**macOS AirPlay Receiver conflict (Port 5000)**

- Disable AirPlay Receiver in System Settings → General → AirDrop & Handoff
- OR use port 5001 (already configured in api.py)

### Phase-Specific Issues

**Phase 1: Trust layer issues**

- Check `trust/config.py` for threshold settings
- Verify frame quality metrics are reasonable
- Check decision engine state transitions

**Phase 2: UI issues**

- Verify PyQt5 is installed
- Check UI update throttling (15 FPS)
- Ensure frame buffer is working

**Phase 3: Database issues**

- Check `data/records.db` exists and is writable
- Verify SQLite is working
- Check image storage permissions

**Phase 4: Location issues**

- Verify seed data loaded (`data/dropoff_seed.json`)
- Check rules file exists (`rules/default.json`)
- Verify region config is set

## Development

### Running Tests

```bash
# Phase 1 tests
python test_trust.py

# Phase 2-4 tests (if implemented)
python test_phase2.py
python test_phase3.py
python test_phase4.py
```

### Exporting Dataset (Phase 3)

From GUI: Click "📤 Export Dataset" button

Or programmatically:

```python
from feedback.dataset_exporter import DatasetExporter
from feedback.dataset_store import DatasetStore

store = DatasetStore()
exporter = DatasetExporter(store)
exporter.export("exports/my_dataset", mode="high_value")
```

### Running Evaluation (Phase 3)

```python
from feedback.evaluation import ModelEvaluator
from feedback.dataset_store import DatasetStore

store = DatasetStore()
evaluator = ModelEvaluator(store)
metrics = evaluator.compute_metrics(model_version="1.0.0")
evaluator.save_metrics("metrics.json", model_version="1.0.0")
```

### Adding Custom Rules (Phase 4)

Create `rules/{country}-{province}-{city}.json`:

```json
{
  "region": {
    "country": "ID",
    "province": "Jawa Barat",
    "city": "Bandung"
  },
  "rules": {
    "PET": {
      "recyclable": true,
      "base_instructions": ["Rinse", "Remove cap"],
      "eligible_dropoff_tags": ["PET", "BOTTLES"]
    }
  }
}
```

## Application Comparison

| Feature            | Desktop CLI         | Desktop GUI         | Web App              |
| ------------------ | ------------------- | ------------------- | -------------------- |
| Realtime Detection | ✅ Yes              | ✅ Yes              | ❌ No (capture only) |
| Modern UI          | ❌ No               | ✅ Yes              | ✅ Yes               |
| Trust Layer        | ✅ Yes              | ✅ Yes              | ❌ No                |
| Enhanced UX        | ❌ No               | ✅ Yes              | ❌ No                |
| Feedback System    | ❌ No               | ✅ Yes              | ❌ No                |
| Location Guidance  | ❌ No               | ✅ Yes              | ❌ No                |
| Installation       | Easy                | Easy                | Medium               |
| Performance        | Excellent           | Excellent           | Good                 |
| Mobile Support     | ❌ No               | ❌ No               | ✅ Yes               |
| Best For           | Development/Testing | End Users (Desktop) | Web Access/Mobile    |

**Recommendation:**

- **Development/Debugging**: Use `app.py` (CLI)
- **Desktop Users**: Use `app_gui.py` (PyQt5 GUI) - **RECOMMENDED**
- **Web/Mobile Access**: Use Web App (`api.py` + `web/index.html`)

## Production Deployment

### Desktop App

Package with PyInstaller for distribution:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed app_gui.py
```

### Web App

For production deployment:

1. **Set Flask to production mode:**
   - Change `debug=True` to `debug=False` in `api.py`
   - Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 api:app
```

2. **Configure CORS properly:**

   - Restrict allowed origins in `api.py`
   - Don't use `CORS(app)` with no parameters in production

3. **Serve static files:**

   - Use Nginx or Apache to serve `web/index.html`
   - Configure reverse proxy to Flask API

4. **Security considerations:**
   - Add rate limiting
   - Implement file size limits for uploads
   - Validate file types
   - Use HTTPS

## Changelog

### v4.0 (Latest)

- ✨ Phase 4: Location-aware recycling guidance
- 📍 Drop-off location database and filtering
- 🗺️ Map integration with external apps
- 📋 Rules engine with region-based rulesets
- 📊 Event logging for map interactions

### v3.0

- ✨ Phase 3: Feedback & data flywheel
- ✅ Confirm/Correct/Unsure feedback system
- 📊 Priority scoring for active learning
- 📤 Dataset export pipeline
- 📈 Evaluation metrics tracking
- 📋 History screen with record details

### v2.0

- ✨ Phase 2: Enhanced UX
- 🎯 Alignment box and quality hints
- 📊 Stability indicators
- 📸 Review mode with result card
- 📋 Top-3 predictions panel
- 💾 Frame buffer for best frame selection

### v1.0

- ✨ Phase 1: Trust & stability layer
- 🔒 Decision engine with lock/unlock logic
- 📊 Temporal aggregation (anti-flicker)
- 🎯 Frame quality assessment
- 📈 Stability metrics

### v0.9

- 🎯 Initial release with CLI desktop app
- 🤖 ResNet18 classification
- 📹 CSRT bbox tracking
- 🌐 Web interface with React UI
- 🇮🇩 Full Bahasa Indonesia support

## License

[Your License Here]

## Contributors

[Your Team/Contributors Here]
