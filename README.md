# PlastiTrace Desktop

Aplikasi desktop Python untuk deteksi jenis plastik secara realtime dari webcam menggunakan PyTorch dan OpenCV dengan bbox tracking stabil.

## Installation

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv
#ensure u use python 3.14 to avoid error
#and make sure u have the c++ or c compiler installed for numpy

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

## Usage

### Desktop Application (CLI - Original)

Run the desktop application with camera loop (command line interface):

```bash
python app.py
```

**Note:** Make sure the virtual environment is activated before running the application.

Tekan **ESC** untuk keluar dari aplikasi.

### Desktop Application (GUI - NEW)

Run the desktop application with PyQt5 GUI (realtime detection with modern interface):

```bash
python app_gui.py
```

**Features:**

- 🎨 Modern GUI dengan PyQt5
- 📹 Realtime video preview
- 🤖 Live plastic classification
- ♻️ Recycling recommendations in Indonesian
- 📊 FPS counter
- 🎯 Bounding box visualization

Tekan **ESC** atau tombol "Keluar" untuk menutup aplikasi.

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

### Desktop App (CLI) Features

- **Realtime bbox detection**: Deteksi objek plastik menggunakan OpenCV contours (tanpa YOLO)
- **Stable tracking**: CSRT tracker untuk bbox yang stabil dan smooth
- **ROI classification**: Crop ROI dari bbox → klasifikasi dengan ResNet18
- **Inference throttling**: Inference hanya setiap N frames (default: setiap 3 frames)
- **Smooth overlay**: Bbox, label, confidence, dan rekomendasi ditampilkan realtime
- **FPS counter**: Monitor performa aplikasi
- **CPU-friendly**: Optimized untuk CPU, CUDA opsional
- **Stabil**: Didesain untuk running >10 menit tanpa masalah

### Desktop App (GUI) Features

- **Modern PyQt5 Interface**: GUI yang clean dan professional dengan 3-panel layout
- **Realtime Video Preview**: Live camera feed dengan bbox overlay
- **Live Classification**: Deteksi dan klasifikasi plastik secara realtime
- **Information Panel**: Tampilan hasil, confidence score, dan rekomendasi
- **Interactive Map**: Peta interaktif dengan 200+ lokasi Bank Sampah di seluruh Indonesia
- **Location Search**: Cari dan filter lokasi berdasarkan jenis plastik
- **Distance Calculation**: Menampilkan jarak dari lokasi pengguna ke setiap lokasi
- **Navigation**: Tombol navigasi langsung ke Google Maps
- **FPS Monitor**: Real-time FPS counter
- **Keyboard Shortcuts**: ESC untuk keluar
- **Smooth Animations**: Transisi yang halus dan responsive

### Web App Features

- **Image Upload**: Upload gambar plastik untuk klasifikasi
- **Webcam Capture**: Ambil foto langsung dari webcam
- **AI Classification**: Identifikasi jenis plastik dengan confidence score
- **Recycling Guide**: Rekomendasi daur ulang untuk setiap jenis plastik
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Processing**: Fast classification with visual feedback

## Architecture

### Desktop App (CLI & GUI)

```
Webcam → Bbox Detection (contours) → CSRT Tracker → ROI Crop →
ResNet18 Classifier (throttled) → Overlay/Display (bbox + label + recommendation)
```

### Web App

```
Browser (Upload/Camera) → Flask API → ResNet18 Classifier → JSON Response →
Web UI Display (Label + Confidence + Recommendations)
```

### Key Components

#### Desktop App

- **app.py**: Original CLI-based realtime detection (OpenCV display)
- **app_gui.py**: NEW - PyQt5 GUI-based realtime detection
- **ui/main_window_new.py**: Main window with 3-panel layout (detection, video, map)
- **ui/map_view.py**: Interactive map with QWebEngine + Leaflet showing 200+ locations
- **location/excel_loader.py**: Load locations from Excel (data_sipsn.xlsx) with geocoding support
- **location/geocode_all.py**: Batch geocoding script to create location cache
- **domain/geo.py**: Geographic utilities (Haversine distance, location filtering)
- **vision/bbox_detector.py**: Deteksi bbox menggunakan Canny edges + contours
- **vision/bbox_tracker.py**: CSRT/KCF tracker untuk stabilisasi bbox
- **vision/smoothing.py**: EMA smoothing untuk bbox dan confidence
- **ml/classifier.py**: ResNet18 classifier dengan FP32 enforcement
- **ui/camera_loop.py**: Main camera loop logic (used by app.py)

#### Web App

- **api.py**: Flask REST API endpoint untuk image classification
- **web/index.html**: React-based web UI (single file, no build required)

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

## Project Structure

```
plastitrace/
├── app.py                     # Entry point (Desktop CLI)
├── app_gui.py                 # Entry point (Desktop GUI - PyQt5)
├── api.py                     # Flask API backend (Web App)
├── requirements.txt
├── data_sipsn.xlsx           # Source Excel file with Bank Sampah locations
├── models/
│   └── plastitrace.pth       # ResNet18 model
├── data/
│   ├── locations_geocoded.json  # Cached geocoded locations (200 locations)
│   ├── dropoff_seed.json        # Seed location data
│   └── dropoff_locations.db     # SQLite database
├── ml/
│   ├── config.py             # Constants & recommendations
│   ├── classifier.py         # PyTorch ResNet18 classifier
│   └── preprocess.py         # Image preprocessing
├── location/
│   ├── excel_loader.py       # Load locations from Excel with geocoding
│   ├── geocode_all.py        # Batch geocoding script
│   ├── dropoff_store.py      # Location database store
│   └── dropoff_schema.py     # Location data schema
├── domain/
│   ├── models.py             # Detection & Location dataclasses
│   ├── geo.py                # Geographic utilities (distance, filtering)
│   └── filtering.py          # Temporal smoothing
├── vision/
│   ├── bbox_detector.py      # Contour-based bbox detection
│   ├── bbox_tracker.py       # CSRT/KCF tracker wrapper
│   └── smoothing.py          # EMA smoothing
├── ui/
│   ├── main_window_new.py    # Main window (3-panel layout)
│   ├── map_view.py           # Interactive map widget
│   ├── video_widget.py       # Video display widget
│   └── camera_loop.py        # Camera loop logic (for app.py)
├── workers/
│   ├── capture_worker.py     # Camera capture worker thread
│   └── inference_worker.py   # ML inference worker thread
├── assets/
│   └── map/
│       └── leaflet.html      # Leaflet map HTML
├── utils/
│   └── softmax.py            # Softmax utility
└── web/
    └── index.html            # Web UI (React + Tailwind)
```

## Configuration

### Desktop App Settings (CLI & GUI)

Default settings in `app.py` and `app_gui.py`:

- `min_area=2000`: Minimum bbox area untuk detection
- `inference_interval=3`: Run inference setiap 3 frames
- `redetect_interval=30`: Re-detect bbox setiap 30 frames (untuk koreksi drift)

### Web App Settings

Default settings in `api.py`:

- Host: `0.0.0.0`
- Port: `5001` (changed from 5000 to avoid AirPlay conflict on macOS)
- Debug: `True` (set to `False` for production)
- CORS: Enabled for all origins (restrict in production)

**Web UI (`index.html`):**

- API URL: `http://localhost:5001/api/classify`

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

## Application Comparison

| Feature            | Desktop CLI         | Desktop GUI         | Web App              |
| ------------------ | ------------------- | ------------------- | -------------------- |
| Realtime Detection | ✅ Yes              | ✅ Yes              | ❌ No (capture only) |
| Modern UI          | ❌ No               | ✅ Yes              | ✅ Yes               |
| Installation       | Easy                | Easy                | Medium               |
| Performance        | Excellent           | Excellent           | Good                 |
| Mobile Support     | ❌ No               | ❌ No               | ✅ Yes               |
| Best For           | Development/Testing | End Users (Desktop) | Web Access/Mobile    |

**Recommendation:**

- **Development/Debugging**: Use `app.py` (CLI)
- **Desktop Users**: Use `app_gui.py` (PyQt5 GUI) - **RECOMMENDED**
- **Web/Mobile Access**: Use Web App (`api.py` + `web/index.html`)

## Location Data & Geocoding

### Loading Locations

The application loads location data from Bank Sampah (waste bank) facilities across Indonesia:

1. **Primary Source**: `data_sipsn.xlsx` - Excel file with location data from SIPSN (Sistem Informasi Pengelolaan Sampah Nasional)

2. **Cache File**: `data/locations_geocoded.json` - Pre-geocoded locations with coordinates (200 locations)

3. **Geocoding**: Addresses from the "Alamat" field are geocoded to get accurate latitude/longitude coordinates

### Setting Up Location Data

**Option 1: Use Pre-geocoded Cache (Recommended)**

- The app automatically uses `data/locations_geocoded.json` if available
- Fast loading, no geocoding needed
- Already includes 200 locations with coordinates

**Option 2: Geocode from Excel**

- Run the geocoding script once to create the cache:

```bash
python location/geocode_all.py
```

- This will geocode all locations from `data_sipsn.xlsx` and save to `data/locations_geocoded.json`
- Takes ~3-4 minutes for 200 locations (1 second per location due to rate limiting)

### Map Features

- **200+ Locations**: Shows all Bank Sampah locations across Indonesia
- **Coverage**: Entire Indonesia (radius up to 4000km, covers Jakarta to Ambon and beyond)
- **Interactive Map**: Click markers to see details, navigate to Google Maps
- **Filtering**: Filter by plastic type (PET, HDPE, PP, PS)
- **Distance**: Shows distance from user location to each facility
- **Marker Clustering**: Groups nearby markers for better performance

## Production Deployment

### Desktop App

- Package with PyInstaller for distribution:

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

## License

[Your License Here]

## Contributors

[Your Team/Contributors Here]

## Changelog

### v2.1 (Latest)

- 🗺️ Added interactive map with 200+ Bank Sampah locations across Indonesia
- 📍 Geocoding support to convert addresses to coordinates
- 💾 Location cache system for fast loading
- 🎯 Shows all locations on map with distance calculation
- 🔍 Location filtering by plastic type
- 🧭 Navigation integration with Google Maps
- 🌏 Coverage for entire Indonesia (up to 4000km radius)

### v2.0

- ✨ Added PyQt5 GUI desktop application (`app_gui.py`)
- 🌐 Added web interface with React UI
- 🔧 Fixed port conflict with macOS AirPlay (port 5001)
- 🇮🇩 Full Bahasa Indonesia support in all interfaces
- 📝 Updated documentation

### v1.0

- 🎯 Initial release with CLI desktop app
- 🤖 ResNet18 classification
- 📹 CSRT bbox tracking
