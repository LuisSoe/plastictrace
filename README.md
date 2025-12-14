# PlastiTrace Desktop

Aplikasi desktop Python untuk deteksi jenis plastik secara realtime dari webcam menggunakan PyTorch dan OpenCV dengan bbox tracking stabil.

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

## Usage

Run the application:

```bash
python app.py
```

**Note:** Make sure the virtual environment is activated before running the application.

Tekan **ESC** untuk keluar dari aplikasi.

## Features

- **Realtime bbox detection**: Deteksi objek plastik menggunakan OpenCV contours (tanpa YOLO)
- **Stable tracking**: CSRT tracker untuk bbox yang stabil dan smooth
- **ROI classification**: Crop ROI dari bbox → klasifikasi dengan ResNet18
- **Inference throttling**: Inference hanya setiap N frames (default: setiap 3 frames)
- **Smooth overlay**: Bbox, label, confidence, dan rekomendasi ditampilkan realtime
- **FPS counter**: Monitor performa aplikasi
- **CPU-friendly**: Optimized untuk CPU, CUDA opsional
- **Stabil**: Didesain untuk running >10 menit tanpa masalah

## Architecture

```
Webcam → Bbox Detection (contours) → CSRT Tracker → ROI Crop →
ResNet18 Classifier (throttled) → Overlay (bbox + label + recommendation)
```

### Key Components

- **vision/bbox_detector.py**: Deteksi bbox menggunakan Canny edges + contours
- **vision/bbox_tracker.py**: CSRT/KCF tracker untuk stabilisasi bbox
- **vision/smoothing.py**: EMA smoothing untuk bbox dan confidence
- **ml/classifier.py**: ResNet18 classifier dengan FP32 enforcement
- **ui/camera_loop.py**: Main camera loop dengan tracking dan throttling

## Model

The application uses a ResNet18 model trained for 4-class plastic classification. The model file should be located at `models/plastitrace.pth`.

**Classes:** HDPE, PET, PP, PS

## Project Structure

```
plastitrace_desktop/
├── app.py                     # Entry point
├── requirements.txt
├── models/
│   └── plastitrace.pth
├── ml/
│   ├── config.py             # Constants & recommendations
│   ├── classifier.py         # PyTorch ResNet18 classifier
│   └── preprocess.py         # Image preprocessing
├── vision/
│   ├── bbox_detector.py      # Contour-based bbox detection
│   ├── bbox_tracker.py       # CSRT/KCF tracker wrapper
│   └── smoothing.py          # EMA smoothing (optional)
└── ui/
    └── camera_loop.py        # Main camera loop with tracking
```

## Configuration

Default settings in `app.py`:

- `min_area=2000`: Minimum bbox area untuk detection
- `inference_interval=3`: Run inference setiap 3 frames
- `redetect_interval=30`: Re-detect bbox setiap 30 frames (untuk koreksi drift)

## Recommendations

Aplikasi menampilkan rekomendasi daur ulang dalam Bahasa Indonesia:

- **HDPE**: Umumnya bisa didaur ulang. Bilas dan masukkan ke sampah daur ulang plastik keras.
- **PET**: Botol minum plastik. Bilas, lepas label bila memungkinkan, buang ke sampah daur ulang.
- **PP**: Wadah makanan/kantong tertentu. Bila bersih, daur ulang; jika tidak ada fasilitas, buang sebagai residu.
- **PS**: Styrofoam/foam. Sulit didaur ulang; hindari pembakaran, buang ke sampah residu.
