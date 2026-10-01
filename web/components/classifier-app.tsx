"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WarningCircle } from "@phosphor-icons/react";
import { AppHeader } from "./app-header";
import { CameraCapture } from "./camera-capture";
import { InputChooser, type InputMode } from "./input-chooser";
import { ResultPanel } from "./result-panel";
import {
  classifyImage,
  getApiUrl,
  type ClassificationResult,
} from "@/lib/api";

export function ClassifierApp() {
  const [mode, setMode] = useState<InputMode>("upload");
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<ClassificationResult | null>(
    null,
  );
  const [isProcessing, setIsProcessing] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const stopCamera = useCallback(() => {
    setStream((current) => {
      current?.getTracks().forEach((track) => track.stop());
      return null;
    });
    setIsCameraActive(false);
  }, []);

  const processImage = useCallback(async (imageData: string) => {
    setIsProcessing(true);
    setPrediction(null);
    setError(null);

    try {
      const result = await classifyImage(imageData);
      setPrediction(result);
    } catch {
      setError(
        `Gagal mengklasifikasi gambar. Pastikan API berjalan di ${getApiUrl()}`,
      );
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      setStream(mediaStream);
      setIsCameraActive(true);
      setError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Akses kamera ditolak";
      setError(`Tidak dapat mengakses kamera: ${message}`);
      setIsCameraActive(false);
    }
  }, []);

  const capturePhoto = useCallback(
    (imageDataUrl: string) => {
      setImage(imageDataUrl);
      stopCamera();
      void processImage(imageDataUrl);
    },
    [processImage, stopCamera],
  );

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("File harus berupa gambar.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result;
      if (typeof result === "string") {
        setImage(result);
        void processImage(result);
      }
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const handleSelectCamera = () => {
    setMode("camera");
    setError(null);
    if (!image) void startCamera();
  };

  const handleSelectUpload = () => {
    setMode("upload");
    stopCamera();
    setError(null);
    fileInputRef.current?.click();
  };

  const reset = () => {
    setImage(null);
    setPrediction(null);
    setIsProcessing(false);
    setError(null);
    stopCamera();
  };

  useEffect(() => {
    if (mode === "camera" && !isCameraActive && !image) {
      void startCamera();
    } else if (mode === "upload") {
      stopCamera();
    }
  }, [mode, isCameraActive, image, startCamera, stopCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="flex min-h-[100dvh] flex-col">
      <AppHeader />

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        {error && (
          <div
            className="mb-6 flex gap-3 rounded-xl border border-[var(--danger)]/30 bg-[var(--danger-bg)] p-4"
            role="alert"
          >
            <WarningCircle
              size={22}
              weight="regular"
              className="mt-0.5 shrink-0 text-[var(--danger)]"
            />
            <div className="min-w-0">
              <p className="text-sm font-semibold text-[var(--ink)]">Error</p>
              <p className="mt-1 text-sm text-[var(--muted)]">{error}</p>
            </div>
          </div>
        )}

        {!image && !(mode === "camera" && isCameraActive) && (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 sm:p-6">
            <h2 className="mb-4 text-lg font-semibold text-[var(--ink)]">
              Pilih metode input
            </h2>
            <InputChooser
              mode={mode}
              onSelectCamera={handleSelectCamera}
              onSelectUpload={handleSelectUpload}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
              aria-hidden
            />
          </div>
        )}

        {mode === "camera" && isCameraActive && !image && stream && (
          <CameraCapture
            stream={stream}
            onCapture={capturePhoto}
            onCancel={() => {
              stopCamera();
              setMode("upload");
            }}
          />
        )}

        {image && (
          <ResultPanel
            image={image}
            prediction={prediction}
            isProcessing={isProcessing}
            onReset={reset}
          />
        )}

      </main>
    </div>
  );
}
