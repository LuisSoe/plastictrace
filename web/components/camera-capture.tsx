"use client";

import { useEffect, useRef } from "react";
import { Camera } from "@phosphor-icons/react";

type CameraCaptureProps = {
  stream: MediaStream | null;
  onCapture: (imageDataUrl: string) => void;
  onCancel: () => void;
};

export function CameraCapture({
  stream,
  onCapture,
  onCancel,
}: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;

    video.srcObject = stream;
    video.onloadedmetadata = () => {
      void video.play().catch(() => {});
    };

    return () => {
      video.srcObject = null;
    };
  }, [stream]);

  const handleCapture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    onCapture(canvas.toDataURL("image/jpeg"));
  };

  return (
    <div className="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)]">
      <div className="relative bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="block max-h-[min(60dvh,480px)] w-full object-contain"
        />
        <div
          className="pointer-events-none absolute inset-0 border-4 border-[var(--accent)]"
          aria-hidden
        >
          <div className="absolute top-1/2 left-1/2 h-48 w-48 -translate-x-1/2 -translate-y-1/2 rounded-lg border-2 border-dashed border-white/70" />
        </div>
      </div>

      <div className="flex flex-wrap justify-center gap-3 border-t border-[var(--border)] p-4">
        <button
          type="button"
          onClick={handleCapture}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-[var(--accent)] px-6 py-2.5 text-sm font-semibold text-[var(--accent-ink)] transition-colors duration-200 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
        >
          <Camera size={18} weight="bold" />
          Ambil Foto
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="cursor-pointer rounded-lg border border-[var(--border)] bg-[var(--surface)] px-6 py-2.5 text-sm font-semibold text-[var(--ink)] transition-colors duration-200 hover:bg-[var(--accent-subtle)] active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
        >
          Batal
        </button>
      </div>

      <canvas ref={canvasRef} className="hidden" aria-hidden />
    </div>
  );
}
