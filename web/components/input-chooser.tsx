"use client";

import { Camera, UploadSimple } from "@phosphor-icons/react";

export type InputMode = "camera" | "upload";

type InputChooserProps = {
  mode: InputMode;
  onSelectCamera: () => void;
  onSelectUpload: () => void;
};

export function InputChooser({
  mode,
  onSelectCamera,
  onSelectUpload,
}: InputChooserProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <button
          type="button"
          onClick={onSelectCamera}
          className={`flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 px-6 py-8 transition-colors duration-200 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${
            mode === "camera"
              ? "border-[var(--accent)] bg-[var(--accent-subtle)]"
              : "border-[var(--border)] bg-[var(--surface)] hover:border-[var(--accent)]"
          }`}
        >
          <Camera
            size={32}
            weight="regular"
            className={
              mode === "camera" ? "text-[var(--accent)]" : "text-[var(--muted)]"
            }
          />
          <div className="text-center">
            <p className="text-base font-semibold text-[var(--ink)]">Kamera</p>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Ambil foto plastik langsung
            </p>
          </div>
        </button>

        <button
          type="button"
          onClick={onSelectUpload}
          className={`flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 px-6 py-8 transition-colors duration-200 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${
            mode === "upload"
              ? "border-[var(--accent)] bg-[var(--accent-subtle)]"
              : "border-[var(--border)] bg-[var(--surface)] hover:border-[var(--accent)]"
          }`}
        >
          <UploadSimple
            size={32}
            weight="regular"
            className={
              mode === "upload" ? "text-[var(--accent)]" : "text-[var(--muted)]"
            }
          />
          <div className="text-center">
            <p className="text-base font-semibold text-[var(--ink)]">Upload</p>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Pilih gambar dari galeri
            </p>
          </div>
        </button>
      </div>

      <p className="text-center text-sm text-[var(--muted)]">
        Foto atau upload gambar plastik, lalu dapatkan jenis dan panduan daur
        ulang.
      </p>
    </div>
  );
}
