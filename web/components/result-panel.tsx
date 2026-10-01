"use client";

import {
  ArrowCounterClockwise,
  CheckCircle,
  Recycle,
  Warning,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import type { ClassificationResult } from "@/lib/api";
import {
  getRecyclingTier,
  RECYCLING_STATUS,
} from "@/lib/recommendations";

type ResultPanelProps = {
  image: string;
  prediction: ClassificationResult | null;
  isProcessing: boolean;
  onReset: () => void;
};

function ResultSkeleton() {
  return (
    <div className="space-y-5 p-5">
      <div className="skeleton mx-auto h-10 w-24" />
      <div className="skeleton mx-auto h-6 w-32" />
      <div className="skeleton h-16 w-full" />
      <div className="skeleton h-20 w-full" />
    </div>
  );
}

function StatusIcon({ tier }: { tier: ReturnType<typeof getRecyclingTier> }) {
  if (tier === "recyclable") {
    return (
      <CheckCircle
        size={22}
        weight="regular"
        className="mt-0.5 shrink-0 text-[var(--success)]"
      />
    );
  }
  if (tier === "difficult") {
    return (
      <WarningCircle
        size={22}
        weight="regular"
        className="mt-0.5 shrink-0 text-[var(--danger)]"
      />
    );
  }
  return (
    <Warning
      size={22}
      weight="regular"
      className="mt-0.5 shrink-0 text-[var(--warning)]"
    />
  );
}

function statusBg(tier: ReturnType<typeof getRecyclingTier>) {
  if (tier === "recyclable") return "bg-[var(--success-bg)]";
  if (tier === "difficult") return "bg-[var(--danger-bg)]";
  return "bg-[var(--warning-bg)]";
}

export function ResultPanel({
  image,
  prediction,
  isProcessing,
  onReset,
}: ResultPanelProps) {
  const tier = prediction ? getRecyclingTier(prediction.label) : null;
  const status = tier ? RECYCLING_STATUS[tier] : null;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)]">
        <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
          <h2 className="text-sm font-semibold text-[var(--ink)]">
            Gambar
          </h2>
          <button
            type="button"
            onClick={onReset}
            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg text-[var(--muted)] transition-colors duration-200 hover:bg-[var(--accent-subtle)] hover:text-[var(--ink)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
            aria-label="Tutup dan mulai lagi"
          >
            <X size={18} weight="regular" />
          </button>
        </div>
        <div className="p-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={image}
            alt="Plastik yang dianalisis"
            className="w-full rounded-lg object-contain"
          />
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)]">
        <div className="border-b border-[var(--border)] px-4 py-3">
          <h2 className="text-sm font-semibold text-[var(--ink)]">
            Hasil klasifikasi
          </h2>
        </div>

        {isProcessing && <ResultSkeleton />}

        {prediction && !isProcessing && (
          <div className="result-enter space-y-5 p-5">
            <div className="text-center">
              <p className="text-4xl font-bold tracking-tight text-[var(--ink)]">
                {prediction.label}
              </p>
              <p className="mt-2 text-lg font-medium text-[var(--muted)]">
                {(prediction.confidence * 100).toFixed(1)}% keyakinan
              </p>
              <div className="mx-auto mt-3 h-0.5 w-16 bg-[var(--accent)]" />
            </div>

            {status && tier && (
              <div
                className={`flex gap-3 rounded-lg p-4 ${statusBg(tier)}`}
              >
                <StatusIcon tier={tier} />
                <div>
                  <p className="text-sm font-semibold text-[var(--ink)]">
                    {status.title}
                  </p>
                  <p className="mt-1 text-sm text-[var(--muted)]">
                    {status.description}
                  </p>
                </div>
              </div>
            )}

            <div className="rounded-lg border border-[var(--border)] p-4">
              <div className="mb-2 flex items-center gap-2">
                <Recycle
                  size={18}
                  weight="regular"
                  className="text-[var(--accent)]"
                />
                <h3 className="text-sm font-semibold text-[var(--ink)]">
                  Rekomendasi
                </h3>
              </div>
              <p className="text-sm leading-relaxed text-[var(--muted)]">
                {prediction.recommendation}
              </p>
            </div>

            <button
              type="button"
              onClick={onReset}
              className="inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg bg-[var(--accent)] py-3 text-sm font-semibold text-[var(--accent-ink)] transition-colors duration-200 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
            >
              <ArrowCounterClockwise size={18} weight="bold" />
              Analisis lagi
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
