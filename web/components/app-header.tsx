"use client";

import { Recycle } from "@phosphor-icons/react";
import { ThemeToggle } from "./theme-toggle";

export function AppHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-[var(--border)] bg-[var(--bg)]/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-3xl items-center justify-between gap-4 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <Recycle
            size={28}
            weight="regular"
            className="shrink-0 text-[var(--accent)]"
            aria-hidden
          />
          <div className="min-w-0">
            <h1 className="truncate text-xl font-bold tracking-tight text-[var(--ink)]">
              PlastiTrace
            </h1>
            <p className="truncate text-xs text-[var(--muted)]">
              Klasifikasi plastik berbasis AI
            </p>
          </div>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
