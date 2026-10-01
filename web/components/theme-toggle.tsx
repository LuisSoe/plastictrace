"use client";

import { Moon, Sun } from "@phosphor-icons/react";
import { useTheme } from "./theme-provider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--muted)] transition-colors duration-200 hover:text-[var(--ink)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] active:scale-[0.98]"
      aria-label={theme === "light" ? "Aktifkan mode gelap" : "Aktifkan mode terang"}
    >
      {theme === "light" ? (
        <Moon size={18} weight="regular" />
      ) : (
        <Sun size={18} weight="regular" />
      )}
    </button>
  );
}
