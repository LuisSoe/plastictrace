import { RECOMMENDATIONS } from "./recommendations";

export type ClassificationResult = {
  label: string;
  confidence: number;
  recommendation: string;
};

function getApiBaseUrl(): string {
  return (
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    "http://localhost:5001"
  );
}

export function getApiUrl(): string {
  return getApiBaseUrl();
}

export async function classifyImage(
  imageDataUrl: string,
): Promise<ClassificationResult> {
  const blob = await fetch(imageDataUrl).then((r) => r.blob());
  const formData = new FormData();
  formData.append("image", blob, "image.jpg");

  const response = await fetch(`${getApiBaseUrl()}/api/classify`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Classification failed");
  }

  const result = (await response.json()) as {
    label: string;
    confidence: number;
    error?: string;
  };

  if (result.error) {
    throw new Error(result.error);
  }

  return {
    label: result.label,
    confidence: result.confidence,
    recommendation:
      RECOMMENDATIONS[result.label] ?? "Tidak ada rekomendasi tersedia.",
  };
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/api/health`, {
      method: "GET",
      cache: "no-store",
    });
    return response.ok;
  } catch {
    return false;
  }
}
