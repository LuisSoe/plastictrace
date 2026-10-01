export const RECOMMENDATIONS: Record<string, string> = {
  HDPE: "Umumnya bisa didaur ulang. Bilas dan masukkan ke sampah daur ulang plastik keras.",
  PET: "Botol minum plastik. Bilas, lepas label bila memungkinkan, buang ke sampah daur ulang.",
  PP: "Wadah makanan/kantong tertentu. Bila bersih, daur ulang; jika tidak ada fasilitas, buang sebagai residu.",
  PS: "Styrofoam/foam. Sulit didaur ulang; hindari pembakaran, buang ke sampah residu.",
};

export type RecyclingTier = "recyclable" | "conditional" | "difficult";

export function getRecyclingTier(label: string): RecyclingTier {
  if (["PET", "HDPE", "PP"].includes(label)) return "recyclable";
  if (label === "PS") return "difficult";
  return "conditional";
}

export const RECYCLING_STATUS: Record<
  RecyclingTier,
  { title: string; description: string }
> = {
  recyclable: {
    title: "Bisa didaur ulang",
    description: "Plastik ini dapat didaur ulang di sebagian besar fasilitas.",
  },
  conditional: {
    title: "Tergantung fasilitas",
    description: "Periksa program daur ulang lokal Anda.",
  },
  difficult: {
    title: "Sulit didaur ulang",
    description: "Jarang diterima oleh fasilitas daur ulang.",
  },
};
