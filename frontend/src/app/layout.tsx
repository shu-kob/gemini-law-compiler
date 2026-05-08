import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Velo-Verify-Gemini",
  description:
    "自転車青切符のハイブリッド判定 — 2008 年式決定論的パース × 2026 年式 LLM",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body className="bg-slate-50 text-slate-900 antialiased">{children}</body>
    </html>
  );
}
