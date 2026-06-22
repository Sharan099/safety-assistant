import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PSA — Passive Safety Assistant",
  description:
    "Regulation-aware RAG for UN/ECE, FMVSS, Euro NCAP, and passive safety engineering",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
