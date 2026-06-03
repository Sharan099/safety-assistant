import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PSA AI — Passive Safety Assistant",
  description: "Regulation-aware RAG for passive safety engineering",
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
