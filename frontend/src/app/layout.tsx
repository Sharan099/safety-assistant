import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Passive Safety Assistant",
  description: "EU passive-safety regulation RAG for engineers",
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
