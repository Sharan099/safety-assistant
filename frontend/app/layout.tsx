import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Safety Assistant",
  description: "Versioned, cited answers over automotive passive-safety regulations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-zinc-950 font-sans text-zinc-100 antialiased">{children}</body>
    </html>
  );
}
