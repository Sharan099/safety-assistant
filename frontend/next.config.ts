import type { NextConfig } from "next";

// Same-origin API: the browser talks to /api/* and /health/* on this host; Next proxies to the
// backend so the HttpOnly session cookie is first-party. In production put both behind one origin.
const API = process.env.API_INTERNAL_URL ?? "http://localhost:8010";

const nextConfig: NextConfig = {
  // An answer can legitimately take up to the agent budget (45 s) plus one LLM call (30 s); the
  // default 30 s proxy timeout turned slow-gateway answers into 500s.
  experimental: { proxyTimeout: 120_000 },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/health/:path*", destination: `${API}/health/:path*` },
    ];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Content-Security-Policy",
            value:
              "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
