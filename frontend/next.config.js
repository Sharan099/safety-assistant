/** @type {import('next').NextConfig} */
const HF_BACKEND =
  process.env.NEXT_PUBLIC_HF_BACKEND_URL ||
  "https://sharan099-passive-safety-assistant.hf.space";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${HF_BACKEND}/api/v1/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
