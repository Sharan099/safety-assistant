/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep webpack happy with pdfjs worker / canvas builds
  webpack: (config) => {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      canvas: false,
    };
    return config;
  },
};

module.exports = nextConfig;
