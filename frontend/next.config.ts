import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // uncomment the following lines in case you want to ignore typescript and eslint errors
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  output: 'standalone',
  trailingSlash: false,
  images: {
    remotePatterns: [
      {
        hostname: '**',
      },
    ],
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
};

export default nextConfig;