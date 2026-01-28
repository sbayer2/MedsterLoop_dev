import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for Docker deployment
  output: 'standalone',
};

export default nextConfig;
