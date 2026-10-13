import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  build: { outDir: "dist", target: "es2022" },
  // local development: the inference server (python -m web.server.app --port 7861) answers /api
  server: { proxy: { "/api": "http://127.0.0.1:7861" } },
  test: { environment: "jsdom", include: ["src/**/*.test.ts"] },
});
