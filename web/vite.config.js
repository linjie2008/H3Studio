import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发：vite dev (5173) 代理 API 到工作台 8765
// 构建：dist 由 app.py 在 /app 下提供
export default defineConfig({
  base: '/app/',
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/media': 'http://127.0.0.1:8765',
      '/comfy-ws': { target: 'ws://127.0.0.1:8765', ws: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
