import { defineStore } from 'pinia'

// 全局提示与连接状态。toast 语义与旧 UI 一致（3.5 秒自动消失）。
export const useUiStore = defineStore('ui', {
  state: () => ({
    toastText: '',
    toastTimer: null,
    connection: '检测推理服务…',
    connectionOk: false,
    deepseekHead: { text: 'DeepSeek 检测中…', state: '', title: '' },
  }),
  actions: {
    toast(text) {
      this.toastText = text
      clearTimeout(this.toastTimer)
      this.toastTimer = setTimeout(() => { this.toastText = '' }, 3500)
    },
    setConnection(text, ok) {
      this.connection = text
      this.connectionOk = ok
    },
    setDeepseekHead(patch) {
      Object.assign(this.deepseekHead, patch)
    },
  },
})
