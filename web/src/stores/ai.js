import { defineStore } from 'pinia'
import { request } from '../api'

// 云端 AI 配置：DeepSeek / 小米 MiMo / MiniMax / 提示词模型选择。
// 与旧 UI 的 refreshDeepseekSettings / refreshMimoSettings / refreshMinimaxSettings / refreshPromptProvider 对应。
export const useAiStore = defineStore('ai', {
  state: () => ({
    deepseek: null,   // {provider, model, configured, key_source, ...}
    mimo: null,
    minimax: null,
    models: {},         // {deepseek: {model, source, catalog}, ...}
    modelsError: '',
    provider: 'auto',
    providerSaved: 'auto',
    status: '正在检查 DeepSeek 配置…',
    mimoStatus: '正在检查 MiMo 配置…',
    minimaxStatus: '正在检查 MiniMax 配置…',
    providerStatus: '正在读取当前模型选择…',
  }),
  actions: {
    async refreshDeepseek() {
      try {
        const d = await request('/api/ai/settings')
        const source = d.key_source === 'environment' ? '环境变量'
          : d.key_source === 'saved' ? '本机私密配置' : '未配置'
        this.deepseek = d
        this.status = d.configured
          ? `已配置：${d.provider} · ${d.model} · ${source}`
          : '未配置 API Key；提示词优化和 AI 分镜暂不可用。'
      } catch (e) {
        this.status = '读取 DeepSeek 配置失败：' + e.message
      }
    },
    canRemoveDeepseek() {
      return this.deepseek && !['none', 'environment'].includes(this.deepseek.key_source)
    },
    async saveDeepseekKey(key) {
      await request('/api/ai/settings', 'POST', { api_key: key })
      await this.refreshDeepseek()
    },
    async removeDeepseekKey() {
      await request('/api/ai/settings', 'DELETE')
      await this.refreshDeepseek()
    },
    async testDeepseek() {
      return request('/api/ai/test', 'POST', {})
    },

    async refreshMimo() {
      try {
        const d = await request('/api/ai/mimo/settings')
        const source = d.key_source === 'environment' ? '环境变量'
          : d.key_source === 'saved' ? '本机私密配置' : '未配置'
        this.mimo = d
        this.mimoStatus = d.configured
          ? `已配置：${d.provider} · ${d.model} · ${source} · DeepSeek 不可用时自动接管`
          : '未配置；DeepSeek 失败时将没有备用模型。'
      } catch (e) {
        this.mimoStatus = '读取 MiMo 配置失败：' + e.message
      }
    },
    canRemoveMimo() {
      return this.mimo && !['none', 'environment'].includes(this.mimo.key_source)
    },
    async saveMimoKey(key) {
      await request('/api/ai/mimo/settings', 'POST', { api_key: key })
      await this.refreshMimo()
    },
    async removeMimoKey() {
      await request('/api/ai/mimo/settings', 'DELETE')
      await this.refreshMimo()
    },
    async testMimo() {
      return request('/api/ai/mimo/test', 'POST', {})
    },

    async refreshMinimax() {
      try {
        const d = await request('/api/ai/minimax/settings')
        const source = d.key_source === 'environment' ? '环境变量'
          : d.key_source === 'saved' ? '本机私密配置' : '未配置'
        this.minimax = d
        this.minimaxStatus = d.configured
          ? `已配置：${d.provider} · ${d.model} · ${source} · 前两家都不可用时自动接管`
          : '未配置；DeepSeek 与 MiMo 都失败时将没有备用模型。'
      } catch (e) {
        this.minimaxStatus = '读取 MiniMax 配置失败：' + e.message
      }
    },
    canRemoveMinimax() {
      return this.minimax && !['none', 'environment'].includes(this.minimax.key_source)
    },
    async saveMinimaxKey(key) {
      await request('/api/ai/minimax/settings', 'POST', { api_key: key })
      await this.refreshMinimax()
    },
    async removeMinimaxKey() {
      await request('/api/ai/minimax/settings', 'DELETE')
      await this.refreshMinimax()
    },
    async testMinimax() {
      return request('/api/ai/minimax/test', 'POST', {})
    },

    async refreshPromptModels() {
      try {
        const d = await request('/api/ai/models')
        this.models = d.providers || {}
        this.modelsError = ''
      } catch (e) {
        this.modelsError = '读取模型清单失败：' + e.message
      }
    },
    async savePromptModel(provider, model) {
      const result = await request('/api/ai/models', 'POST', { provider, model })
      await this.refreshPromptModels()
      return result.model
    },

    async refreshProvider() {
      try {
        const d = await request('/api/ai/provider')
        this.provider = d.selected_provider
        this.providerSaved = d.selected_provider
        const labels = {
          auto: '自动模式：DeepSeek 优先；失败时依次切换 MiMo、MiniMax。',
          deepseek: '仅使用 DeepSeek；失败时不切换备用模型。',
          mimo: '仅使用小米 MiMo；不请求 DeepSeek。',
          minimax: '仅使用 MiniMax；不请求 DeepSeek 与 MiMo。',
        }
        this.providerStatus = labels[d.selected_provider] || ''
      } catch (e) {
        this.providerStatus = '读取模型选择失败：' + e.message
      }
    },
    async setProvider(value) {
      const result = await request('/api/ai/provider', 'POST', { provider: value })
      this.providerSaved = result.selected_provider
      await this.refreshProvider()
      return result.selected_provider
    },

    async refreshHeadStatus(force = false) {
      // 返回原始状态，徽章文案映射放在组件层（与旧 UI 的 badge 逻辑一致）。
      return request('/api/ai/status' + (force ? '?refresh=1' : ''))
    },
  },
})
