import { defineStore } from 'pinia'
import { request } from '../api'

// 提示词 Skill 列表与项目绑定。对应旧 UI 的 promptSkills + renderPromptSkillSelect。
export const useSkillsStore = defineStore('skills', {
  state: () => ({
    list: [],        // [{id, name, description, builtin, protected, source}]
    loaded: false,
    loadError: '',
  }),
  getters: {
    labelOf: (state) => (skillId) => {
      const skill = state.list.find(s => s.id === skillId)
      if (!skill) return ''
      return skill.builtin ? `官方 · ${skill.name}`
        : skill.source === 'AICG3D' ? `AICG3D · ${skill.name}`
        : skill.name
    },
  },
  actions: {
    async refresh() {
      try {
        this.list = await request('/api/skills')
        this.loadError = ''
      } catch (e) {
        this.loadError = '读取 Skill 列表失败：' + e.message
      } finally {
        this.loaded = true
      }
    },
    async preview(skillId) {
      return request(`/api/skills/${encodeURIComponent(skillId)}/preview`)
    },
    async importSkill(file) {
      const response = await fetch('/api/skills/import?name=' + encodeURIComponent(file.name), {
        method: 'POST', body: file,
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`)
      await this.refresh()
      return result
    },
    async remove(skillId) {
      const result = await request('/api/skills/' + encodeURIComponent(skillId), 'DELETE')
      this.list = result.skills || []
    },
    async recommend(projectId) {
      return request('/api/skill-agent/recommend', 'POST', { project_id: projectId })
    },
    async suggestParameters(projectId) {
      return request('/api/skill-agent/parameters', 'POST', { project_id: projectId })
    },
  },
})
