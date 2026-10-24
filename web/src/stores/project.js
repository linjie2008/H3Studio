import { defineStore } from 'pinia'
import { request } from '../api'
import { alignCanvasDimension } from '../lib/fields'

// 项目状态与保存管线。save() 的语义逐行移植自旧 UI：
// - 600ms 防抖 + saveRevision 计数判断保存期间是否有新编辑；
// - PUT 全量项目，服务端按 _data_version 做乐观锁（冲突返回 400 文案）；
// - 在途保存未完成时再次 save() 会排队等待，完成后立即续保存。
export const useProjectStore = defineStore('project', {
  state: () => ({
    projects: [],
    project: null,
    assets: [],
    saveState: '已保存',
    saveTimer: null,
    saveInFlight: null,
    saveRevision: 0,
    baseSnapshot: null,
    loaded: false,
    loadError: '',
  }),

  actions: {
    async load() {
      try {
        this.projects = await request('/api/projects')
        const stored = localStorage.getItem('h3studio.projectId')
        if (!this.project && this.projects.length) {
          this.project = this.projects.find(p => p.id === stored) || this.projects[0]
        } else if (this.project) {
          this.project = this.projects.find(p => p.id === this.project.id) || null
        }
        if (this.project) localStorage.setItem('h3studio.projectId', this.project.id)
        else localStorage.removeItem('h3studio.projectId')
        this.baseSnapshot = this.project ? JSON.parse(JSON.stringify(this.project)) : null
        await this.loadAssets()
        this.loadError = ''
      } catch (e) {
        this.loadError = e.message
      } finally {
        this.loaded = true
      }
    },

    async loadAssets() {
      this.assets = this.project
        ? await request('/api/assets?project_id=' + encodeURIComponent(this.project.id))
        : []
    },

    async create(name) {
      const project = await request('/api/projects', 'POST', { name })
      this.projects.push(project)
      this.project = project
      this.assets = []
      localStorage.setItem('h3studio.projectId', project.id)
      return project
    },

    async remove(projectId) {
      await request('/api/projects/' + projectId, 'DELETE')
      this.projects = this.projects.filter(p => p.id !== projectId)
      if (this.project?.id === projectId) {
        this.project = null
        localStorage.removeItem('h3studio.projectId')
      }
    },

    async clone(projectId, name) {
      const clone = await request(`/api/projects/${projectId}/clone`, 'POST', { name })
      this.projects.push(clone)
      this.project = clone
      await this.loadAssets()
      localStorage.setItem('h3studio.projectId', clone.id)
      return clone
    },

    select(projectId) {
      this.project = this.projects.find(p => p.id === projectId) || null
      this.baseSnapshot = this.project ? JSON.parse(JSON.stringify(this.project)) : null
      if (this.project) localStorage.setItem('h3studio.projectId', this.project.id)
      else localStorage.removeItem('h3studio.projectId')
      this.loadAssets()
    },

    // 单字段写入 + 触发防抖保存；coerce 已在调用处完成类型转换。
    set(key, value) {
      if (!this.project) return
      this.project[key] = value
      this.scheduleSave()
    },

    scheduleSave() {
      if (!this.project) return
      this.project.width = alignCanvasDimension(this.project.width, 1344)
      this.project.height = alignCanvasDimension(this.project.height, 768)
      this.saveRevision++
      this.saveState = '未保存'
      clearTimeout(this.saveTimer)
      this.saveTimer = setTimeout(() => this.save(), 600)
    },

    async save() {
      if (!this.project) return false
      clearTimeout(this.saveTimer)
      if (this.saveInFlight) {
        const finished = await this.saveInFlight
        if (!finished) return false
        return this.save()
      }
      const projectId = this.project.id
      const revision = this.saveRevision
      // 项目数据全部是 JSON 安全值；JSON 快照与实际 PUT 载荷完全一致，
      // 也避免 structuredClone 对响应式代理的兼容问题。
      const snapshot = JSON.parse(JSON.stringify(this.project))
      const signature = JSON.stringify(snapshot)
      this.saveState = '保存中…'
      const task = (async () => {
        try {
          const saved = await request('/api/projects/' + projectId, 'PUT', snapshot)
          this.projects = this.projects.map(p => p.id === projectId ? saved : p)
          if (this.project?.id === projectId) {
            const unchanged = this.saveRevision === revision &&
              JSON.stringify(this.project) === signature
            if (unchanged) {
              this.project = saved
              this.baseSnapshot = JSON.parse(JSON.stringify(saved))
              this.saveState = '已保存'
            } else {
              // 保存期间用户又改了：只吸收服务端版本号，再排一次保存。
              this.project._data_version = saved._data_version
              this.saveState = '正在保存较新的修改…'
              this.scheduleSave()
            }
          } else {
            this.saveState = '已保存'
          }
          return true
        } catch (e) {
          // 乐观锁冲突：项目在别处（另一个窗口 / 智能体 / 素材变动）被改过。
          // 必须把本地改过的字段合并回最新版本再存一次，否则「把速度改成
          // Turbo 8 步」这类改动会被静默丢掉，表现为「选了 8 步却跑 4 步」。
          if (/其他页面更新|版本无效/.test(e.message)) {
            try {
              const latest = await request('/api/projects/' + projectId)
              const merged = JSON.parse(JSON.stringify(latest))
              const base = this.baseSnapshot || snapshot
              const changed = (value, baseValue) =>
                JSON.stringify(value) !== JSON.stringify(baseValue)
              const skip = new Set(['id', 'shots', 'next_subject_number', '_data_version'])
              for (const key of Object.keys(this.project || {})) {
                if (key.startsWith('_') || skip.has(key)) continue
                if (changed(this.project[key], base[key])) merged[key] = this.project[key]
              }
              for (const shot of (this.project?.shots || [])) {
                const target = (merged.shots || []).find(item => item.id === shot.id)
                const baseShot = (base.shots || []).find(item => item.id === shot.id)
                if (!target || !baseShot) continue
                for (const key of Object.keys(shot)) {
                  if (key.startsWith('_') || key === 'id') continue
                  if (changed(shot[key], baseShot[key])) target[key] = shot[key]
                }
              }
              const saved = await request('/api/projects/' + projectId, 'PUT', merged)
              this.projects = this.projects.map(p => p.id === projectId ? saved : p)
              if (this.project?.id === projectId) this.project = saved
              this.baseSnapshot = JSON.parse(JSON.stringify(saved))
              this.saveState = '已保存'
              this.saveError = ''
              return true
            } catch (second) {
              this.saveState = '保存失败'
              this.saveError = second.message
              return false
            }
          }
          this.saveState = '保存失败'
          this.saveError = e.message
          return false
        }
      })()
      this.saveInFlight = task
      try {
        return await task
      } finally {
        if (this.saveInFlight === task) this.saveInFlight = null
      }
    },

    async uploadAsset(file, projectId = this.project?.id) {
      if (!projectId) throw new Error('请先选择项目')
      const asset = await request(
        '/api/assets?project_id=' + encodeURIComponent(projectId) +
        '&name=' + encodeURIComponent(file.name), 'POST', file)
      if (this.project?.id === asset.project_id) this.assets.push(asset)
      return asset
    },

    toggleSharedRef(kind, assetId) {
      const list = this.project.shared_refs?.[kind] || []
      const selected = list.includes(assetId)
      this.project.shared_refs ||= { images: [], videos: [], audios: [] }
      this.project.shared_refs[kind] = selected
        ? list.filter(v => v !== assetId)
        : [...list, assetId]
      return !selected // 返回 true 表示设为共享
    },
  },
})
