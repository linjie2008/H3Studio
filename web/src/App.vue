<script setup>
import { onMounted, onUnmounted, ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useProjectStore } from './stores/project'
import { useAiStore } from './stores/ai'
import { useUiStore } from './stores/ui'
import { request } from './api'

const route = useRoute()
const router = useRouter()
const projectStore = useProjectStore()
const ai = useAiStore()
const ui = useUiStore()

const PROJECT_ID_KEY = 'h3studio.projectId'

const railItems = [
  { workspace: 'projects', icon: '▦', label: '项目', to: '/projects' },
  { workspace: 'create', icon: '✦', label: '创作', to: '/create' },
  { workspace: 'characters', icon: '♙', label: '设定', to: '/characters' },
  { workspace: 'timeline', icon: '▤', label: '镜头', to: '/timeline' },
  { workspace: 'videos', icon: '▣', label: '视频', to: '/videos' },
  { workspace: 'workflow', icon: '⌁', label: '流程', to: '/workflow' },
  { workspace: 'settings', icon: '⚙', label: '设置', to: '/settings' },
]
const activeWorkspace = computed(() => route.name || '')

const project = computed(() => projectStore.project)

// —— 头部：项目选择 / 新建 / 删除 / 项目文件 ——
async function onNewProject() {
  const name = prompt('项目名称', '新项目')
  if (name === null) return
  try {
    if (project.value && !(await projectStore.save())) return
    await projectStore.create(name)
    router.push('/settings')
    ui.toast('已创建项目')
  } catch (e) { ui.toast(e.message) }
}

async function onDeleteProject() {
  if (!project.value) return
  if (!confirm(`删除项目“${project.value.name}”？该项目的素材和历史文件会保留。`)) return
  try {
    await projectStore.remove(project.value.id)
    ui.toast('项目已删除')
  } catch (e) { ui.toast('删除项目失败：' + e.message) }
}

function selectProject(projectId) {
  if (project.value?.id === projectId) return
  projectStore.select(projectId)
}

async function exportPack(director = false) {
  if (!project.value) { ui.toast('请先选择项目'); return }
  await projectStore.save()
  const a = document.createElement('a')
  a.href = `/api/projects/${project.value.id}/${director ? 'director-pack' : 'pack'}`
  a.download = project.value.name + (director ? '.mmxpack.zip' : '.h3pack.zip')
  a.click()
}

async function onImportPack(event) {
  const file = event.target.files[0]
  event.target.value = ''
  if (!file) return
  try {
    if (project.value && !(await projectStore.save())) return
    const imported = await request('/api/packs/import', 'POST', file)
    projectStore.projects.push(imported)
    projectStore.select(imported.id)
    ui.toast('项目包导入成功')
  } catch (e) { ui.toast('导入失败：' + e.message) }
}

// —— 头部：保存状态 / DeepSeek 徽章 / 推理服务连接 ——
const deepseekBadge = ref({ text: 'DeepSeek 检测中…', cls: '', title: '' })
async function refreshDeepseekHead(force = false) {
  deepseekBadge.value = { ...deepseekBadge.value, text: 'DeepSeek 检测中…' }
  try {
    const d = await ai.refreshHeadStatus(force)
    const text = {
      connected: 'DeepSeek 最近成功', unverified: 'DeepSeek 待验证',
      unconfigured: 'DeepSeek 未配置', model_unavailable: 'DeepSeek 模型不可用',
      insufficient_balance: 'DeepSeek 余额不足', completion_error: 'DeepSeek 最近失败',
      error: 'DeepSeek 连接失败',
    }[d.state] || 'DeepSeek 连接失败'
    const cls = d.state === 'connected' ? 'ok'
      : ['error', 'model_unavailable', 'insufficient_balance', 'completion_error'].includes(d.state) ? 'error' : ''
    const time = d.checked_at ? new Date(d.checked_at * 1000).toLocaleTimeString() : ''
    deepseekBadge.value = { text, cls, title: `${d.message} · ${d.model || '未配置模型'} · ${time}；点击重新检测。` }
  } catch (e) {
    deepseekBadge.value = { text: 'DeepSeek 连接失败', cls: 'error', title: e.message }
  }
}

async function checkHealth() {
  try {
    const d = await request('/api/health')
    ui.setConnection(
      d.connected ? (d.ops_loaded ? '推理服务已连接' : '请重启 ComfyUI 载入增强节点') : '推理服务未连接',
      Boolean(d.connected && d.ops_loaded))
  } catch {
    ui.setConnection('连接检测失败', false)
  }
}

let healthTimer, deepseekTimer
onMounted(async () => {
  await projectStore.load()
  await Promise.all([ai.refreshDeepseek(), ai.refreshMimo(), ai.refreshProvider()])
  checkHealth()
  refreshDeepseekHead()
  healthTimer = setInterval(checkHealth, 15000)
  deepseekTimer = setInterval(() => refreshDeepseekHead(), 60000)
})
onUnmounted(() => {
  clearInterval(healthTimer)
  clearInterval(deepseekTimer)
})
</script>

<template>
  <div class="app-frame">
    <aside class="app-rail" aria-label="主导航">
      <div class="rail-logo" title="H3 Studio · 导演台（8765）">H3</div>
      <button
        v-for="item in railItems" :key="item.workspace" type="button"
        :class="{ active: activeWorkspace === item.workspace }"
        :title="item.label" @click="router.push(item.to)">
        <span>{{ item.icon }}</span><small>{{ item.label }}</small>
      </button>
    </aside>
    <main class="app-shell">
      <header class="app-head">
        <div class="brand">
          <div class="sub">MINIMAX H3 · DIRECTOR STUDIO · 8765</div>
          <h1>H3 Studio · 导演台</h1>
        </div>
        <div class="project-tools">
          <span class="header-label">项目</span>
          <div id="projectList">
            <button
              v-for="p in projectStore.projects" :key="p.id" type="button"
              class="project-item" :class="{ active: p.id === project?.id }"
              @click="selectProject(p.id)">{{ p.name }}</button>
          </div>
          <button type="button" class="top-btn" @click="onNewProject">＋ 新建</button>
          <button type="button" class="top-btn danger" title="删除当前项目"
            :disabled="!project" @click="onDeleteProject">删除当前项目</button>
          <details class="project-menu">
            <summary>项目文件</summary>
            <div class="project-menu-body">
              <button type="button" class="alt" :disabled="!project" @click="exportPack(false)">导出项目包</button>
              <button type="button" class="alt" :disabled="!project" @click="exportPack(true)">导出导演包</button>
              <label class="alt file-import">导入项目包
                <input type="file" accept=".zip" @change="onImportPack">
              </label>
            </div>
          </details>
        </div>
        <div class="head-right">
          <span id="saveState" class="muted">{{ projectStore.saveState }}</span>
          <button
            type="button" id="deepseekHeadStatus" :class="deepseekBadge.cls"
            :title="deepseekBadge.title"
            @click="refreshDeepseekHead(true)">{{ deepseekBadge.text }}</button>
          <div id="connection" class="status" :class="{ ok: ui.connectionOk }">{{ ui.connection }}</div>
        </div>
      </header>

      <router-view v-slot="{ Component }">
        <component :is="Component" />
      </router-view>
    </main>
    <div id="toast" class="toast" :style="{ display: ui.toastText ? 'block' : 'none' }">{{ ui.toastText }}</div>
  </div>
</template>
