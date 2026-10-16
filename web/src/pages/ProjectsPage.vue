<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useProjectStore } from '../stores/project'
import { useUiStore } from '../stores/ui'
import { request } from '../api'

const router = useRouter()
const projectStore = useProjectStore()
const ui = useUiStore()

const selection = ref(new Set())
const validSelection = computed(() =>
  [...selection.value].filter(id => projectStore.projects.some(p => p.id === id)))

function toggleSelect(projectId, checked) {
  const next = new Set(selection.value)
  checked ? next.add(projectId) : next.delete(projectId)
  selection.value = next
}
function selectAll() {
  selection.value = new Set(projectStore.projects.map(p => p.id))
}
function clearSelection() {
  selection.value = new Set()
}

async function openProject(projectId) {
  try {
    if (projectStore.project && !(await projectStore.save())) return
    projectStore.select(projectId)
    router.push('/settings')
  } catch (e) { ui.toast(e.message) }
}

async function newProject() {
  const name = prompt('项目名称', '新项目')
  if (name === null) return
  try {
    if (projectStore.project && !(await projectStore.save())) return
    await projectStore.create(name)
    router.push('/settings')
    ui.toast('已创建项目')
  } catch (e) { ui.toast(e.message) }
}

async function deleteProject(projectId) {
  const item = projectStore.projects.find(p => p.id === projectId)
  if (!item || !confirm(`删除项目“${item.name}”？该项目的素材和历史文件会保留。`)) return
  try {
    await projectStore.remove(projectId)
    ui.toast('项目已删除')
  } catch (e) { ui.toast('删除项目失败：' + e.message) }
}

async function deleteSelected() {
  const ids = validSelection.value
  if (!ids.length) return
  if (!confirm(`删除选中的 ${ids.length} 个项目？素材和历史文件会保留。`)) return
  try {
    const result = await request('/api/projects/batch-delete', 'POST', { project_ids: ids })
    selection.value = new Set()
    await projectStore.load()
    if (result.blocked?.length) {
      ui.toast(`${result.deleted.length} 个项目已删除；${result.blocked.length} 个因正在生成被跳过`)
    } else {
      ui.toast(`已删除 ${result.deleted.length} 个项目`)
    }
  } catch (e) { ui.toast(e.message) }
}

async function cloneProject(projectId) {
  const source = projectStore.projects.find(p => p.id === projectId)
  if (!source) return
  const name = prompt('复制后的项目名称', `${source.name} 副本`)
  if (name === null) return
  try {
    if (projectStore.project && !(await projectStore.save())) return
    await projectStore.clone(projectId, name)
    router.push('/settings')
    ui.toast(`已完整复制为“${name}”`)
  } catch (e) { ui.toast('复制项目失败：' + e.message) }
}
</script>

<template>
  <section class="projects-home">
    <div class="projects-home-head">
      <div>
        <span class="eyebrow">你的工作空间</span>
        <h2>项目</h2>
        <p>选择项目，继续创作你的下一支视频。</p>
      </div>
      <button type="button" class="start-btn" @click="newProject">＋ 新建项目</button>
    </div>

    <div class="project-bulk-toolbar">
      <button type="button" class="alt" :disabled="!projectStore.projects.length" @click="selectAll">全选项目</button>
      <button type="button" class="alt" :disabled="!validSelection.length" @click="clearSelection">清空选择</button>
      <button type="button" class="danger" :disabled="!validSelection.length" @click="deleteSelected">
        批量删除{{ validSelection.length ? `（已选 ${validSelection.length}）` : '' }}
      </button>
    </div>

    <p v-if="projectStore.loadError" class="muted">读取项目失败：{{ projectStore.loadError }}</p>
    <p v-else-if="!projectStore.loaded" class="muted">正在读取项目…</p>
    <p v-else-if="!projectStore.projects.length" class="muted">还没有项目；点击「＋ 新建项目」开始创作。</p>

    <div v-else class="project-card-grid">
      <article
        v-for="p in projectStore.projects" :key="p.id"
        class="project-card" :class="{ 'is-active': p.id === projectStore.project?.id }">
        <label class="project-bulk-select">
          <input
            type="checkbox"
            :checked="selection.has(p.id)"
            @change="toggleSelect(p.id, $event.target.checked)">
          选择
        </label>
        <button type="button" class="project-card-open" @click="openProject(p.id)">
          <span class="project-card-cover"><span>H3</span><i>✦</i></span>
          <span class="project-card-body">
            <strong>{{ p.name }}</strong>
            <small>{{ p.shots?.length || 0 }} 个镜头 ·
              {{ p.id === projectStore.project?.id ? '正在使用' : '点击继续创作' }}</small>
          </span>
        </button>
        <div class="row" style="padding:0 12px 12px;gap:8px">
          <button type="button" class="alt" style="flex:1" @click="cloneProject(p.id)">复制</button>
          <button type="button" class="danger" style="flex:1" @click="deleteProject(p.id)">删除项目</button>
        </div>
      </article>
    </div>
  </section>
</template>
