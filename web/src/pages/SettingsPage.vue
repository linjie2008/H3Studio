<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useProjectStore } from '../stores/project'
import { useAiStore } from '../stores/ai'
import { useSkillsStore } from '../stores/skills'
import { useUiStore } from '../stores/ui'
import { request } from '../api'
import { CANVAS_MP_PRESETS, canvasPresetFor, applyCanvasPreset, coerceField } from '../lib/fields'

const router = useRouter()
const projectStore = useProjectStore()
const ai = useAiStore()
const skills = useSkillsStore()
const ui = useUiStore()

const project = computed(() => projectStore.project)

// —— 画布 ——
const canvasOrientation = computed(() =>
  Number(project.value?.height) > Number(project.value?.width) ? 'portrait' : 'landscape')

function setOrientation(event) {
  if (!project.value) return
  const portrait = event.target.value === 'portrait'
  if (project.value.width === project.value.height) {
    const nearest = CANVAS_MP_PRESETS.reduce((best, row) =>
      Math.abs(row[1] * row[2] - project.value.width * project.value.height) <
      Math.abs(best[1] * best[2] - project.value.width * project.value.height) ? row : best)
    applyCanvasPreset(project.value, nearest[0], portrait)
  } else if ((project.value.height > project.value.width) !== portrait) {
    ;[project.value.width, project.value.height] = [project.value.height, project.value.width]
  }
  projectStore.scheduleSave()
}

const presetOptions = computed(() => {
  if (!project.value) return []
  const selected = canvasPresetFor(project.value.width, project.value.height)
  const portrait = canvasOrientation.value === 'portrait'
  const options = []
  if (!selected) options.push({ value: '', label: `自定义 · ${project.value.width} × ${project.value.height}` })
  for (const [mp, w, h] of CANVAS_MP_PRESETS) {
    options.push({ value: mp, label: `${mp === '0.98' ? 'H3 官方 · ' : ''}${mp} MP · ${portrait ? `${h} × ${w}` : `${w} × ${h}`}` })
  }
  return options
})

function setPreset(event) {
  if (!event.target.value) return
  applyCanvasPreset(project.value, event.target.value)
  projectStore.scheduleSave()
}

async function applyOfficialCanvas() {
  if (!project.value) return
  applyCanvasPreset(project.value, '0.98')
  projectStore.scheduleSave()
  if (await projectStore.save()) {
    ui.toast(`已保存 H3 官方画布 ${project.value.width}×${project.value.height}`)
  }
}

// —— 通用字段绑定（与旧 UI data-project 处理器一致） ——
function onField(event, key) {
  projectStore.set(key, coerceField(key, event.target.value, event.target))
}

// —— 模型下拉（来自 /api/models） ——
const catalog = ref({ semantic_adapters: [], latent_models: [], h3_clip_models: [], h3_video_vaes: [], sage_attention: false, veda_attention: false })
onMounted(async () => {
  try { catalog.value = await request('/api/models') } catch { /* 模型目录不可用时保留空列表 */ }
})

const clipModelOptions = computed(() => {
  const clips = catalog.value.h3_clip_models || []
  const current = project.value?.h3_clip_model
  const label = name => name.includes('heretic') ? 'Heretic NVFP4 · 默认推荐 (15GB)'
    : name.includes('nvfp4_awq') ? 'NVFP4 AWQ · 速度优先 (15GB)'
    : name.includes('int8') ? 'INT8 (26GB)' : name
  const options = clips.map(name => ({ value: name, label: label(name) }))
  if (current && !clips.includes(current)) options.unshift({ value: current, label: current })
  return options
})
const videoVaeOptions = computed(() => {
  const vaes = catalog.value.h3_video_vaes || []
  const current = project.value?.h3_video_vae
  const options = vaes.map(name => ({ value: name, label: name }))
  if (current && !vaes.includes(current)) options.unshift({ value: current, label: current })
  return options
})

// —— DeepSeek / MiMo / 模型选择面板 ——
const deepseekKey = ref('')
const deepseekTestStatus = ref('')
const deepseekTesting = ref(false)
async function saveDeepseekKey() {
  const key = deepseekKey.value.trim()
  if (!key) { ui.toast('请先粘贴 DeepSeek API Key'); return }
  deepseekKey.value = ''
  try {
    await ai.saveDeepseekKey(key)
    ui.toast('DeepSeek API Key 已保存到 WSL 用户私密配置')
    await refreshHead()
  } catch (e) { ui.toast(e.message); deepseekKey.value = key }
}
async function testDeepseek() {
  deepseekTesting.value = true
  deepseekTestStatus.value = '正在向 DeepSeek 发送测试请求…'
  try {
    const result = await ai.testDeepseek()
    deepseekTestStatus.value = `连接成功 · ${result.provider} · ${result.model}`
  } catch (e) {
    deepseekTestStatus.value = `连接失败：${e.message}`
  } finally {
    deepseekTesting.value = false
    refreshHead()
  }
}
async function removeDeepseekKey() {
  if (!confirm('清除保存在此 WSL 用户下的 DeepSeek API Key？')) return
  try {
    await ai.removeDeepseekKey()
    ui.toast('已清除 DeepSeek API Key')
    refreshHead()
  } catch (e) { ui.toast(e.message) }
}

const mimoKey = ref('')
const mimoTestStatus = ref('')
const mimoTesting = ref(false)
async function saveMimoKey() {
  const key = mimoKey.value.trim()
  if (!key) { ui.toast('请先粘贴小米 MiMo API Key'); return }
  mimoKey.value = ''
  try {
    await ai.saveMimoKey(key)
    ui.toast('小米 MiMo API Key 已保存到 WSL 用户私密配置')
  } catch (e) { ui.toast(e.message); mimoKey.value = key }
}
async function testMimo() {
  mimoTesting.value = true
  mimoTestStatus.value = '正在向小米 MiMo 发送测试请求…'
  try {
    const result = await ai.testMimo()
    mimoTestStatus.value = `连接成功 · ${result.provider} · ${result.model}`
  } catch (e) {
    mimoTestStatus.value = `连接失败：${e.message}`
  } finally { mimoTesting.value = false }
}
async function removeMimoKey() {
  if (!confirm('清除保存在此 WSL 用户下的小米 MiMo API Key？')) return
  try {
    await ai.removeMimoKey()
    ui.toast('已清除小米 MiMo API Key')
  } catch (e) { ui.toast(e.message) }
}

async function onProviderChange(event) {
  const select = event.target
  const previous = ai.providerSaved
  select.disabled = true
  try {
    const selected = await ai.setProvider(select.value)
    ui.toast(selected === 'mimo' ? '已切换为仅小米 MiMo'
      : selected === 'deepseek' ? '已切换为仅 DeepSeek' : '已切换为自动备用模式')
  } catch (e) {
    select.value = previous
    ui.toast('切换失败：' + e.message)
  } finally { select.disabled = false }
}

async function refreshHead() {
  // 头部徽章由 App.vue 管理；这里只触发一次带缓存刷新，保持语义一致。
  try { await ai.refreshHeadStatus(true) } catch { /* 忽略 */ }
}

// —— Skill 面板 ——
const skillSearch = ref('')
const skillPreview = ref(null) // {name, files:[{name,content}]}
const skillRecommendations = ref(null) // {summary, recommendations:[{id,name,reason}]}
const recommending = ref(false)

const currentSkillId = computed(() =>
  project.value?.prompt_skill ||
  (skills.list.find(s => s.builtin)?.id || 'h3-prompt-writing'))

const filteredSkills = computed(() => {
  const query = skillSearch.value.trim().toLowerCase()
  const active = currentSkillId.value
  return skills.list.filter(skill =>
    skill.id === active || !query ||
    `${skill.name} ${skill.description} ${skill.id}`.toLowerCase().includes(query))
})

const selectedSkill = computed(() => skills.list.find(s => s.id === currentSkillId.value))

const skillDescription = computed(() => {
  const base = selectedSkill.value?.description || '导入一个 SKILL.md 或 Skill ZIP，再在这里切换使用。'
  return `${base} · 显示 ${filteredSkills.value.length}/${skills.list.length} 个 Skill`
})

function skillPrefix(skill) {
  return skill.builtin ? '官方 · ' : skill.source === 'AICG3D' ? 'AICG3D 社区 · ' : '自定义 · '
}

async function onSkillChange(event) {
  if (!project.value) return
  project.value.prompt_skill = event.target.value
  skillSettingsProposal.value = null
  await projectStore.save()
}

async function previewSkill() {
  const id = currentSkillId.value
  if (!id) return
  try {
    skillPreview.value = await skills.preview(id)
  } catch (e) { ui.toast(e.message) }
}

async function onImportSkill(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  try {
    const result = await skills.importSkill(file)
    if (project.value) {
      project.value.prompt_skill = result.id
      await projectStore.save()
    }
    ui.toast(`已导入 Skill：${result.name}`)
  } catch (e) { ui.toast('导入失败：' + e.message) }
}

async function deleteSkill() {
  const skill = selectedSkill.value
  if (!skill || skill.protected) return
  if (!confirm(`删除 Skill「${skill.name}」？`)) return
  try {
    await skills.remove(skill.id)
    if (project.value?.prompt_skill === skill.id) {
      project.value.prompt_skill = 'h3-prompt-writing'
      projectStore.scheduleSave()
    }
    ui.toast('Skill 已删除')
  } catch (e) { ui.toast(e.message) }
}

async function recommendSkills() {
  if (!project.value) return
  const projectId = project.value.id
  recommending.value = true
  skillRecommendations.value = null
  try {
    if (!(await projectStore.save())) return
    const result = await skills.recommend(projectId)
    if (project.value?.id !== projectId) return
    skillRecommendations.value = result
  } catch (e) { ui.toast(e.message) } finally { recommending.value = false }
}

async function useRecommendedSkill(skillId) {
  if (!project.value) return
  if (!skills.list.some(item => item.id === skillId)) {
    ui.toast('这个 Skill 已不存在，请重新推荐')
    return
  }
  skillSearch.value = ''
  project.value.prompt_skill = skillId
  if (await projectStore.save()) {
    ui.toast(`已为当前项目选择「${skills.labelOf(skillId)}」`)
  }
}

// —— Skill 参数智能体 ——
const skillSettingsProposal = ref(null) // {summary, changes, settings, projectId, skill_id}
const suggesting = ref(false)

function parameterValueLabel(key, value) {
  if (key === 'generation_speed') return value === 'turbo4' ? '极速 · 4 步（4 步 LoRA）' : value === 'turbo' ? 'Turbo · 8 步（8 步 LoRA）'
    : value === 'smart' ? '智能均衡 · 按模式' : value === 'goldfish' ? '金鱼高清·本机版' : '标准 · 20 步'
  if (key === 'sage_attention' || key === 'upscale' || key === 'confirm_upscale') return value ? '开启' : '关闭'
  if (key === 'width' || key === 'height') return `${value}px`
  if (key === 'memory_policy') return value === 'auto_vram' ? '自动显存优化' : '手动'
  if (key === 'refine_mode') return ({ off: '关闭', selflift: '两阶段 SelfLift' })[value] || value
  if (key === 'continuity_frames') return `${value} 帧`
  if (key === 'refine_steps') return `${value} 步`
  if (key?.startsWith('upscale_')) return `${value}p`
  return String(value)
}

async function suggestSkillSettings() {
  if (!project.value) return
  const projectId = project.value.id
  const skillId = currentSkillId.value
  suggesting.value = true
  try {
    if (!(await projectStore.save())) return
    const result = await skills.suggestParameters(projectId)
    if (project.value?.id !== projectId || skillId !== result.skill_id) {
      ui.toast('项目或 Skill 已变化，请重新生成建议')
      return
    }
    skillSettingsProposal.value = { ...result, projectId }
  } catch (e) { ui.toast(e.message) } finally { suggesting.value = false }
}

async function applySkillSettings() {
  const proposal = skillSettingsProposal.value
  if (!proposal || !project.value || proposal.projectId !== project.value.id ||
      proposal.skill_id !== currentSkillId.value) {
    clearSkillSettingsProposal()
    ui.toast('项目或 Skill 已变化，请重新生成建议')
    return
  }
  Object.assign(project.value, proposal.settings)
  clearSkillSettingsProposal()
  if (await projectStore.save()) ui.toast('Skill 参数建议已应用并保存')
}
function clearSkillSettingsProposal() { skillSettingsProposal.value = null }

// —— 存储状态 ——
const storageStatus = ref('正在检查数据库…')
onMounted(async () => {
  try {
    const d = await request('/api/storage/status')
    storageStatus.value = `${d.backend} · 项目 ${d.counts.projects} · 素材 ${d.counts.assets} · 生成历史 ${d.counts.runs} · 媒体文件目录：${d.media_directory}`
  } catch (e) { storageStatus.value = '读取数据库状态失败：' + e.message }
})

// —— 分段拼接 ——
const seamDetails = ref(null)
function openSeamSettings() {
  const el = seamDetails.value
  if (!el) return
  el.open = true
  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}
const remergeStatus = ref('')
const remerging = ref(false)
async function remergeLatest() {
  if (!project.value) return
  const projectId = project.value.id
  remerging.value = true
  remergeStatus.value = '正在重新拼接已有分段…'
  try {
    if (!(await projectStore.save())) throw new Error('项目设置保存失败')
    const run = await request(`/api/projects/${projectId}/remerge`, 'POST', {})
    if (project.value?.id !== projectId) return
    remergeStatus.value = ''
    remergeLink.value = '/media/outputs/' + run.merged
    ui.toast('已按当前项目设置重新拼接，无需重新生成镜头')
  } catch (e) { remergeStatus.value = '拼接失败：' + e.message } finally { remerging.value = false }
}
const remergeLink = ref('')

// —— 全片共享参考素材 ——
const sharedDefs = [
  { key: 'images', kind: 'image', label: '共享参考图' },
  { key: 'videos', kind: 'video', label: '共享参考视频' },
  { key: 'audios', kind: 'audio', label: '共享参考音频' },
]
function sharedCandidates(kind) {
  return projectStore.assets.filter(a => a.kind === kind || (kind === 'audio' && a.kind === 'video'))
}
function isShared(kind, assetId) {
  return (project.value?.shared_refs?.[kind] || []).includes(assetId)
}
async function toggleShared(kind, assetId) {
  if (!project.value) return
  const before = [...(project.value.shared_refs?.[kind] || [])]
  const selected = isShared(kind, assetId)
  projectStore.toggleSharedRef(kind, assetId)
  if (await projectStore.save()) {
    ui.toast(selected ? '已取消全片共享' : '已设为全片共享参考')
  } else {
    project.value.shared_refs[kind] = before
  }
}

onMounted(async () => { await skills.refresh() })
</script>

<template>
  <section v-if="project" class="settings-workspace">
    <div class="settings-scroll">
      <section class="panel settings-page" style="background:#fff;border-radius:14px">
        <div class="row between topbar">
          <div><h2 style="margin:0">项目设置</h2><p class="hint">统一设置全片画布、导出和增强方式。</p></div>
          <button type="button" class="alt" @click="router.push('/advanced')">高级模型参数</button>
        </div>

        <!-- 云端 AI 提示词助手 · DeepSeek -->
        <details class="character-library">
          <summary>云端 AI 提示词助手 · DeepSeek</summary>
          <p class="hint">提示词优化和 AI 分镜将发送到 DeepSeek 云端。用量按 DeepSeek API 账户计费；API Key 仅保存在此 WSL 用户的私有配置中，不会进入项目包。</p>
          <div class="row">
            <input v-model="deepseekKey" type="password" autocomplete="new-password"
              placeholder="粘贴 DeepSeek API Key（保存后会清空输入框）">
            <button type="button" @click="saveDeepseekKey">保存 Key</button>
            <button type="button" class="alt" :disabled="deepseekTesting" @click="testDeepseek">
              {{ deepseekTesting ? '测试中…' : '测试连接' }}</button>
            <button type="button" class="alt" :disabled="!ai.canRemoveDeepseek()" @click="removeDeepseekKey">清除</button>
          </div>
          <p class="muted">{{ ai.status }}</p>
          <p class="muted" aria-live="polite">{{ deepseekTestStatus }}</p>
        </details>

        <!-- 提示词模型选择 -->
        <section class="character-library">
          <h3>提示词模型选择</h3>
          <div class="row">
            <select :value="ai.provider" aria-label="提示词模型选择" @change="onProviderChange">
              <option value="auto">自动 · DeepSeek 优先，MiMo 备用</option>
              <option value="deepseek">仅 DeepSeek</option>
              <option value="mimo">仅小米 MiMo</option>
            </select>
          </div>
          <p class="muted">{{ ai.providerStatus }}</p>
        </section>

        <!-- 备用提示词模型 · 小米 MiMo -->
        <details class="character-library">
          <summary>备用提示词模型 · 小米 MiMo</summary>
          <p class="hint">当 DeepSeek 的真实推理临时不可用时，Studio 会自动改用 MiMo 完成提示词优化、AI 分镜和翻译。两者都不可用时才会报错；MiMo 用量按小米账户计费。</p>
          <div class="row">
            <input v-model="mimoKey" type="password" autocomplete="new-password"
              placeholder="粘贴小米 MiMo API Key（保存后会清空输入框）">
            <button type="button" @click="saveMimoKey">保存 Key</button>
            <button type="button" class="alt" :disabled="mimoTesting" @click="testMimo">
              {{ mimoTesting ? '测试中…' : '测试连接' }}</button>
            <button type="button" class="alt" :disabled="!ai.canRemoveMimo()" @click="removeMimoKey">清除</button>
          </div>
          <p class="muted">{{ ai.mimoStatus }}</p>
          <p class="muted" aria-live="polite">{{ mimoTestStatus }}</p>
        </details>

        <!-- 项目级提示词 Skill -->
        <details class="character-library" open>
          <summary>项目级提示词 Skill</summary>
          <p class="hint">绑定到当前项目：项目内所有镜头的 AI 提示词优化和 AI 分镜都使用所选 Skill；每个项目可独立选择。H3 官方格式规则始终作为基础规则。已接入 AICG3D Skill 库；可搜索、预览，或导入自己的 SKILL.md / ZIP。</p>
          <div class="row" style="margin:8px 0">
            <button type="button" :disabled="recommending || !project" @click="recommendSkills">
              {{ recommending ? '正在分析项目…' : '✨ 根据项目自动推荐' }}</button>
            <span class="muted">不用浏览全部 Skill，先获得最多 5 个候选</span>
          </div>
          <div v-if="skillRecommendations" class="skill-agent-result">
            <p>{{ skillRecommendations.summary || '已筛选适合当前项目的 Skill。' }}</p>
            <div v-for="(item, index) in skillRecommendations.recommendations || []" :key="item.id"
              style="margin-top:9px;padding:10px;border:1px solid #d5e5fb;border-radius:9px;background:#fff">
              <strong>{{ index + 1 }}. {{ item.name }}</strong>
              <p class="hint" style="margin:5px 0">{{ item.reason }}</p>
              <button type="button" class="alt"
                :disabled="item.id === currentSkillId"
                @click="useRecommendedSkill(item.id)">
                {{ item.id === currentSkillId ? '当前正在使用' : '选择这个 Skill' }}</button>
            </div>
          </div>
          <input v-model="skillSearch" type="search"
            placeholder="搜索 Skill 名称或用途，如：短剧、对白、追逐"
            aria-label="搜索提示词 Skill" style="width:100%;margin:8px 0;font-size:16px">
          <div class="row">
            <select :value="currentSkillId" aria-label="当前项目绑定的创作 Skill" @change="onSkillChange">
              <option v-for="skill in filteredSkills" :key="skill.id" :value="skill.id">
                {{ skillPrefix(skill) }}{{ skill.name }}</option>
            </select>
            <button type="button" class="alt" @click="previewSkill">预览 Skill 内容</button>
            <label class="file-import">导入 Skill
              <input type="file" accept=".md,.zip" @change="onImportSkill">
            </label>
            <button type="button" class="danger"
              :disabled="!selectedSkill || selectedSkill.protected" @click="deleteSkill">删除所选 Skill</button>
          </div>
          <p class="muted">{{ skills.loadError || skillDescription }}</p>
        </details>

        <!-- Skill 智能体 · 参数匹配 -->
        <section class="character-library skill-agent-panel">
          <h3>Skill 智能体 · 自动匹配项目参数</h3>
          <p class="hint">根据当前项目绑定的 Skill、现有画布和 ComfyUI 已安装模型，生成画幅、分辨率、速度、显存与增强参数建议。未在 Skill 中体现的设置会尽量保留。建议会先展示理由，点击应用后才写入项目。</p>
          <div class="row">
            <span class="prompt-skill-badge">Skill：{{ skills.labelOf(currentSkillId) }}</span>
            <button type="button" :disabled="suggesting || !project" @click="suggestSkillSettings">
              {{ suggesting ? '正在分析 Skill…' : '✨ 根据 Skill 匹配参数' }}</button>
          </div>
          <div v-if="skillSettingsProposal" class="skill-agent-result">
            <p>{{ skillSettingsProposal.summary || '参数建议已生成。' }}</p>
            <ul>
              <li v-if="!skillSettingsProposal.changes?.length">当前项目设置已符合 Skill 建议，无需调整。</li>
              <li v-for="change in skillSettingsProposal.changes || []" :key="change.key">
                <strong>{{ change.label }}：{{ parameterValueLabel(change.key, change.before) }} → {{ parameterValueLabel(change.key, change.after) }}</strong>
                <span v-if="change.reason"> — {{ change.reason }}</span>
              </li>
            </ul>
            <div class="row">
              <button type="button" :disabled="!skillSettingsProposal.changes?.length" @click="applySkillSettings">应用建议到项目</button>
              <button type="button" class="alt" @click="clearSkillSettingsProposal">暂不应用</button>
            </div>
          </div>
        </section>

        <!-- 项目数据库与存储 -->
        <details class="character-library">
          <summary>项目数据库与存储</summary>
          <p class="hint">项目文字、角色/场景/道具资料、素材索引和生成历史保存在 SQLite 数据库；图片、视频和音频文件保存在素材目录，并由数据库记录关联。旧 state.json 会保留为迁移备份。</p>
          <p class="muted">{{ storageStatus }}</p>
        </details>

        <!-- 项目字段 -->
        <div class="grid3">
          <div class="field"><label>项目名称</label>
            <input :value="project.name" @input="onField($event, 'name')"></div>
          <div class="field"><label>画布方向</label>
            <select :value="canvasOrientation" @change="setOrientation">
              <option value="landscape">16:9 横屏</option>
              <option value="portrait">9:16 竖屏</option>
            </select></div>
          <div class="field"><label>画布像素（MP）</label>
            <select :value="canvasPresetFor(project.width, project.height)" @change="setPreset">
              <option v-for="opt in presetOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
            <button type="button" class="alt" @click="applyOfficialCanvas">设为 H3 官方尺寸</button>
            <p class="hint">官方画布为横屏 1344×768、竖屏 768×1344；手动修改为其他尺寸后显示“自定义”。</p></div>
          <div class="field"><label>宽度</label>
            <input type="number" min="32" step="32" :value="project.width" @input="onField($event, 'width')"></div>
          <div class="field"><label>高度</label>
            <input type="number" min="32" step="32" :value="project.height" @input="onField($event, 'height')"></div>
          <div class="field"><label>H3 注意力加速</label>
            <select :value="project.attention_mode" @change="onField($event, 'attention_mode')">
              <option value="off">关闭</option>
              <option value="sage">SageAttention · 降低显存</option>
              <option value="veda">Veda · 稀疏注意力加速</option>
            </select>
            <p class="hint">两种算法只能选一种。Sage 偏显存优化；Veda 使用 90% 稀疏率预测注意力区块以加速。模型/节点不可用时会在生成前提示。</p></div>
          <div class="field"><label>H3 文本编码器</label>
            <select :value="project.h3_clip_model" @change="onField($event, 'h3_clip_model')">
              <option v-for="opt in clipModelOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
            <p class="hint">默认 Heretic NVFP4（约 15GB）；优先速度可选 NVFP4 AWQ。</p></div>
          <div class="field"><label>H3 视频 VAE</label>
            <select :value="project.h3_video_vae" @change="onField($event, 'h3_video_vae')">
              <option v-for="opt in videoVaeOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
            <p class="hint">FP16 为默认兼容选项；若安装 H3 INT8 VAE，也会出现在此列表。</p></div>
          <p class="hint">ComfyUI 官方 VAE 卷积加速需在启动参数添加 <code>--fast fp16_accumulation</code>，重启 ComfyUI 后生效。</p>
          <div class="field"><label>H3 显存策略</label>
            <select :value="project.memory_policy" @change="onField($event, 'memory_policy')">
              <option value="manual">按项目画布</option>
              <option value="auto_vram">按显存自动调整</option>
            </select>
            <p class="hint">自动模式只调整 H3 生成与 latent 精修画布；最终合片仍按项目画布导出。计划和调整原因会记录在运行报告中。</p></div>
          <div class="field"><label>一镜到底生成方式</label>
            <select :value="project.continuous_render_mode" @change="onField($event, 'continuous_render_mode')">
              <option value="auto">自动 · 短片单次生成</option>
              <option value="segmented">逐镜头生成并接续</option>
            </select>
            <p class="hint">短片优先单次生成；长片用前一镜画面引导下一镜，再合成完整视频。</p>
            <button type="button" class="alt" @click="openSeamSettings">调整分段拼接</button></div>
          <div class="field"><label>关闭强一镜到底时的导出</label>
            <select :value="project.export_mode" @change="onField($event, 'export_mode')">
              <option value="all">合并全片</option>
              <option value="segments">仅分段输出</option>
            </select>
            <p class="hint">开启后会导出完整视频；合计不超过 15 秒的参考图分镜尽量单次生成，更长序列会分段接续，可能出现跳变。</p></div>
          <div class="field"><label>SeedVR2</label>
            <select :value="String(project.upscale)" @change="onField($event, 'upscale')">
              <option value="false">关闭</option><option value="true">开启</option>
            </select></div>
          <div class="field"><label>SeedVR2 短边 / 最长边</label>
            <div class="row">
              <select style="width:46%" :value="String(project.upscale_resolution)" @change="onField($event, 'upscale_resolution')">
                <option>720</option><option>1080</option>
              </select>
              <select style="width:46%" :value="String(project.upscale_max_resolution)" @change="onField($event, 'upscale_max_resolution')">
                <option>1280</option><option>1920</option>
              </select>
            </div></div>
          <div class="field"><label>超分前确认</label>
            <select :value="String(project.confirm_upscale)" @change="onField($event, 'confirm_upscale')">
              <option value="true">先看 H3 原片再决定</option>
              <option value="false">自动超分</option>
            </select></div>
          <div class="field"><label>接续引导长度</label>
            <select :value="String(project.continuity_frames)" @change="onField($event, 'continuity_frames')">
              <option value="5">5 帧</option><option value="22">22 帧 · 推荐</option>
              <option value="39">39 帧</option><option value="56">56 帧</option>
            </select></div>
          <div class="field"><label>换人镜头的身份优先</label>
            <select :value="String(project.identity_first_on_switch)" @change="onField($event, 'identity_first_on_switch')">
              <option value="false">关闭 · 保留强段间引导</option>
              <option value="true">开启 · 换人时不传旧人物尾帧</option>
            </select>
            <p class="hint">适合每段使用不同人物图；可减少脸和服装串到下一段，但接缝可能更明显。</p></div>
          <div class="field"><label>人脸细节增强</label>
            <select :value="String(project.face_refine)" @change="onField($event, 'face_refine')">
              <option value="false">关闭</option>
              <option value="true">开启（多脸检测 + 局部超分）</option>
            </select>
            <p class="hint">可增强检测到的人脸纹理；原视频里五官位置已变形或脸只有十几像素时，无法还原正确五官。</p></div>
        </div>

        <!-- 分段视频拼接 -->
        <details ref="seamDetails" class="character-library">
          <summary>分段视频拼接</summary>
          <p class="hint">自动模式会检查实际画面；只有接缝构图相近、处理后确实更平滑才补帧或匹配亮度。可用已生成的分段视频重新拼接，无需重新生成。</p>
          <div class="grid3">
            <div class="field"><label>拼接模式</label>
              <select :value="project.seam_mode" @change="onField($event, 'seam_mode')">
                <option value="auto">自动 · 推荐</option>
                <option value="conservative">保守 · 只对齐画面</option>
                <option value="manual">手动调节</option>
              </select></div>
            <div v-if="project.seam_mode === 'manual'" class="field"><label>最多跳过下一段开头</label>
              <select :value="String(project.seam_prefix_frames)" @change="onField($event, 'seam_prefix_frames')">
                <option value="0">不跳过</option><option value="1">1 帧</option>
                <option value="2">2 帧</option><option value="3">3 帧</option>
              </select></div>
            <div v-if="project.seam_mode === 'manual'" class="field"><label>最多运动补帧</label>
              <select :value="String(project.seam_motion_frames)" @change="onField($event, 'seam_motion_frames')">
                <option value="0">关闭</option><option value="2">2 帧</option><option value="3">3 帧</option>
              </select></div>
            <div v-if="project.seam_mode === 'manual'" class="field"><label>接缝亮度匹配</label>
              <select :value="String(project.seam_exposure)" @change="onField($event, 'seam_exposure')">
                <option value="true">自动检查并匹配</option>
                <option value="false">关闭</option>
              </select></div>
          </div>
          <div class="row" style="margin-top:14px">
            <button type="button" class="alt" :disabled="remerging" @click="remergeLatest">用当前设置重新拼接最近视频</button>
            <span class="muted" aria-live="polite">{{ remergeStatus }}</span>
            <a v-if="remergeLink" :href="remergeLink" target="_blank">查看新视频</a>
          </div>
        </details>

        <div class="divider"></div>
        <h3>全片统一提示词</h3>
        <p class="hint">填写全片固定要求；具体角色、地点、道具与画风资料请在“角色与设定库”管理。</p>
        <textarea rows="3" placeholder="例如：整体采用电影写实风格；所有镜头色调统一。"
          :value="project.shared_prompt" @input="onField($event, 'shared_prompt')"></textarea>

        <div class="divider"></div>
        <h3>全片共享参考素材</h3>
        <p class="hint">选择后会自动提供给项目中的所有镜头。</p>
        <div class="grid3">
          <div v-for="def in sharedDefs" :key="def.key" class="field">
            <label>{{ def.label }}</label>
            <div class="shared-ref-gallery">
              <button
                v-for="asset in sharedCandidates(def.kind)" :key="asset.id" type="button"
                class="shared-ref-choice" :class="{ selected: isShared(def.key, asset.id) }"
                :title="`${isShared(def.key, asset.id) ? '取消共享' : '设为全片共享'}：${asset.name}`"
                @click="toggleShared(def.key, asset.id)">
                <span class="shared-ref-preview">
                  <img v-if="asset.kind === 'image'" :src="'/media/assets/' + asset.id" loading="lazy" alt="">
                  <video v-else-if="asset.kind === 'video'" :src="'/media/assets/' + asset.id" muted preload="metadata"></video>
                  <span v-else>♫</span>
                </span>
                <strong>{{ (isShared(def.key, asset.id) ? '✓ ' : '') + asset.name }}</strong>
              </button>
              <span v-if="!sharedCandidates(def.kind).length" class="muted">尚无此类素材，请先上传。</span>
            </div>
            <small class="hint">直接点击缩略图选择或取消；绿色边框表示已提供给全部镜头。</small>
          </div>
        </div>
      </section>
    </div>
  </section>

  <section v-else class="settings-workspace">
    <div class="settings-scroll">
      <div class="panel" style="background:#fff;border-radius:14px">
        <h2>开始创作</h2>
        <p>新建或选择一个项目后，在这里统一设置画布、导出和增强方式。</p>
        <button type="button" @click="router.push('/projects')">前往项目列表</button>
      </div>
    </div>
  </section>

  <!-- Skill 内容预览弹窗 -->
  <dialog v-if="skillPreview" open class="skill-preview-dialog" @close="skillPreview = null">
    <div class="row between">
      <h2 id="skillPreviewTitle">{{ skillPreview.name }} · Skill 内容</h2>
      <button type="button" class="alt" @click="skillPreview = null">关闭</button>
    </div>
    <div class="skill-preview-body">
      <section v-for="file in skillPreview.files" :key="file.name">
        <strong>{{ file.name }}</strong>
        <pre>{{ file.content }}</pre>
      </section>
    </div>
  </dialog>
</template>
