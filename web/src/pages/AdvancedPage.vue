<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useProjectStore } from '../stores/project'
import { useUiStore } from '../stores/ui'
import { request } from '../api'
import { coerceField } from '../lib/fields'

const router = useRouter()
const projectStore = useProjectStore()
const ui = useUiStore()

const project = computed(() => projectStore.project)

const catalog = ref({ semantic_adapters: [] })
onMounted(async () => {
  try { catalog.value = await request('/api/models') } catch { /* 目录不可用时保留空列表 */ }
})

const adapterOptions = computed(() => [
  { value: '', label: '关闭' },
  ...(catalog.value.semantic_adapters || []).map(name => ({ value: name, label: name })),
])
const sizePreset = computed(() => {
  const w = Number(project.value?.output_width || 0)
  const h = Number(project.value?.output_height || 0)
  return w && h ? `${w}x${h}` : (w || h ? '' : '0x0')
})

function onField(event, key) {
  projectStore.set(key, coerceField(key, event.target.value, event.target))
}

function onSizePreset(event) {
  const [w, h] = String(event.target.value || '0x0').split('x').map(item => Number(item) || 0)
  projectStore.set('output_width', w)
  projectStore.set('output_height', h)
}
</script>

<template>
  <section v-if="project" class="settings-workspace">
    <div class="settings-scroll">
      <section class="panel settings-page" style="background:#fff;border-radius:14px">
        <div class="row between topbar">
          <div>
            <h2 style="margin:0">高级生成参数</h2>
            <p class="hint">这些选项会改变模型的采样、精修、显存和段间衔接行为；不确定时保留默认值。</p>
          </div>
          <button type="button" class="alt" @click="router.push('/settings')">返回项目设置</button>
        </div>
        <div class="divider"></div>
        <h3>H3 模型内部增强</h3>
        <div class="grid3">
          <div class="field"><label>语义桥适配器</label>
            <select :value="project.semantic_adapter || ''" @change="onField($event, 'semantic_adapter')">
              <option v-for="opt in adapterOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
            <p class="hint">需将适配器放入 ComfyUI/models/semantic_bridge。</p></div>
          <div class="field"><label>语义桥强度</label>
            <input type="number" min="0" max="1" step="0.01"
              :value="project.semantic_alpha" @input="onField($event, 'semantic_alpha')"></div>
          <div class="field"><label>语义桥幅度匹配</label>
            <select :value="String(project.semantic_magnitude_match)" @change="onField($event, 'semantic_magnitude_match')">
              <option value="true">开启</option><option value="false">关闭</option>
            </select></div>
          <div class="field"><label>镜头间显存清理</label>
            <select :value="String(project.clear_vram_between_shots)" @change="onField($event, 'clear_vram_between_shots')">
              <option value="false">关闭（生成更快）</option>
              <option value="true">开启（减少模型缓存显存占用）</option>
            </select>
            <p class="hint">每个镜头结束后请求 ComfyUI 卸载缓存模型；后续镜头可能需要重新加载权重。</p></div>
        </div>
      </section>

      <section class="panel settings-page" style="background:#fff;border-radius:14px;margin-top:16px">
        <h3 style="margin-top:0">清晰度</h3>
        <p class="hint">人物不清晰、AI 味重，先调这里。核心是三件事：画布用 H3 原生尺寸、二采用两阶段采样、成片尺寸单独指定。</p>
        <div class="grid3">
          <div class="field"><label>二采方式</label>
            <select :value="project.refine_mode" @change="onField($event, 'refine_mode')">
              <option value="off">关闭 · 只采样一次</option>
              <option value="selflift">两阶段 SelfLift（推荐 · 清晰度来源）</option>
            </select>
            <p class="hint">两阶段 = 低清前缀 → H3 学习式 latent 放大 → 高清续采；音轨锁定首采，参考图引导随分辨率重建。</p></div>
          <div class="field"><label>成片尺寸（宽 × 高）</label>
            <div class="row">
              <input type="number" min="0" step="2" style="width:46%"
                :value="project.output_width" @input="onField($event, 'output_width')">
              <input type="number" min="0" step="2" style="width:46%"
                :value="project.output_height" @input="onField($event, 'output_height')">
            </div>
            <p class="hint">填 0 表示不重采样。H3 原生画布 1344×768（1.75:1），超分后是 1890×1080；填 1920×1080 会等比放大后<b>居中裁切</b>到正好这个尺寸（不拉伸、不加黑边）。</p></div>
          <div class="field"><label>成片方向预设</label>
            <select :value="sizePreset" @change="onSizePreset">
              <option value="">自定义</option>
              <option value="1920x1080">1920 × 1080 横屏</option>
              <option value="1080x1920">1080 × 1920 竖屏</option>
              <option value="2560x1440">2560 × 1440 横屏</option>
              <option value="1440x2560">1440 × 2560 竖屏</option>
              <option value="0x0">不重采样</option>
            </select>
            <p class="hint">方向要和项目画布一致，否则竖屏片会被裁成横屏。</p></div>
        </div>
        <p class="hint">画布请保持在 1344×768（横屏）或 768×1344（竖屏）附近：超出 H3 原生画布，模型会跑在分布之外，皮肤和织物发蜡发平，正是「AI 味重」的来源。需要更大分辨率就用超分补，不要把画布拉大。</p>
      </section>
    </div>
  </section>

  <section v-else class="settings-workspace">
    <div class="settings-scroll">
      <div class="panel" style="background:#fff;border-radius:14px">
        <h2>未选择项目</h2>
        <p>请先新建或选择一个项目。</p>
        <button type="button" @click="router.push('/projects')">前往项目列表</button>
      </div>
    </div>
  </section>
</template>
