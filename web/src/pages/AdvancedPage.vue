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

const catalog = ref({ semantic_adapters: [], latent_models: [] })
onMounted(async () => {
  try { catalog.value = await request('/api/models') } catch { /* 目录不可用时保留空列表 */ }
})

const adapterOptions = computed(() => [
  { value: '', label: '关闭' },
  ...(catalog.value.semantic_adapters || []).map(name => ({ value: name, label: name })),
])
const latentModelOptions = computed(() => {
  const models = catalog.value.latent_models || []
  const current = project.value?.latent_model
  const options = models.map(name => ({ value: name, label: name }))
  if (current && !models.includes(current)) options.unshift({ value: current, label: current })
  return options
})

function onField(event, key) {
  projectStore.set(key, coerceField(key, event.target.value, event.target))
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
          <div class="field"><label>SelfLift 渐进一采</label>
            <select :value="String(project.selflift)" @change="onField($event, 'selflift')">
              <option value="false">关闭</option><option value="true">开启</option>
            </select></div>
          <div class="field"><label>SelfLift 低清比例</label>
            <input type="number" min="0.25" max="1" step="0.05"
              :value="project.selflift_scale" @input="onField($event, 'selflift_scale')"></div>
          <div class="field"><label>SelfLift 高清步数</label>
            <input type="number" min="1" max="7" step="1"
              :value="project.selflift_highres_steps" @input="onField($event, 'selflift_highres_steps')"></div>
          <div class="field"><label>SelfLift 像素锚混合 ρ</label>
            <input type="number" min="0" max="1" step="0.05"
              :value="project.selflift_rho" @input="onField($event, 'selflift_rho')"></div>
          <div class="field"><label>SelfLift 高清阶段空间分块</label>
            <select :value="String(project.selflift_tiling)" @change="onField($event, 'selflift_tiling')">
              <option value="false">关闭</option><option value="true">开启</option>
            </select></div>
          <div class="field"><label>空间分块数量</label>
            <input type="number" min="1" max="8" step="1"
              :value="project.selflift_tile_count" @input="onField($event, 'selflift_tile_count')"></div>
          <div class="field"><label>空间分块重叠像素</label>
            <input type="number" min="0" max="2048" step="1"
              :value="project.selflift_tile_overlap" @input="onField($event, 'selflift_tile_overlap')"></div>
          <div class="field"><label>H3 3D latent 权重</label>
            <select :value="project.latent_model" @change="onField($event, 'latent_model')">
              <option v-for="opt in latentModelOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select></div>
          <div class="field"><label>H3 二采 / latent 放大</label>
            <select :value="project.refine_mode" @change="onField($event, 'refine_mode')">
              <option value="off">关闭</option>
              <option value="same">原画布精修</option>
              <option value="latent">3D latent 放大后精修</option>
              <option value="latent_only">只做 3D latent 放大</option>
            </select></div>
          <div class="field"><label>二采步数</label>
            <input type="number" min="1" max="16" step="1"
              :value="project.refine_steps" @input="onField($event, 'refine_steps')"></div>
          <div class="field"><label>二采轮数</label>
            <input type="number" min="1" max="8" step="1"
              :value="project.refine_passes" @input="onField($event, 'refine_passes')"></div>
          <div class="field"><label>二采重绘幅度</label>
            <input type="number" min="0.01" max="1" step="0.01"
              :value="project.refine_denoise" @input="onField($event, 'refine_denoise')"></div>
          <div class="field"><label>3D latent 目标画布</label>
            <div class="row">
              <input type="number" min="32" step="32" style="width:46%"
                :value="project.refine_width" @input="onField($event, 'refine_width')">
              <input type="number" min="32" step="32" style="width:46%"
                :value="project.refine_height" @input="onField($event, 'refine_height')">
            </div></div>
        </div>
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
