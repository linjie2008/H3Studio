<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

// 未迁移工作区的占位页：说明进度并提供跳回旧 UI 的入口。
const route = useRoute()
const router = useRouter()

const META = {
  create: { title: '创作工作台', hint: '镜头编辑、素材绑定、提示词优化与生成。' },
  characters: { title: '角色与设定库', hint: '角色、地点、道具与画风档案。' },
  timeline: { title: '镜头时间轴', hint: '拖放排序、时长调整、场景分组与批量操作。' },
  videos: { title: '生成视频库', hint: '预览、下载、删除生成的视频。' },
  workflow: { title: 'ComfyUI 节点工作流', hint: '按当前项目参数绘制的执行图。' },
}
const meta = computed(() => META[route.params.workspace] || { title: '页面迁移中', hint: '' })
const oldUiUrl = computed(() => '/?workspace=' + route.params.workspace)
</script>

<template>
  <section class="settings-workspace">
    <div class="settings-scroll">
      <div class="panel" style="background:#fff;border-radius:14px;max-width:640px">
        <span class="eyebrow">尚未迁移到新版</span>
        <h2>{{ meta.title }}</h2>
        <p class="hint">{{ meta.hint }}</p>
        <p>此工作区仍在旧版界面中提供。项目设置已可在此新版体验；其余页面完成后会逐步替换。</p>
        <div class="row" style="margin-top:16px">
          <a :href="oldUiUrl"><button type="button">在旧版界面打开</button></a>
          <button type="button" class="alt" @click="router.push('/settings')">返回项目设置</button>
        </div>
      </div>
    </div>
  </section>
</template>
