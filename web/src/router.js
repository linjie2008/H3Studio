import { createRouter, createWebHistory } from 'vue-router'
import ProjectsPage from './pages/ProjectsPage.vue'
import SettingsPage from './pages/SettingsPage.vue'
import AdvancedPage from './pages/AdvancedPage.vue'
import PlaceholderPage from './pages/PlaceholderPage.vue'

// 基路径 /app/：旧 UI 仍在 `/`，新 UI 从 `/app` 进入（服务端由 server/routes 提供）。
// 尚未迁移的工作区先落到占位页，并给出跳回旧 UI 的链接。
const router = createRouter({
  history: createWebHistory('/app/'),
  routes: [
    { path: '/', redirect: '/projects' },
    { path: '/projects', name: 'projects', component: ProjectsPage },
    { path: '/settings', name: 'settings', component: SettingsPage },
    { path: '/advanced', name: 'advanced', component: AdvancedPage },
    { path: '/:workspace(create|characters|timeline|videos|workflow)',
      component: PlaceholderPage, props: true },
    { path: '/:pathMatch(.*)*', redirect: '/projects' },
  ],
})

export default router
