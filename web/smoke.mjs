// 浏览器冒烟测试：新 Vue 工作台 (/app) 的渲染与保存管线。
// 只创建一个专用测试项目，验证完成后删除，不触碰既有项目。
import { chromium } from 'playwright'

const BASE = 'http://127.0.0.1:8768'
const TEST_PROJECT_NAME = 'ZZ冒烟测试-自动删除'

async function api(path, method = 'GET', body) {
  const resp = await fetch(BASE + path, {
    method,
    headers: { 'Content-Type': 'application/json', Origin: BASE },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!resp.ok) throw new Error(`${method} ${path} -> HTTP ${resp.status}`)
  return resp.json()
}

async function cleanup() {
  const projects = await api('/api/projects')
  // 用前缀匹配：第 4 步会把测试项目改名为 `-改`，精确名匹配会漏删。
  for (const p of projects.filter(p => p.name.startsWith(TEST_PROJECT_NAME))) {
    await api(`/api/projects/${p.id}`, 'DELETE')
    console.log(`cleaned up leftover test project ${p.id}`)
  }
}

const consoleErrors = []
await cleanup()

const browser = await chromium.launch()
const page = await browser.newPage()
page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()) })
page.on('pageerror', e => consoleErrors.push(String(e)))
page.on('dialog', d => d.accept(TEST_PROJECT_NAME))

// 1. 项目主页渲染
await page.goto(`${BASE}/app`, { waitUntil: 'networkidle' })
const cardCount = await page.locator('.project-card').count()
if (cardCount < 1) throw new Error('项目主页未渲染任何项目卡片')
console.log(`1. 项目主页 OK（${cardCount} 张卡片）`)

// 2. 新建项目 → 跳设置页
await page.getByRole('button', { name: '＋ 新建项目' }).click()
await page.waitForURL('**/settings', { timeout: 10000 })
await page.waitForSelector("h2:has-text('项目设置')", { timeout: 10000 })
console.log('2. 新建项目并跳转设置页 OK')

const projects = await api('/api/projects')
const testProject = projects.find(p => p.name === TEST_PROJECT_NAME)
if (!testProject) throw new Error('测试项目未创建')
const pid = testProject.id
const originalVersion = testProject._data_version

// 3. 设置页字段渲染
const selectCount = await page.locator('.grid3 select').count()
if (selectCount < 10) throw new Error(`设置页下拉框数量异常: ${selectCount}`)
const nameValue = await page.locator('.grid3 input').first().inputValue()
if (nameValue !== TEST_PROJECT_NAME) throw new Error(`项目名称未回显: ${nameValue}`)
console.log(`3. 设置页字段渲染 OK（${selectCount} 个下拉框）`)

// 4. 改名 → 防抖自动保存 → API 验证落库
await page.locator('.grid3 input').first().fill(TEST_PROJECT_NAME + '-改')
await page.waitForFunction(async ({ pid, v }) => {
  const p = await (await fetch(`/api/projects/${pid}`)).json()
  return p.name.endsWith('-改') && p._data_version > v
}, { pid, v: originalVersion }, { timeout: 8000 })
console.log('4. 防抖自动保存 → 服务端已落库 OK')

// 5. 画布预设：0.98MP → 1344×768
await page.locator('.grid3 select').nth(1).selectOption('0.98')
await page.waitForFunction(async pid => {
  const p = await (await fetch(`/api/projects/${pid}`)).json()
  return p.width === 1344 && p.height === 768
}, pid, { timeout: 8000 })
console.log('5. 画布预设（0.98MP → 1344×768）OK')

// 6. 保存指示器回到「已保存」
await page.waitForFunction(
  () => document.getElementById('saveState').textContent === '已保存',
  { timeout: 8000 })
console.log('6. 保存状态指示器 OK')

// 7. 高级参数页
await page.getByRole('button', { name: '高级模型参数' }).click()
await page.waitForSelector("h2:has-text('高级生成参数')", { timeout: 8000 })
console.log('7. 高级参数页 OK')

// 8. 占位页 + 旧 UI 跳转链接（侧栏按钮可访问名为“图标+文字”，用非精确匹配）
await page.locator(".app-rail button", { hasText: '镜头' }).click()
await page.waitForSelector("h2:has-text('镜头时间轴')", { timeout: 8000 })
const oldLink = await page.locator("a[href='/?workspace=timeline']").count()
if (oldLink !== 1) throw new Error('占位页缺少旧 UI 跳转链接')
console.log('8. 占位页与旧 UI 跳转链接 OK')

// 9. 旧 UI 未被破坏
const oldResp = await page.request.get(`${BASE}/`)
if (oldResp.status() !== 200 || !(await oldResp.text()).includes('漫剧工作台'))
  throw new Error('旧版界面受损')
console.log("9. 旧版界面 '/' 未受影响 OK")

await browser.close()
await cleanup()

const realErrors = consoleErrors.filter(e => !e.includes('favicon'))
if (realErrors.length) {
  console.error('console errors:')
  for (const e of realErrors.slice(0, 10)) console.error('  ', e.slice(0, 300))
  console.error(`\nFAILED: ${realErrors.length} console errors`)
  process.exit(1)
}
console.log('\nALL BROWSER SMOKE TESTS PASSED')
