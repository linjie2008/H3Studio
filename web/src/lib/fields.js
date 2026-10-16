// 从旧 UI 提取的画布与字段类型逻辑，供设置页与工作台共用。

export const CANVAS_MP_PRESETS = [
  ['0.2', 608, 352], ['0.3', 736, 416], ['0.4', 864, 480], ['0.5', 960, 544],
  ['0.6', 1056, 608], ['0.7', 1152, 640], ['0.8', 1216, 672], ['0.9', 1280, 736],
  ['0.98', 1344, 768], ['1.0', 1376, 768], ['1.2', 1504, 832], ['1.5', 1664, 928],
  ['1.8', 1824, 1024], ['2.0', 1920, 1088],
]

export function canvasPresetFor(width, height) {
  return CANVAS_MP_PRESETS.find(
    ([, w, h]) => (w === Number(width) && h === Number(height)) ||
                  (h === Number(width) && w === Number(height))
  )?.[0] || ''
}

// 与旧 UI applyCanvasPreset 一致：默认按当前宽高判断竖横屏。
export function applyCanvasPreset(project, mp, portrait = Number(project?.height) > Number(project?.width)) {
  const row = CANVAS_MP_PRESETS.find(item => item[0] === mp)
  if (!row) return false
  project.width = portrait ? row[2] : row[1]
  project.height = portrait ? row[1] : row[2]
  return true
}

// 与旧 UI alignCanvasDimension 一致：就近对齐到 32 的倍数。
export function alignCanvasDimension(value, fallback) {
  const number = Math.max(32, Number.parseInt(value) || fallback)
  const lower = Math.max(32, Math.floor(number / 32) * 32)
  const upper = lower + 32
  return number - lower <= upper - number ? lower : upper
}

// 与旧 UI change 处理器一致的字段类型转换表。
const NUMBER_KEYS = new Set([
  'width', 'height', 'upscale_resolution', 'upscale_max_resolution', 'continuity_frames',
  'semantic_alpha', 'selflift_scale', 'selflift_highres_steps', 'selflift_rho',
  'selflift_tile_count', 'selflift_tile_overlap', 'refine_steps', 'refine_passes',
  'refine_denoise', 'refine_width', 'refine_height', 'seam_prefix_frames',
  'seam_motion_frames',
])
const BOOL_KEYS = new Set([
  'upscale', 'confirm_upscale', 'face_refine', 'semantic_magnitude_match', 'selflift',
  'selflift_tiling', 'clear_vram_between_shots', 'sage_attention', 'continuous_take',
  'identity_first_on_switch', 'seam_exposure',
])

export function coerceField(key, rawValue, element) {
  if (element?.type === 'checkbox') return element.checked
  if (NUMBER_KEYS.has(key)) return +rawValue
  if (BOOL_KEYS.has(key)) return rawValue === 'true'
  return rawValue
}

export function isProjectField(key) {
  return NUMBER_KEYS.has(key) || BOOL_KEYS.has(key) || typeof key === 'string'
}
