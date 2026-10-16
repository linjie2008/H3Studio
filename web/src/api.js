// 与旧 UI request() 语义一致：非 2xx 抛出后端 error 文案（含 409/400 冲突）。
export async function request(path, method = 'GET', body) {
  const response = await fetch(path, {
    method,
    headers: body && !(body instanceof Blob) ? { 'Content-Type': 'application/json' } : {},
    body: body instanceof Blob ? body : body ? JSON.stringify(body) : undefined,
  })
  if (!response.ok) {
    let data = null
    try { data = await response.json() } catch { /* 非 JSON 错误体 */ }
    throw new Error(data?.error || `HTTP ${response.status}`)
  }
  return response.json()
}
