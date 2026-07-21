// NEXT_PUBLIC_* 在构建期烧入 bundle,无法随部署改变。Docker 镜像的入口脚本
// 会在启动时生成 /__llmwiki_env.js(内容来自 PUBLIC_API_URL 环境变量),
// 使改了宿主机端口映射的部署无需重新构建即可让浏览器找到 API。
// 读取顺序:运行时注入 → 构建期 NEXT_PUBLIC_API_URL → 默认 localhost:8000。

declare global {
  interface Window {
    __LLMWIKI_ENV__?: { API_URL?: string }
  }
}

export function apiUrl(): string {
  if (typeof window !== 'undefined' && window.__LLMWIKI_ENV__?.API_URL) {
    return window.__LLMWIKI_ENV__.API_URL.replace(/\/+$/, '')
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
}
