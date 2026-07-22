import { pdfjs } from 'react-pdf'

// worker 从本地 bundle 加载(构建期由打包器把 pdfjs-dist 的 worker 产出为
// 静态资源),版本与 react-pdf 内置的 pdfjs 严格一致;不依赖任何 CDN,
// 纯内网/离线部署可用。
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

export function ensurePdfWorker(): void {
  // Already configured above — this is a no-op guard for call sites
}

export { pdfjs }
