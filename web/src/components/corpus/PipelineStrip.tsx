'use client'

// 语料分类流水线状态条(仅本地模式):待分类 / 运行进度 / 今日入库 / 失败,
// 附"立即分类/停止分类"按钮(停止会同时关闭自动分类,避免队列被自动重启)。
// 轮询 /v1/corpus/pipeline/status,运行中加密轮询。

import * as React from 'react'
import { Loader2, Play, Square } from 'lucide-react'
import { apiFetch } from '@/lib/api'
import { useUserStore } from '@/stores'

interface PipelineStatus {
  running: boolean
  progress: { done: number; total: number } | null
  auto: { enabled: boolean; interval: number }
  counts: { pending: number; imported: number; imported_today: number; excluded: number; failed: number }
}

const isLocal = process.env.NEXT_PUBLIC_MODE === 'local'

export function PipelineStrip() {
  const token = useUserStore((s) => s.accessToken)
  const [status, setStatus] = React.useState<PipelineStatus | null>(null)
  const [starting, setStarting] = React.useState(false)
  const [stopping, setStopping] = React.useState(false)

  const refresh = React.useCallback(() => {
    if (!token) return
    apiFetch<PipelineStatus>('/v1/corpus/pipeline/status', token)
      .then(setStatus)
      .catch(() => {})
  }, [token])

  React.useEffect(() => {
    if (!isLocal) return
    refresh()
    const t = setInterval(refresh, status?.running ? 3000 : 15000)
    return () => clearInterval(t)
  }, [refresh, status?.running])

  if (!isLocal || !status) return null
  const { counts } = status
  if (!status.running && counts.pending === 0 && counts.imported === 0 && counts.failed === 0) return null

  const startRun = async () => {
    if (!token || starting || status.running) return
    setStarting(true)
    try {
      await apiFetch('/v1/corpus/pipeline/run', token, { method: 'POST', body: JSON.stringify({}) })
      refresh()
    } catch { /* 状态条下轮刷新即可 */ } finally {
      setStarting(false)
    }
  }

  const stopRun = async () => {
    if (!token || stopping) return
    setStopping(true)
    try {
      await apiFetch('/v1/corpus/pipeline/stop', token, { method: 'POST' })
      refresh()
    } catch { /* 状态条下轮刷新即可 */ } finally {
      setStopping(false)
    }
  }

  return (
    <div className="shrink-0 border-b border-border bg-muted/30 px-4 py-1.5 flex items-center gap-4 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">分类流水线</span>
      {status.running ? (
        <span className="flex items-center gap-1.5 text-foreground">
          <Loader2 className="size-3 animate-spin" />
          分类中
          {status.progress && status.progress.total > 0 && (
            <span className="tabular-nums">{status.progress.done}/{status.progress.total}</span>
          )}
        </span>
      ) : (
        <span>待分类 <span className="tabular-nums text-foreground">{counts.pending}</span></span>
      )}
      <span>今日入库 <span className="tabular-nums text-foreground">{counts.imported_today}</span></span>
      {counts.failed > 0 && (
        <span className="text-destructive">失败 <span className="tabular-nums">{counts.failed}</span></span>
      )}
      <span className="text-muted-foreground/60">自动分类{status.auto.enabled ? '已开启' : '未开启'}</span>
      <div className="flex-1" />
      {!status.running && counts.pending > 0 && (
        <button
          onClick={startRun}
          disabled={starting}
          className="flex items-center gap-1 rounded-md border border-border px-2 py-0.5 hover:bg-accent hover:text-foreground transition-colors cursor-pointer disabled:opacity-50"
        >
          <Play className="size-3" />
          立即分类
        </button>
      )}
      {status.running && (
        <button
          onClick={stopRun}
          disabled={stopping}
          className="flex items-center gap-1 rounded-md border border-border px-2 py-0.5 text-destructive hover:bg-destructive/10 transition-colors cursor-pointer disabled:opacity-50"
        >
          <Square className="size-3" />
          {stopping ? '停止中…' : '停止分类'}
        </button>
      )}
    </div>
  )
}
