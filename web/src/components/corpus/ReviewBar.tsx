'use client'

// 复核操作(L3-P3):待复核条目的 原因徽标 + 通过/修改标签/不收录。
// C1 法律、R1 数据出境按规范强制逐条复核,批量通过时自动跳过。

import * as React from 'react'
import { Check, Pencil, Trash2 } from 'lucide-react'
import { apiFetch } from '@/lib/api'
import { useUserStore } from '@/stores'
import type { CorpusMeta } from '@/lib/corpus'

export function reviewReasons(meta: CorpusMeta): string[] {
  const reasons: string[] = []
  if (meta.domain === 'C1' || meta.domain_ext.includes('C1')) reasons.push('法律·强制复核')
  if (meta.rule_type.includes('R1')) reasons.push('数据出境·强制复核')
  if (meta.domain === 'X9') reasons.push('X9 码表外兜底')
  if (meta.confidence !== null && meta.confidence < 0.5) reasons.push('低置信')
  return reasons
}

export function isMandatoryReview(meta: CorpusMeta): boolean {
  return meta.domain === 'C1' || meta.domain_ext.includes('C1') || meta.rule_type.includes('R1')
}

export interface CodetableOptions {
  stages: { code: string; name: string }[]
  domains: { code: string; name: string }[]
  genres: { code: string; name: string }[]
  rules: { code: string; name: string }[]
  evidence: { code: string; name: string }[]
  origins: { code: string; name: string }[]
  timeliness: { code: string; name: string }[]
}

export async function reviewEntry(
  token: string, docId: string, action: 'approve' | 'update' | 'exclude',
  labels?: Record<string, unknown>, note?: string,
) {
  return apiFetch(`/v1/corpus/entries/${docId}/review`, token, {
    method: 'POST',
    body: JSON.stringify({ action, labels, note: note || '' }),
  })
}

export function ReviewBadges({ meta }: { meta: CorpusMeta }) {
  const reasons = reviewReasons(meta)
  if (reasons.length === 0) return null
  return (
    <>
      {reasons.map((r) => (
        <span key={r} className="shrink-0 rounded-full border border-red-500/30 px-1.5 py-px text-[10px] leading-4 text-red-600 dark:text-red-500">
          {r}
        </span>
      ))}
    </>
  )
}

export function ReviewActions({ docId, meta, onDone, onEdit }: {
  docId: string
  meta: CorpusMeta
  onDone: () => void
  onEdit: () => void
}) {
  const token = useUserStore((s) => s.accessToken)
  const [busy, setBusy] = React.useState(false)

  const act = async (action: 'approve' | 'exclude') => {
    if (!token || busy) return
    if (action === 'exclude' && !window.confirm('确定不收录该条目?条目文件将被移除,源文档标记为不收录。')) return
    setBusy(true)
    try {
      await reviewEntry(token, docId, action)
      onDone()
    } catch { /* 留在队列里,下次刷新可重试 */ } finally {
      setBusy(false)
    }
  }

  return (
    <span className="flex shrink-0 items-center gap-1" onClick={(e) => e.stopPropagation()}>
      <button onClick={() => act('approve')} disabled={busy} title="通过(转已入库)"
        className="rounded-md border border-border p-1 text-emerald-600 hover:bg-accent transition-colors cursor-pointer disabled:opacity-50">
        <Check className="size-3.5" />
      </button>
      <button onClick={onEdit} disabled={busy} title="修改标签"
        className="rounded-md border border-border p-1 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors cursor-pointer disabled:opacity-50">
        <Pencil className="size-3.5" />
      </button>
      <button onClick={() => act('exclude')} disabled={busy} title="不收录"
        className="rounded-md border border-border p-1 text-muted-foreground hover:bg-accent hover:text-destructive transition-colors cursor-pointer disabled:opacity-50">
        <Trash2 className="size-3.5" />
      </button>
    </span>
  )
}

export function ReviewDialog({ docId, meta, options, onClose, onDone }: {
  docId: string
  meta: CorpusMeta
  options: CodetableOptions
  onClose: () => void
  onDone: () => void
}) {
  const token = useUserStore((s) => s.accessToken)
  const [stage, setStage] = React.useState(meta.stage)
  const [domain, setDomain] = React.useState(meta.domain)
  const [genre, setGenre] = React.useState(meta.genre)
  const [rules, setRules] = React.useState(meta.rule_type.join('/'))
  const [evidence, setEvidence] = React.useState(meta.evidence)
  const [origin, setOrigin] = React.useState(meta.origin)
  const [timeliness, setTimeliness] = React.useState(meta.timeliness)
  const [note, setNote] = React.useState('')
  const [busy, setBusy] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const submit = async () => {
    if (!token || busy) return
    setBusy(true)
    setError(null)
    try {
      await reviewEntry(token, docId, 'update', {
        stage, domain, genre,
        rule_type: rules.split(/[/、,,]/).map((s) => s.trim()).filter(Boolean),
        evidence, origin, timeliness,
      }, note)
      onDone()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存失败')
    } finally {
      setBusy(false)
    }
  }

  const sel = 'w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring'
  const field = (label: string, node: React.ReactNode) => (
    <label className="block text-xs text-muted-foreground">
      {label}
      <div className="mt-1">{node}</div>
    </label>
  )
  const options4 = (list: { code: string; name: string }[]) =>
    list.map((o) => <option key={o.code} value={o.code}>{o.code === o.name ? o.name : `${o.code} ${o.name}`}</option>)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div className="w-[520px] max-w-[92vw] rounded-lg border border-border bg-background p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}>
        <h3 className="text-sm font-medium">修改标签 — <code className="text-xs">{meta.entry_id}</code></h3>
        <div className="mt-4 grid grid-cols-2 gap-3">
          {field('阶段', <select className={sel} value={stage} onChange={(e) => setStage(e.target.value)}>{options4(options.stages)}</select>)}
          {field('服务大类', <select className={sel} value={domain} onChange={(e) => setDomain(e.target.value)}>{options4(options.domains)}</select>)}
          {field('体裁', <select className={sel} value={genre} onChange={(e) => setGenre(e.target.value)}>{options4(options.genres)}</select>)}
          {field('隐性规则(/分隔)', <input className={sel} value={rules} onChange={(e) => setRules(e.target.value)} placeholder="R1/R2" />)}
          {field('证据强度', <select className={sel} value={evidence} onChange={(e) => setEvidence(e.target.value)}>{options4(options.evidence)}</select>)}
          {field('来源域', <select className={sel} value={origin} onChange={(e) => setOrigin(e.target.value)}>{options4(options.origins)}</select>)}
          {field('时效', <select className={sel} value={timeliness} onChange={(e) => setTimeliness(e.target.value)}>{options4(options.timeliness)}</select>)}
          {field('复核备注', <input className={sel} value={note} onChange={(e) => setNote(e.target.value)} placeholder="改判理由(入审计)" />)}
        </div>
        {error && <p className="mt-3 text-xs text-destructive">{error}</p>}
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="rounded-md px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent transition-colors cursor-pointer">取消</button>
          <button onClick={submit} disabled={busy}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent transition-colors cursor-pointer disabled:opacity-50">
            {busy ? '保存中…' : '保存并转已入库'}
          </button>
        </div>
      </div>
    </div>
  )
}
