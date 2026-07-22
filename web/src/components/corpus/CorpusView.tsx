'use client'

// 语料库视图 — faceted browser over classified corpus entries (八维标注).
// Knowledge view: facet filters + 阶段×大类层 coverage matrix + entry table.
// Business view: 7 类需求 → 27 场景 navigation derived from the same entries.

import * as React from 'react'
import { PipelineStrip } from './PipelineStrip'
import {
  CodetableOptions, ReviewActions, ReviewBadges, ReviewDialog,
  isMandatoryReview, reviewEntry,
} from './ReviewBar'
import { useRouter } from 'next/navigation'
import { ArrowLeft, LayoutGrid, ListFilter, RefreshCw, X } from 'lucide-react'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import { apiFetch } from '@/lib/api'
import { useUserStore } from '@/stores'
import { useKBDocuments } from '@/hooks/useKBDocuments'
import {
  buildBusinessView,
  buildCoverageGrid,
  businessPendingCount,
  collectCorpusEntries,
  collectFacetOptions,
  collectWikiRollups,
  computeKpis,
  entryMatches,
  FACET_LABELS,
  filterEntries,
  LAYER_LABELS,
  LAYERS,
  reviewStatus,
  STAGE_LABELS,
  STAGES,
  type CorpusEntry,
  type FacetKey,
  type FacetSelections,
} from '@/lib/corpus'

interface GraphEdge {
  source: string
  target: string
  type: string
}

const PANEL_FACETS: FacetKey[] = [
  'stage', 'domain', 'genre', 'rule', 'evidence', 'origin',
  'dept', 'geo', 'industry', 'timeliness', 'state',
]

const MAX_CHIPS = 12

export function CorpusView({ kbId, kbSlug, kbName }: { kbId: string; kbSlug: string; kbName: string }) {
  const router = useRouter()
  const token = useUserStore((s) => s.accessToken)
  const { documents, loading, refetchDocuments } = useKBDocuments(kbId)
  const [view, setView] = React.useState<'knowledge' | 'business'>('knowledge')
  const [selections, setSelections] = React.useState<FacetSelections>({})
  const [citedDocIds, setCitedDocIds] = React.useState<Set<string> | null>(null)
  const [codetable, setCodetable] = React.useState<CodetableOptions | null>(null)
  const [editing, setEditing] = React.useState<CorpusEntry | null>(null)
  const [batchBusy, setBatchBusy] = React.useState(false)

  // 复核对话框的码表取值(本地模式;端点不存在时静默,操作按钮不受影响)
  React.useEffect(() => {
    if (!token || codetable) return
    apiFetch<CodetableOptions>('/v1/corpus/codetable', token)
      .then(setCodetable)
      .catch(() => {})
  }, [token, codetable])

  const onReviewed = React.useCallback(() => { refetchDocuments() }, [refetchDocuments])

  const batchApprove = React.useCallback(async (targets: CorpusEntry[]) => {
    if (!token || batchBusy) return
    setBatchBusy(true)
    try {
      for (const e of targets) {
        await reviewEntry(token, e.doc.id, 'approve', undefined, '批量通过(低风险)')
      }
    } catch { /* 剩余条目留在队列 */ } finally {
      setBatchBusy(false)
      refetchDocuments()
    }
  }, [token, batchBusy, refetchDocuments])

  // Citation targets from the reference graph power the 引用溯源率 KPI.
  React.useEffect(() => {
    let cancelled = false
    apiFetch<{ edges: GraphEdge[] }>(`/v1/knowledge-bases/${kbId}/graph`, token ?? '')
      .then((res) => {
        if (cancelled) return
        const cited = new Set<string>()
        for (const edge of res.edges ?? []) {
          if (edge.type === 'cites') cited.add(edge.target)
        }
        setCitedDocIds(cited)
      })
      .catch(() => {
        // KPI simply shows — when the graph is unavailable.
      })
    return () => {
      cancelled = true
    }
  }, [kbId, token])

  const entries = React.useMemo(() => collectCorpusEntries(documents), [documents])
  const filtered = React.useMemo(() => filterEntries(entries, selections), [entries, selections])
  const coverage = React.useMemo(() => buildCoverageGrid(entries), [entries])
  const businessClasses = React.useMemo(() => buildBusinessView(entries), [entries])
  const pendingBusiness = React.useMemo(() => businessPendingCount(entries), [entries])
  const kpis = React.useMemo(
    () => computeKpis(entries, citedDocIds, undefined, collectWikiRollups(documents)),
    [entries, citedDocIds, documents],
  )

  const toggleFacet = React.useCallback((key: FacetKey, value: string) => {
    setSelections((prev) => {
      const next = { ...prev }
      if (next[key] === value) delete next[key]
      else next[key] = value
      return next
    })
  }, [])

  const openEntry = React.useCallback(
    (entry: CorpusEntry) => {
      // 文件视图的 ?doc= 约定为 document_number(整数),不是 UUID
      const search = entry.doc.document_number != null ? `?doc=${entry.doc.document_number}` : ''
      router.push(`/wikis/${kbSlug}/files${search}`)
    },
    [router, kbSlug],
  )

  const reprocessEntry = React.useCallback(async (entry: CorpusEntry) => {
    if (!token) return
    try {
      await apiFetch(`/v1/corpus/entries/${entry.doc.id}/reprocess`, token, { method: 'POST' })
      toast.success('已开始重新识别:重新提取源文件后将自动重新分类入库')
      setTimeout(refetchDocuments, 800)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '重新识别失败')
    }
  }, [token, refetchDocuments])

  const activeSelections = Object.entries(selections).filter(([, v]) => v) as Array<[FacetKey, string]>

  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      {/* Header */}
      <div className="shrink-0 border-b border-border px-4 py-2.5 flex items-center gap-3">
        <button
          onClick={() => router.push(`/wikis/${kbSlug}`)}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
        >
          <ArrowLeft className="size-3.5" />
          {kbName}
        </button>
        <span className="text-xs text-muted-foreground/40">/</span>
        <span className="text-sm font-medium">语料库</span>
        <span className="text-xs text-muted-foreground tabular-nums">
          {filtered.length === entries.length ? `${entries.length} 条` : `${filtered.length} / ${entries.length} 条`}
        </span>
        <div className="flex-1" />
        <div className="flex items-center rounded-md border border-border p-0.5 text-xs">
          <button
            onClick={() => setView('knowledge')}
            className={cn(
              'px-2.5 py-1 rounded-[5px] transition-colors cursor-pointer flex items-center gap-1.5',
              view === 'knowledge' ? 'bg-accent text-foreground' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            <ListFilter className="size-3" />
            知识视图
          </button>
          <button
            onClick={() => setView('business')}
            className={cn(
              'px-2.5 py-1 rounded-[5px] transition-colors cursor-pointer flex items-center gap-1.5',
              view === 'business' ? 'bg-accent text-foreground' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            <LayoutGrid className="size-3" />
            业务视图
          </button>
        </div>
      </div>

      <PipelineStrip />
      {editing && codetable && (
        <ReviewDialog docId={editing.doc.id} meta={editing.meta} options={codetable}
          onClose={() => setEditing(null)} onDone={onReviewed} />
      )}

      {loading ? (
        <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">加载中…</div>
      ) : entries.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
          <p className="text-sm font-medium">还没有已分类语料</p>
          <p className="text-xs text-muted-foreground max-w-md">
            点击上方状态条的<span className="font-medium text-foreground">立即分类</span>让流水线自动标注工作区文件
            (先在 设置 → 语料分类流水线 配好 LLM 端点),或用{' '}
            <code className="rounded bg-muted px-1">corpus/import_annotations.py</code> 导入现成的八维标注明细。
          </p>
        </div>
      ) : view === 'business' ? (
        <BusinessPane
          classes={businessClasses}
          pending={pendingBusiness}
          selections={selections}
          onSelectScene={(code) => {
            setSelections((prev) => (prev.business === code ? {} : { business: code }))
            setView('knowledge')
          }}
        />
      ) : (
        <div className="flex flex-1 min-h-0">
          {/* Facet panel */}
          <div className="w-56 shrink-0 border-r border-border overflow-y-auto px-3 py-3 space-y-4">
            {PANEL_FACETS.map((key) => (
              <FacetGroup
                key={key}
                facet={key}
                entries={entries}
                selections={selections}
                onToggle={toggleFacet}
              />
            ))}
          </div>

          {/* Main pane */}
          <div className="flex-1 min-w-0 overflow-y-auto">
            <div className="px-4 py-3 space-y-4">
              {activeSelections.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  {activeSelections.map(([key, value]) => (
                    <button
                      key={key}
                      onClick={() => toggleFacet(key, value)}
                      className="flex items-center gap-1 rounded-full border border-border bg-accent px-2 py-0.5 text-[11px] hover:bg-muted transition-colors cursor-pointer"
                    >
                      <span className="text-muted-foreground">{FACET_LABELS[key]}</span>
                      <span>{key === 'stage' ? `${value} ${STAGE_LABELS[value] ?? ''}`.trim() : key === 'layer' ? `${value} ${LAYER_LABELS[value] ?? ''}`.trim() : value}</span>
                      <X className="size-2.5 text-muted-foreground" />
                    </button>
                  ))}
                  <button
                    onClick={() => setSelections({})}
                    className="text-[11px] text-muted-foreground hover:text-foreground transition-colors cursor-pointer px-1"
                  >
                    清空
                  </button>
                </div>
              )}

              <KpiStrip kpis={kpis} onShowPending={() => setSelections({ state: '待复核' })} />

              <CoverageMatrix
                coverage={coverage}
                selections={selections}
                onSelectCell={(stage, layer) =>
                  setSelections((prev) =>
                    prev.stage === stage && prev.layer === layer
                      ? { ...prev, stage: undefined, layer: undefined }
                      : { ...prev, stage, layer },
                  )
                }
              />

              {selections.state === '待复核' && (() => {
                const lowRisk = filtered.filter(
                  (e) => reviewStatus(e.meta) === 'pending_review' && !isMandatoryReview(e.meta))
                return lowRisk.length > 0 ? (
                  <div className="mb-2 flex items-center justify-between rounded-md border border-border bg-muted/30 px-3 py-1.5 text-xs">
                    <span className="text-muted-foreground">
                      低风险条目可批量通过;法律 C1 / 数据出境 R1 须逐条复核。
                    </span>
                    <button onClick={() => batchApprove(lowRisk)} disabled={batchBusy}
                      className="rounded-md border border-border px-2 py-0.5 hover:bg-accent transition-colors cursor-pointer disabled:opacity-50">
                      {batchBusy ? '处理中…' : `批量通过 ${lowRisk.length} 条低风险`}
                    </button>
                  </div>
                ) : null
              })()}
              <EntryTable entries={filtered} onOpen={openEntry}
                onReviewed={onReviewed} onEdit={setEditing} onReprocess={reprocessEntry} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

function FacetGroup({
  facet,
  entries,
  selections,
  onToggle,
}: {
  facet: FacetKey
  entries: CorpusEntry[]
  selections: FacetSelections
  onToggle: (key: FacetKey, value: string) => void
}) {
  const [expanded, setExpanded] = React.useState(false)
  // Options counted against entries filtered by every *other* facet, so counts
  // reflect what selecting this value would actually return.
  const options = React.useMemo(() => {
    const others = { ...selections }
    delete others[facet]
    const base = entries.filter((e) => entryMatches(e.meta, others))
    return collectFacetOptions(base, facet)
  }, [entries, selections, facet])

  if (options.length === 0) return null
  const shown = expanded ? options : options.slice(0, MAX_CHIPS)

  return (
    <div>
      <div className="mb-1.5 text-[11px] font-medium text-muted-foreground">{FACET_LABELS[facet]}</div>
      <div className="flex flex-wrap gap-1">
        {shown.map((opt) => {
          const active = selections[facet] === opt.value
          const label =
            facet === 'stage'
              ? `${opt.value} ${STAGE_LABELS[opt.value] ?? ''}`.trim()
              : opt.value
          return (
            <button
              key={opt.value}
              onClick={() => onToggle(facet, opt.value)}
              className={cn(
                'rounded-full border px-2 py-0.5 text-[11px] transition-colors cursor-pointer',
                active
                  ? 'border-foreground/40 bg-accent text-foreground'
                  : 'border-border text-muted-foreground hover:text-foreground hover:bg-accent',
              )}
              title={`${label} — ${opt.count} 条`}
            >
              {label}
              <span className="ml-1 text-muted-foreground/60 tabular-nums">{opt.count}</span>
            </button>
          )
        })}
        {options.length > MAX_CHIPS && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="rounded-full px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            {expanded ? '收起' : `+${options.length - MAX_CHIPS}`}
          </button>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

function KpiStrip({ kpis, onShowPending }: { kpis: ReturnType<typeof computeKpis>; onShowPending: () => void }) {
  const pct = (x: number) => (kpis.total ? `${Math.round((x / kpis.total) * 100)}%` : '—')
  const cells: Array<{ label: string; value: string; hint: string; alert?: boolean; onClick?: () => void }> = [
    { label: '分面完备率', value: pct(kpis.completeness), hint: '八维必填维度全非空的条目占比 (公理一)' },
    {
      label: '货架覆盖率',
      value: `${kpis.shelfFilled}/${kpis.shelfTotal}`,
      hint: '主阶段×大类层非空格子数 — 空格即补采方向',
    },
    { label: '时效达标率', value: pct(kpis.onTime), hint: '复审未逾期条目占比 (review_due)' },
    {
      label: '引用溯源率',
      value: kpis.cited === null ? '—' : pct(kpis.cited),
      hint: '被至少一个 wiki 页面引用的条目占比',
    },
    {
      label: 'Wiki 覆盖率',
      value: kpis.wikiCovered === null || kpis.wikiCellsWithEntries === 0
        ? '—'
        : `${kpis.wikiCovered}/${kpis.wikiCellsWithEntries}`,
      hint: '有语料的货架格中已有维基页覆盖的格数(缺口即建页方向,详见 lint)',
    },
    {
      label: '待复核',
      value: String(kpis.pendingReview),
      hint: '人工复核队列条目数 — 点击筛选',
      alert: kpis.pendingReview > 0,
      onClick: kpis.pendingReview > 0 ? onShowPending : undefined,
    },
  ]
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
      {cells.map((cell) => (
        <button
          key={cell.label}
          onClick={cell.onClick}
          disabled={!cell.onClick}
          title={cell.hint}
          className={cn(
            'rounded-lg border border-border px-3 py-2 text-left',
            cell.onClick ? 'cursor-pointer hover:bg-accent transition-colors' : 'cursor-default',
          )}
        >
          <div className="text-[11px] text-muted-foreground">{cell.label}</div>
          <div
            className={cn(
              'mt-0.5 text-lg font-medium tabular-nums',
              cell.alert && 'text-amber-600 dark:text-amber-500',
            )}
          >
            {cell.value}
          </div>
        </button>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------

function CoverageMatrix({
  coverage,
  selections,
  onSelectCell,
}: {
  coverage: ReturnType<typeof buildCoverageGrid>
  selections: FacetSelections
  onSelectCell: (stage: string, layer: string) => void
}) {
  const max = Math.max(1, ...STAGES.flatMap((s) => LAYERS.map((l) => coverage.counts[s][l])))
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-2 flex items-baseline gap-2">
        <span className="text-xs font-medium">覆盖率账本</span>
        <span className="text-[11px] text-muted-foreground">主阶段 × 大类层 · 空格即下一轮补采方向</span>
      </div>
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular-nums">
          <thead>
            <tr>
              <th className="pr-2 pb-1 text-left font-normal text-muted-foreground">阶段＼层</th>
              {LAYERS.map((l) => (
                <th key={l} className="px-1 pb-1 font-normal text-muted-foreground">
                  {l} {LAYER_LABELS[l]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {STAGES.map((s) => (
              <tr key={s}>
                <td className="pr-2 py-0.5 text-muted-foreground whitespace-nowrap">
                  {s} {STAGE_LABELS[s]}
                </td>
                {LAYERS.map((l) => {
                  const count = coverage.counts[s][l]
                  const active = selections.stage === s && selections.layer === l
                  return (
                    <td key={l} className="px-1 py-0.5">
                      <button
                        onClick={() => onSelectCell(s, l)}
                        className={cn(
                          'w-14 rounded-md border py-1 text-center transition-colors cursor-pointer',
                          active
                            ? 'border-foreground/40 bg-accent text-foreground'
                            : count === 0
                              ? 'border-dashed border-border text-muted-foreground/40 hover:text-muted-foreground'
                              : 'border-border hover:bg-accent',
                        )}
                        style={
                          count > 0 && !active
                            ? { backgroundColor: `color-mix(in oklab, var(--accent) ${Math.round((count / max) * 70)}%, transparent)` }
                            : undefined
                        }
                        title={`${s} ${STAGE_LABELS[s]} × ${l} ${LAYER_LABELS[l]}: ${count} 条`}
                      >
                        {count}
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {coverage.emptyCells.length > 0 && (
        <div className="mt-2 text-[11px] text-muted-foreground">
          空格 {coverage.emptyCells.length} 个:
          {' '}
          {coverage.emptyCells.slice(0, 8).map((c) => `${c.stage}×${c.layer}`).join(' · ')}
          {coverage.emptyCells.length > 8 && ` · +${coverage.emptyCells.length - 8}`}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

function EntryTable({ entries, onOpen, onReviewed, onEdit, onReprocess }: {
  entries: CorpusEntry[]
  onOpen: (e: CorpusEntry) => void
  onReviewed: () => void
  onEdit: (e: CorpusEntry) => void
  onReprocess: (e: CorpusEntry) => void
}) {
  if (entries.length === 0) {
    return <div className="py-8 text-center text-xs text-muted-foreground">当前筛选无匹配条目。</div>
  }
  return (
    <div className="divide-y divide-border rounded-lg border border-border">
      {entries.map((entry) => (
        <EntryRow key={entry.doc.id} entry={entry} onOpen={() => onOpen(entry)}
          onReviewed={onReviewed} onEdit={() => onEdit(entry)}
          onReprocess={() => onReprocess(entry)} />
      ))}
    </div>
  )
}

function EntryRow({ entry, onOpen, onReviewed, onEdit, onReprocess }: {
  entry: CorpusEntry
  onOpen: () => void
  onReviewed: () => void
  onEdit: () => void
  onReprocess: () => void
}) {
  const { meta, doc } = entry
  const status = reviewStatus(meta)
  const geo = [...meta.geo_region, ...meta.geo_country_names].join('·') || '通用'
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === 'Enter') onOpen() }}
      className="group block w-full px-3 py-2 text-left hover:bg-accent/50 transition-colors cursor-pointer"
    >
      <div className="flex items-center gap-2">
        <span className="truncate text-[13px] font-medium">{doc.title || doc.filename}</span>
        {status === 'pending_review' && <StateBadge tone="warn">待复核</StateBadge>}
        {status === 'pending_review' && <ReviewBadges meta={meta} />}
        {status === 'pending_review' && (
          <>
            <span className="flex-1" />
            <ReviewActions docId={doc.id} meta={meta} onDone={onReviewed} onEdit={onEdit} />
          </>
        )}
        {status === 'overdue' && <StateBadge tone="alert">复审到期</StateBadge>}
        {(meta.lifecycle_state === '已过期' || meta.lifecycle_state === '待更新') && (
          <StateBadge tone="alert">{meta.lifecycle_state}</StateBadge>
        )}
        {meta.timeliness === 'M1' && <StateBadge tone="warn">M1 高时效</StateBadge>}
        {status !== 'pending_review' && <span className="flex-1" />}
        <button
          onClick={(e) => { e.stopPropagation(); onReprocess() }}
          title="重新提取源文件(含 OCR 兜底)并重新分类入库 — 用于提取质量差的条目"
          className="shrink-0 flex items-center gap-1 rounded-md border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-accent hover:text-foreground transition-all cursor-pointer"
        >
          <RefreshCw className="size-2.5" />
          重新识别
        </button>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
        <code className="rounded bg-muted px-1 py-px text-[10px]">{meta.entry_id}</code>
        <span>
          {meta.stage} {STAGE_LABELS[meta.stage] ?? ''}
        </span>
        <span>{[meta.domain, ...meta.domain_ext].join('+')}</span>
        <span>{meta.genre}</span>
        <span>{geo}</span>
        <span>{meta.evidence}</span>
        <span>{meta.origin}</span>
        {meta.timeliness !== 'M1' && <span>{meta.timeliness}</span>}
        {meta.business?.scene && meta.business.code !== '待定' && (
          <span className="text-muted-foreground/70">
            {meta.business.code} {meta.business.scene}
          </span>
        )}
      </div>
    </div>
  )
}

function StateBadge({ tone, children }: { tone: 'warn' | 'alert'; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        'shrink-0 rounded-full border px-1.5 py-px text-[10px] leading-4',
        tone === 'warn'
          ? 'border-amber-500/30 text-amber-600 dark:text-amber-500'
          : 'border-red-500/30 text-red-600 dark:text-red-500',
      )}
    >
      {children}
    </span>
  )
}

// ---------------------------------------------------------------------------

function BusinessPane({
  classes,
  pending,
  selections,
  onSelectScene,
}: {
  classes: ReturnType<typeof buildBusinessView>
  pending: number
  selections: FacetSelections
  onSelectScene: (code: string) => void
}) {
  if (classes.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
        <p className="text-sm font-medium">暂无业务视图数据</p>
        <p className="text-xs text-muted-foreground max-w-md">
          业务标签由八维标注自动派生 — 导入 <code className="rounded bg-muted px-1">标注明细_业务视图.csv</code>
          (经 <code className="rounded bg-muted px-1">derive_business_view.py</code> 处理)后即可按 7 类需求 / 27 场景导航。
        </p>
      </div>
    )
  }
  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      <div className="mb-3 flex items-baseline gap-2">
        <span className="text-xs font-medium">7 类企业需求 → 27 业务场景</span>
        <span className="text-[11px] text-muted-foreground">按需求频度 P1→P7 排序 · 业务标签由八维自动派生</span>
        {pending > 0 && <span className="text-[11px] text-muted-foreground">业务待定 {pending} 条</span>}
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {classes.map((cls) => (
          <div key={cls.code} className="rounded-lg border border-border p-3">
            <div className="flex items-baseline gap-2">
              <span className="text-xs font-medium">
                {cls.code} {cls.name}
              </span>
              <span className="rounded bg-muted px-1 text-[10px] text-muted-foreground">{cls.priority}</span>
              <span className="ml-auto text-[11px] text-muted-foreground tabular-nums">{cls.count} 条</span>
            </div>
            <div className="mt-2 space-y-0.5">
              {cls.scenes.map((scene) => (
                <button
                  key={scene.code}
                  onClick={() => onSelectScene(scene.code)}
                  className={cn(
                    'flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[12px] transition-colors cursor-pointer',
                    selections.business === scene.code
                      ? 'bg-accent text-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent/60',
                  )}
                >
                  <span className="text-[10px] text-muted-foreground/60 tabular-nums">{scene.code}</span>
                  <span className="flex-1 truncate">{scene.scene}</span>
                  <span className="text-[11px] text-muted-foreground tabular-nums">{scene.count}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
