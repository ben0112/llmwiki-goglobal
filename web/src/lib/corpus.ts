// 出海智能体语料 (go-global corpus) — client-side helpers over the structured
// 八维 metadata that corpus/import_annotations.py writes into documents.metadata.
// All derivations (facet values, coverage grid, business view) are computed
// from the loaded document list, so the UI adapts to any 码表 version.

import type { DocumentListItem } from '@/lib/types'

export interface CorpusBusiness {
  code: string
  scene: string
  class: string
  priority: string
  pending?: boolean
}

export interface CorpusMeta {
  spec_version: string
  entry_id: string
  stage: string
  stage_ext: string[]
  domain: string
  domain_ext: string[]
  genre: string
  rule_type: string[]
  evidence: string
  origin: string
  gov_dept: string[]
  geo_scope: string
  geo_region: string[]
  geo_country: string[]
  geo_country_names: string[]
  industry: string[]
  mode: string[]
  timeliness: string
  lifecycle_state: string
  effective_date: string | null
  review_due: string | null
  confidence: number | null
  source_url?: string
  source_site?: string
  consumer_agents?: string[]
  business?: CorpusBusiness
}

export interface CorpusEntry {
  doc: DocumentListItem
  meta: CorpusMeta
}

export function getCorpusMeta(doc: DocumentListItem): CorpusMeta | null {
  let meta: unknown = doc.metadata
  if (typeof meta === 'string') {
    try {
      meta = JSON.parse(meta)
    } catch {
      return null
    }
  }
  if (!meta || typeof meta !== 'object') return null
  const m = meta as Record<string, unknown>
  if (typeof m.spec_version !== 'string' || typeof m.stage !== 'string') return null
  const arr = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : [])
  return {
    spec_version: m.spec_version,
    entry_id: String(m.entry_id ?? ''),
    stage: String(m.stage ?? ''),
    stage_ext: arr(m.stage_ext),
    domain: String(m.domain ?? ''),
    domain_ext: arr(m.domain_ext),
    genre: String(m.genre ?? ''),
    rule_type: arr(m.rule_type),
    evidence: String(m.evidence ?? ''),
    origin: String(m.origin ?? ''),
    gov_dept: arr(m.gov_dept),
    geo_scope: String(m.geo_scope ?? '通用'),
    geo_region: arr(m.geo_region),
    geo_country: arr(m.geo_country),
    geo_country_names: arr(m.geo_country_names),
    industry: arr(m.industry),
    mode: arr(m.mode),
    timeliness: String(m.timeliness ?? ''),
    lifecycle_state: String(m.lifecycle_state ?? ''),
    effective_date: typeof m.effective_date === 'string' ? m.effective_date : null,
    review_due: typeof m.review_due === 'string' ? m.review_due : null,
    confidence: typeof m.confidence === 'number' ? m.confidence : null,
    source_url: typeof m.source_url === 'string' ? m.source_url : undefined,
    source_site: typeof m.source_site === 'string' ? m.source_site : undefined,
    consumer_agents: arr(m.consumer_agents),
    business:
      m.business && typeof m.business === 'object'
        ? {
            code: String((m.business as Record<string, unknown>).code ?? ''),
            scene: String((m.business as Record<string, unknown>).scene ?? ''),
            class: String((m.business as Record<string, unknown>).class ?? ''),
            priority: String((m.business as Record<string, unknown>).priority ?? ''),
            pending: Boolean((m.business as Record<string, unknown>).pending),
          }
        : undefined,
  }
}

export function collectCorpusEntries(docs: DocumentListItem[]): CorpusEntry[] {
  const entries: CorpusEntry[] = []
  for (const doc of docs) {
    const meta = getCorpusMeta(doc)
    if (meta) entries.push({ doc, meta })
  }
  return entries
}

// ---------------------------------------------------------------------------
// Labels (display only — validation lives in corpus/codetables + lint)
// ---------------------------------------------------------------------------

export const STAGE_LABELS: Record<string, string> = {
  S0: '全程',
  S1: '引航',
  S2: '启航',
  S3: '助航',
  S4: '护航',
}

export const LAYER_LABELS: Record<string, string> = {
  G: '政府公共',
  C: '合规基础',
  O: '运营支撑',
  Z: '专项场景',
  X: '兜底',
}

export const TIMELINESS_LABELS: Record<string, string> = {
  M1: 'M1 高时效',
  M2: 'M2 中时效',
  M3: 'M3 低时效',
}

export const STAGES = ['S0', 'S1', 'S2', 'S3', 'S4'] as const
export const LAYERS = ['G', 'C', 'O', 'Z', 'X'] as const

// ---------------------------------------------------------------------------
// Facets
// ---------------------------------------------------------------------------

export type FacetKey =
  | 'stage'
  | 'layer'
  | 'domain'
  | 'genre'
  | 'rule'
  | 'evidence'
  | 'origin'
  | 'dept'
  | 'geo'
  | 'industry'
  | 'timeliness'
  | 'state'
  | 'business'

export const FACET_LABELS: Record<FacetKey, string> = {
  stage: '阶段',
  layer: '大类层',
  domain: '服务大类',
  genre: '体裁',
  rule: '隐性规则',
  evidence: '证据强度',
  origin: '来源域',
  dept: '归口部门',
  geo: '国别区域',
  industry: '行业',
  timeliness: '时效',
  state: '状态',
  business: '业务场景',
}

/** Values an entry exposes for a facet (primary + secondary where relevant). */
export function facetValues(meta: CorpusMeta, key: FacetKey): string[] {
  switch (key) {
    case 'stage':
      return [meta.stage, ...meta.stage_ext].filter(Boolean)
    case 'layer':
      return meta.domain ? [meta.domain[0]] : []
    case 'domain':
      return [meta.domain, ...meta.domain_ext].filter(Boolean)
    case 'genre':
      return meta.genre ? [meta.genre] : []
    case 'rule':
      return meta.rule_type.filter((r) => r && r !== 'R0')
    case 'evidence':
      return meta.evidence ? [meta.evidence] : []
    case 'origin':
      return meta.origin ? [meta.origin] : []
    case 'dept':
      return meta.gov_dept.filter(Boolean)
    case 'geo':
      return [...meta.geo_region, ...meta.geo_country_names].filter(Boolean)
    case 'industry':
      return meta.industry.filter((i) => i && i !== '通用')
    case 'timeliness':
      return meta.timeliness ? [meta.timeliness] : []
    case 'state':
      return meta.lifecycle_state ? [meta.lifecycle_state] : []
    case 'business':
      return meta.business?.code && meta.business.code !== '待定' ? [meta.business.code] : []
  }
}

export type FacetSelections = Partial<Record<FacetKey, string>>

export function entryMatches(meta: CorpusMeta, selections: FacetSelections): boolean {
  for (const [key, value] of Object.entries(selections)) {
    if (!value) continue
    const values = facetValues(meta, key as FacetKey)
    if (key === 'business' && !value.includes('.')) {
      // Class prefix: "B4" matches any B4.x scene.
      if (!values.some((v) => v === value || v.startsWith(`${value}.`))) return false
    } else if (!values.includes(value)) {
      return false
    }
  }
  return true
}

export function filterEntries(entries: CorpusEntry[], selections: FacetSelections): CorpusEntry[] {
  return entries.filter((e) => entryMatches(e.meta, selections))
}

export interface FacetOption {
  value: string
  count: number
}

/** Distinct values (with entry counts) present for a facet, most frequent first. */
export function collectFacetOptions(entries: CorpusEntry[], key: FacetKey): FacetOption[] {
  const counts = new Map<string, number>()
  for (const e of entries) {
    for (const v of new Set(facetValues(e.meta, key))) {
      counts.set(v, (counts.get(v) ?? 0) + 1)
    }
  }
  const options = [...counts.entries()].map(([value, count]) => ({ value, count }))
  options.sort((a, b) => b.count - a.count || a.value.localeCompare(b.value, 'zh-Hans-CN'))
  return options
}

// ---------------------------------------------------------------------------
// Coverage ledger (货架覆盖率: 主阶段 × 大类层)
// ---------------------------------------------------------------------------

export interface CoverageGrid {
  counts: Record<string, Record<string, number>>
  total: number
  emptyCells: Array<{ stage: string; layer: string }>
}

export function buildCoverageGrid(entries: CorpusEntry[]): CoverageGrid {
  const counts: Record<string, Record<string, number>> = {}
  for (const s of STAGES) {
    counts[s] = {}
    for (const l of LAYERS) counts[s][l] = 0
  }
  for (const { meta } of entries) {
    const layer = meta.domain?.[0]
    if (counts[meta.stage] && layer && layer in counts[meta.stage]) {
      counts[meta.stage][layer] += 1
    }
  }
  const emptyCells: Array<{ stage: string; layer: string }> = []
  for (const s of STAGES) {
    for (const l of LAYERS) {
      if (l !== 'X' && counts[s][l] === 0) emptyCells.push({ stage: s, layer: l })
    }
  }
  return { counts, total: entries.length, emptyCells }
}

// ---------------------------------------------------------------------------
// Business view (7 类需求 → 27 场景, derived — never separately annotated)
// ---------------------------------------------------------------------------

export interface BusinessScene {
  code: string
  scene: string
  count: number
}

export interface BusinessClass {
  code: string
  name: string
  priority: string
  count: number
  scenes: BusinessScene[]
}

export function buildBusinessView(entries: CorpusEntry[]): BusinessClass[] {
  const classes = new Map<string, BusinessClass>()
  for (const { meta } of entries) {
    const b = meta.business
    if (!b?.code || b.code === '待定') continue
    const classCode = b.code.split('.')[0]
    let cls = classes.get(classCode)
    if (!cls) {
      cls = { code: classCode, name: b.class, priority: b.priority, count: 0, scenes: [] }
      classes.set(classCode, cls)
    }
    cls.count += 1
    let scene = cls.scenes.find((s) => s.code === b.code)
    if (!scene) {
      scene = { code: b.code, scene: b.scene, count: 0 }
      cls.scenes.push(scene)
    }
    scene.count += 1
  }
  const result = [...classes.values()]
  // 频度优先级 P1 → P7 (the spec's demand-frequency ordering).
  result.sort((a, b) => a.priority.localeCompare(b.priority) || a.code.localeCompare(b.code))
  for (const cls of result) cls.scenes.sort((a, b) => a.code.localeCompare(b.code, undefined, { numeric: true }))
  return result
}

export function businessPendingCount(entries: CorpusEntry[]): number {
  return entries.filter((e) => !e.meta.business || e.meta.business.pending || e.meta.business.code === '待定').length
}

// ---------------------------------------------------------------------------
// Quality KPIs (spec §5.4 质量 KPI)
// ---------------------------------------------------------------------------

const REQUIRED_DIMENSIONS: Array<keyof CorpusMeta> = [
  'stage', 'domain', 'genre', 'evidence', 'origin', 'gov_dept',
  'timeliness', 'lifecycle_state', 'review_due',
]

export interface CorpusKpis {
  total: number
  /** 分面完备率: entries with every required dimension non-empty. */
  completeness: number
  /** 货架覆盖率: non-empty stage × G/C/O/Z cells. */
  shelfFilled: number
  shelfTotal: number
  /** 时效达标率: entries whose review_due has not passed. */
  onTime: number
  /** 引用溯源率: entries cited by at least one wiki page (null until graph loads). */
  cited: number | null
  pendingReview: number
}

export function computeKpis(
  entries: CorpusEntry[],
  citedDocIds: Set<string> | null,
  todayIso?: string,
): CorpusKpis {
  const today = todayIso ?? new Date().toISOString().slice(0, 10)
  let completeness = 0
  let onTime = 0
  let pendingReview = 0
  let cited = 0
  for (const { doc, meta } of entries) {
    const complete = REQUIRED_DIMENSIONS.every((f) => {
      const v = meta[f]
      return Array.isArray(v) ? v.length > 0 : Boolean(v)
    })
    if (complete) completeness += 1
    if (meta.review_due && meta.review_due.slice(0, 10) >= today) onTime += 1
    if (meta.lifecycle_state === '待复核') pendingReview += 1
    if (citedDocIds?.has(doc.id)) cited += 1
  }
  const grid = buildCoverageGrid(entries)
  const shelfTotal = STAGES.length * 4 // X 兜底 excluded — it is not a target shelf
  const shelfFilled = shelfTotal - grid.emptyCells.length
  return {
    total: entries.length,
    completeness,
    shelfFilled,
    shelfTotal,
    onTime,
    cited: citedDocIds ? cited : null,
    pendingReview,
  }
}

// ---------------------------------------------------------------------------
// Lifecycle / review status
// ---------------------------------------------------------------------------

export type ReviewStatus = 'ok' | 'overdue' | 'pending_review'

export function reviewStatus(meta: CorpusMeta, todayIso?: string): ReviewStatus {
  if (meta.lifecycle_state === '待复核') return 'pending_review'
  const today = todayIso ?? new Date().toISOString().slice(0, 10)
  if (meta.review_due && meta.review_due.slice(0, 10) < today) return 'overdue'
  return 'ok'
}
