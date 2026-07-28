export interface KnowledgeBase {
  id: string
  user_id: string
  name: string
  slug: string
  description: string | null
  source_count: number
  wiki_page_count: number
  created_at: string
  updated_at: string
  kind?: 'wiki' | 'course'
}

export interface Document {
  id: string
  knowledge_base_id: string
  user_id: string
  filename: string
  title: string | null
  path: string
  file_type: string
  file_size: number
  status: string
  page_count: number | null
  content: string | null
  tags: string[]
  date: string | null
  metadata: Record<string, unknown> | null
  error_message: string | null
  url: string | null
  version: number
  document_number: number | null
  sort_order: number | null
  archived: boolean
  // 原始文件字节的 SHA-256(本地模式;上传内容去重用)
  content_hash?: string | null
  // 工作区内相对路径(本地模式;hosted 列表可能不带)
  relative_path?: string | null
  // 待复查标记:引用的语料/页面变更后置位,编辑保存后清除
  stale_since?: string | null
  created_at: string
  updated_at: string
}

export type DocumentListItem = Omit<Document, 'content'>

export interface ReadDocument {
  id: string
  filename: string
  path: string
  file_type: string
  status: string
  knowledge_base_id: string | null
  title: string | null
  file_size: number | null
  page_count: number | null
  tags: string[]
  date: string | null
  metadata: Record<string, unknown> | null
  error_message: string | null
  version: number | null
  document_number: number | null
  sort_order: number | null
  archived: boolean
  stale_since: string | null
  created_at: string | null
  updated_at: string | null
}

export interface ReadPage<T> {
  revision: number
  items: T[]
  next_cursor: string | null
  total_count: number
}

export interface ReadFolder {
  name: string
  path: string
  document_count: number
}

export interface DocumentBrowsePage extends ReadPage<ReadDocument> {
  folders: ReadFolder[]
  source_count: number
  failed_count: number
  corpus_count: number
}

export interface CorpusSummary {
  revision: number
  total_count: number
  filtered_count: number
  facets: Record<string, Record<string, number>>
  coverage: Record<string, unknown>
  business_classes: Record<string, number>
  business_scenes: Record<string, number>
  kpis: Record<string, number | null>
}

export interface GraphSummary {
  revision: number
  node_count: number
  edge_count: number
  cited_document_ids: string[]
}

export interface ReadDocumentStatus {
  id: string | null
  document_number: number | null
  status: string
  error_message: string | null
  version: number | null
}

export interface ReadDocumentStatusPage {
  revision: number
  items: ReadDocumentStatus[]
}

export type UploadPreflightCode =
  | 'accepted'
  | 'duplicate_name'
  | 'duplicate_content'
  | 'unsupported'
  | 'too_large'

export interface UploadPreflightItem {
  path: string
  filename: string
  size: number
  sha256?: string | null
  accepted?: boolean | null
  code?: UploadPreflightCode | null
  existing_document_id?: string | null
}

export interface UploadPreflightResponse {
  revision: number
  items: UploadPreflightItem[]
}

export type PropertyType = 'text' | 'number' | 'date' | 'checkbox' | 'select' | 'url'

export interface TypedProperty {
  type: PropertyType
  value: string | number | boolean | null
  options?: string[]
}

export type PropertyMap = Record<string, TypedProperty>

export interface WikiNode {
  title: string
  path?: string
  docNumber?: number | null
  children?: WikiNode[]
  // Course mode only — derived from the lesson doc's metadata.course.status + tree order.
  status?: 'complete' | 'in_progress' | 'not_started'
  locked?: boolean
}

export interface WikiSubsection {
  id: string
  title: string
}
