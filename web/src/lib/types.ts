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
