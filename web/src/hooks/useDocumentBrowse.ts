'use client'

import * as React from 'react'

import { useReadPage } from '@/hooks/useReadPage'
import type {
  DocumentBrowsePage,
  DocumentListItem,
  ReadDocument,
  ReadFolder,
} from '@/lib/types'

interface DocumentBrowseOptions {
  kbId: string
  token: string | null
  path: string
  query?: string
  sort?: 'name' | 'date' | 'type' | 'path'
  direction?: 'asc' | 'desc'
  pollInterval?: number
}

export function readDocumentToListItem(document: ReadDocument): DocumentListItem {
  return {
    ...document,
    knowledge_base_id: document.knowledge_base_id ?? '',
    user_id: '',
    file_size: document.file_size ?? 0,
    version: document.version ?? 0,
    url: null,
    relative_path: null,
    content_hash: null,
    created_at: document.created_at ?? '',
    updated_at: document.updated_at ?? '',
  }
}

export function useDocumentBrowse({
  kbId,
  token,
  path,
  query = '',
  sort = 'name',
  direction = 'asc',
  pollInterval = process.env.NEXT_PUBLIC_MODE === 'local' ? 2_000 : 0,
}: DocumentBrowseOptions) {
  const search = new URLSearchParams({ path, sort, direction })
  if (query.trim()) search.set('query', query.trim())
  const url = kbId
    ? `/v1/knowledge-bases/${kbId}/documents/browse?${search.toString()}`
    : ''
  const queryKey = `${kbId}|${path}|${query.trim()}|${sort}|${direction}`
  const read = useReadPage<ReadDocument>({
    url,
    token,
    queryKey,
    pollInterval,
    initialLimit: 100,
  })
  const documents = React.useMemo(
    () => read.items.map(readDocumentToListItem),
    [read.items],
  )
  const page = read.page as DocumentBrowsePage | null
  const [folders, setFolders] = React.useState<ReadFolder[]>([])
  React.useEffect(() => setFolders([]), [queryKey])
  React.useEffect(() => {
    if (page && (!read.hasPrevious || page.folders.length)) {
      setFolders(
        page.folders.filter((folder) => folder.path.replace(/\/+$/, '') !== '/wiki'),
      )
    }
  }, [page, read.hasPrevious])

  return {
    ...read,
    items: documents,
    folders,
    sourceCount: page?.source_count ?? 0,
    failedCount: page?.failed_count ?? 0,
    corpusCount: page?.corpus_count ?? 0,
  }
}
