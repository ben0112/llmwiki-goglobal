'use client'

import * as React from 'react'

import { readDocumentToListItem } from '@/hooks/useDocumentBrowse'
import { useReadPage } from '@/hooks/useReadPage'
import type { DocumentListItem, ReadDocument } from '@/lib/types'

export function useWikiPages(kbId: string, token: string | null) {
  const read = useReadPage<ReadDocument>({
    url: kbId ? `/v1/knowledge-bases/${kbId}/wiki/pages` : '',
    token,
    queryKey: kbId,
    initialLimit: 200,
    pollInterval: process.env.NEXT_PUBLIC_MODE === 'local' ? 5_000 : 0,
  })
  const [documents, setDocuments] = React.useState<DocumentListItem[]>([])
  const revisionRef = React.useRef<number | null>(null)

  React.useEffect(() => {
    if (!read.page) return
    const converted = read.items.map(readDocumentToListItem)
    setDocuments((previous) => {
      if (revisionRef.current !== read.page?.revision) return converted
      const byId = new Map(previous.map((document) => [document.id, document]))
      for (const document of converted) byId.set(document.id, document)
      return Array.from(byId.values())
    })
    revisionRef.current = read.page.revision
    if (read.hasNext && !read.refreshing) read.loadNext()
  }, [read.hasNext, read.items, read.loadNext, read.page, read.refreshing])

  React.useEffect(() => {
    setDocuments([])
    revisionRef.current = null
  }, [kbId])

  return { ...read, documents, setDocuments }
}
