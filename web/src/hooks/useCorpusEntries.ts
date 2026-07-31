'use client'

import * as React from 'react'

import { useReadPage } from '@/hooks/useReadPage'
import { corpusEntryFromRecord, type CorpusEntry, type FacetSelections } from '@/lib/corpus'
import { apiUrl } from '@/lib/runtime-env'
import type { CorpusEntryRecord, CorpusSummary } from '@/lib/types'

const isLocal = process.env.NEXT_PUBLIC_MODE === 'local'

interface CorpusEntriesOptions {
  kbId: string
  token: string | null
  selections: FacetSelections
  query?: string
  sort?: 'name' | 'stage' | 'domain' | 'review_due' | 'updated'
  direction?: 'asc' | 'desc'
}

function queryString(
  selections: FacetSelections,
  query: string,
  sort?: string,
  direction?: string,
) {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(selections).sort(([a], [b]) => a.localeCompare(b))) {
    if (value) params.set(key, value)
  }
  if (query.trim()) params.set('query', query.trim())
  if (sort) params.set('sort', sort)
  if (direction) params.set('direction', direction)
  return params.toString()
}

export function useCorpusEntries({
  kbId,
  token,
  selections,
  query = '',
  sort = 'name',
  direction = 'asc',
}: CorpusEntriesOptions) {
  const filters = queryString(selections, query, sort, direction)
  const summaryFilters = queryString(selections, query)
  const queryKey = `${kbId}|${filters}`
  const entriesUrl = kbId
    ? `/v1/knowledge-bases/${kbId}/corpus/entries?${filters}`
    : ''
  const summaryUrl = kbId
    ? `${apiUrl()}/v1/knowledge-bases/${kbId}/corpus/summary?${summaryFilters}`
    : ''
  const pages = useReadPage<CorpusEntryRecord>({
    url: entriesUrl,
    token,
    queryKey,
    initialLimit: 200,
    pollInterval: process.env.NEXT_PUBLIC_MODE === 'local' ? 5_000 : 0,
  })
  const [summary, setSummary] = React.useState<CorpusSummary | null>(null)
  const [summaryLoading, setSummaryLoading] = React.useState(true)
  const [summaryError, setSummaryError] = React.useState<Error | null>(null)
  const summaryController = React.useRef<AbortController | null>(null)
  const summaryEtags = React.useRef(new Map<string, string>())

  const fetchSummary = React.useCallback(async () => {
    if (!summaryUrl || (!isLocal && !token)) {
      setSummaryLoading(false)
      return
    }
    summaryController.current?.abort()
    const controller = new AbortController()
    summaryController.current = controller
    setSummaryLoading(true)
    setSummaryError(null)
    const headers = new Headers({ Accept: 'application/json' })
    if (!isLocal && token) headers.set('Authorization', `Bearer ${token}`)
    const etag = summaryEtags.current.get(summaryUrl)
    if (etag) headers.set('If-None-Match', etag)
    try {
      const response = await fetch(summaryUrl, { headers, signal: controller.signal })
      if (response.status === 304) return
      if (!response.ok) throw new Error(`Corpus summary request failed: ${response.status}`)
      const result = (await response.json()) as CorpusSummary
      const responseEtag = response.headers.get('ETag')
      if (responseEtag) summaryEtags.current.set(summaryUrl, responseEtag)
      if (!controller.signal.aborted) setSummary(result)
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setSummaryError(error instanceof Error ? error : new Error('Corpus summary request failed'))
      }
    } finally {
      if (summaryController.current === controller) setSummaryLoading(false)
    }
  }, [summaryUrl, token])

  React.useEffect(() => {
    void fetchSummary()
    return () => summaryController.current?.abort()
  }, [fetchSummary])

  const entries = React.useMemo(() => {
    const result: CorpusEntry[] = []
    for (const record of pages.items) {
      const entry = corpusEntryFromRecord(record)
      if (entry) result.push(entry)
    }
    return result
  }, [pages.items])

  const reload = React.useCallback(async () => {
    await Promise.all([pages.reload(), fetchSummary()])
  }, [fetchSummary, pages.reload])

  return {
    ...pages,
    entries,
    summary,
    summaryLoading,
    error: pages.error ?? summaryError,
    loading: pages.loading || (summary === null && summaryLoading),
    reload,
  }
}
