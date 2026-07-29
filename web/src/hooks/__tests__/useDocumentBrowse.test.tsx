import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useDocumentBrowse } from '@/hooks/useDocumentBrowse'
import type { DocumentBrowsePage, ReadFolder } from '@/lib/types'

function page(folders: ReadFolder[]): DocumentBrowsePage {
  return {
    revision: 7,
    items: [],
    next_cursor: null,
    total_count: 0,
    folders,
    source_count: 0,
    failed_count: 0,
    corpus_count: 0,
  }
}

function response(body: DocumentBrowsePage): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useDocumentBrowse', () => {
  it('clears stale folders when a refresh returns an empty folder list', async () => {
    const wiki = { name: 'wiki', path: '/wiki/', document_count: 2 }
    vi.stubGlobal(
      'fetch',
      vi.fn()
        .mockResolvedValueOnce(response(page([wiki])))
        .mockResolvedValueOnce(response(page([]))),
    )
    const { result } = renderHook(() =>
      useDocumentBrowse({ kbId: 'kb-1', token: 'token', path: '/', pollInterval: 0 }),
    )
    await waitFor(() => expect(result.current.folders).toEqual([wiki]))

    await act(async () => result.current.reload())

    await waitFor(() => expect(result.current.folders).toEqual([]))
  })
})
