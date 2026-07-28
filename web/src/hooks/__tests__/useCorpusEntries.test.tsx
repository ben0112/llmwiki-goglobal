import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useCorpusEntries } from '@/hooks/useCorpusEntries'
import { buildBatchReviewTargetIds, type FacetSelections } from '@/lib/corpus'
import type { CorpusEntryRecord, CorpusSummary, ReadPage } from '@/lib/types'

function entry(id: string): CorpusEntryRecord {
  return {
    id,
    filename: `${id}.md`,
    title: id,
    path: '/corpus/',
    document_number: Number(id.replace(/\D/g, '')) || 1,
    metadata: { spec_version: '1', stage: 'S1' },
  }
}

function page(id: string, nextCursor: string | null, revision = 7): ReadPage<CorpusEntryRecord> {
  return { revision, items: [entry(id)], next_cursor: nextCursor, total_count: 4 }
}

const summary: CorpusSummary = {
  revision: 7,
  total_count: 4,
  filtered_count: 4,
  facets: { stage: { S1: 4 } },
  coverage: { counts: {}, total: 4 },
  business_classes: {},
  business_scenes: {},
  kpis: { total: 4 },
}

function json(body: unknown, etag = '"kb-read-7"') {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json', ETag: etag },
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useCorpusEntries', () => {
  it('starts summary and first-page reads in parallel and aborts both on filter changes', async () => {
    const requests: Array<{
      url: string
      signal: AbortSignal
      resolve: (response: Response) => void
    }> = []
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((resolve, reject) => {
        const signal = init?.signal as AbortSignal
        requests.push({ url: String(input), signal, resolve })
        signal.addEventListener(
          'abort',
          () => reject(new DOMException('aborted', 'AbortError')),
          { once: true },
        )
      }),
    ))
    const { result, rerender } = renderHook(
      ({ selections }: { selections: FacetSelections }) =>
        useCorpusEntries({ kbId: 'kb-1', token: 'token', selections }),
      { initialProps: { selections: { stage: 'S1' } } },
    )

    await waitFor(() => expect(requests).toHaveLength(2))
    expect(requests.some((request) => request.url.includes('/corpus/entries'))).toBe(true)
    expect(requests.some((request) => request.url.includes('/corpus/summary'))).toBe(true)

    rerender({ selections: { stage: 'S2' } })
    await waitFor(() => expect(requests).toHaveLength(4))
    expect(requests[0].signal.aborted).toBe(true)
    expect(requests[1].signal.aborted).toBe(true)

    for (const request of requests.slice(2)) {
      request.resolve(request.url.includes('/summary') ? json(summary) : json(page('doc-2', null)))
    }
    await waitFor(() => expect(result.current.entries[0]?.doc.id).toBe('doc-2'))
    expect(result.current.summary).toEqual(summary)
  })

  it('keeps the visible page during a stale-cursor restart and retains at most three pages', async () => {
    let resolveReplacement!: (response: Response) => void
    const replacement = new Promise<Response>((resolve) => { resolveReplacement = resolve })
    const entryResponses: Response[] = [
      json(page('doc-1', 'cursor-1')),
      json(page('doc-2', 'cursor-2')),
      json(page('doc-3', 'cursor-3')),
      json(page('doc-4', 'cursor-4')),
      new Response(JSON.stringify({ detail: { code: 'stale_cursor', revision: 8 } }), {
        status: 409,
        headers: { 'Content-Type': 'application/json' },
      }),
    ]
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      if (String(input).includes('/summary')) return Promise.resolve(json(summary))
      const [response] = entryResponses.splice(0, 1)
      return response ? Promise.resolve(response) : replacement
    }))
    const { result } = renderHook(() =>
      useCorpusEntries({ kbId: 'kb-1', token: 'token', selections: {} }),
    )
    await waitFor(() => expect(result.current.entries[0]?.doc.id).toBe('doc-1'))

    for (const expected of ['doc-2', 'doc-3', 'doc-4']) {
      act(() => result.current.loadNext())
      await waitFor(() => expect(result.current.entries[0]?.doc.id).toBe(expected))
    }
    expect(result.current.cachedPageCount).toBe(3)
    act(() => result.current.loadPrevious())
    await waitFor(() => expect(result.current.entries[0]?.doc.id).toBe('doc-3'))
    act(() => result.current.loadNext())
    await waitFor(() => expect(result.current.entries[0]?.doc.id).toBe('doc-4'))
    const visible = result.current.entries

    act(() => result.current.loadNext())
    await waitFor(() => expect(result.current.refreshing).toBe(true))
    expect(result.current.entries).toBe(visible)
    resolveReplacement(json(page('replacement', null, 8), '"kb-read-8"'))
    await waitFor(() => expect(result.current.entries[0]?.doc.id).toBe('replacement'))
  })

  it('rejects batch-review target lists larger than 200 ids', () => {
    expect(buildBatchReviewTargetIds(Array.from({ length: 200 }, (_, index) => `doc-${index}`)))
      .toHaveLength(200)
    expect(() => buildBatchReviewTargetIds(Array.from({ length: 201 }, (_, index) => `doc-${index}`)))
      .toThrow(/200/)
  })
})
