import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  clearDocumentResolverCache,
  useDocumentResolver,
} from '@/hooks/useDocumentResolver'
import type { ReadDocument } from '@/lib/types'

const DOCUMENT: ReadDocument = {
  id: 'doc-1',
  filename: 'one.pdf',
  path: '/',
  file_type: 'pdf',
  status: 'ready',
  knowledge_base_id: 'kb-1',
  title: 'One',
  file_size: 1,
  page_count: 1,
  tags: [],
  date: null,
  metadata: null,
  error_message: null,
  version: 1,
  document_number: 1,
  sort_order: 0,
  archived: false,
  stale_since: null,
  created_at: null,
  updated_at: null,
}

function response(status = 200) {
  return new Response(status === 200 ? JSON.stringify(DOCUMENT) : null, {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => {
  clearDocumentResolverCache()
  vi.unstubAllGlobals()
})

describe('useDocumentResolver', () => {
  it('coalesces identical concurrent resolutions', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response())
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() =>
      useDocumentResolver({ kbId: 'kb-1', token: 'token', revision: 7, navigationKey: 'wiki/a' }),
    )

    const [first, second] = await act(async () =>
      Promise.all([
        result.current.resolve({ documentNumber: 1 }),
        result.current.resolve({ documentNumber: 1 }),
      ]),
    )

    expect(first).toEqual(DOCUMENT)
    expect(second).toBe(first)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('scopes positive and negative cache entries by revision', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response(404))
      .mockResolvedValueOnce(response())
    vi.stubGlobal('fetch', fetchMock)
    const { result, rerender } = renderHook(
      ({ revision }) =>
        useDocumentResolver({ kbId: 'kb-1', token: 'token', revision, navigationKey: 'wiki/a' }),
      { initialProps: { revision: 7 } },
    )

    expect(await result.current.resolve({ logicalReference: 'one.pdf' })).toBeNull()
    expect(await result.current.resolve({ logicalReference: 'one.pdf' })).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(1)

    rerender({ revision: 8 })
    expect(await result.current.resolve({ logicalReference: 'one.pdf' })).toEqual(DOCUMENT)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('aborts pending resolutions when Wiki navigation changes', async () => {
    let signal: AbortSignal | undefined
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      signal = init?.signal as AbortSignal
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener(
          'abort',
          () => reject(new DOMException('aborted', 'AbortError')),
          { once: true },
        )
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    const { result, rerender } = renderHook(
      ({ navigationKey }) =>
        useDocumentResolver({ kbId: 'kb-1', token: 'token', revision: 7, navigationKey }),
      { initialProps: { navigationKey: 'wiki/a' } },
    )
    const pending = result.current.resolve({ documentNumber: 1 })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    rerender({ navigationKey: 'wiki/b' })
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    expect(signal?.aborted).toBe(true)
  })
})
