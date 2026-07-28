import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useReadPage } from '@/hooks/useReadPage'
import type { ReadPage } from '@/lib/types'

type Item = { id: string }

function page(id: string, nextCursor: string | null, revision = 7): ReadPage<Item> {
  return {
    revision,
    items: [{ id }],
    next_cursor: nextCursor,
    total_count: 2,
  }
}

function jsonResponse(body: unknown, etag = '"kb-read-7"'): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json', ETag: etag },
  })
}

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  Object.defineProperty(document, 'hidden', {
    configurable: true,
    value: false,
  })
})

describe('useReadPage', () => {
  it('aborts the old request when query parameters change', async () => {
    const requests: Array<{
      signal: AbortSignal
      resolve: (response: Response) => void
    }> = []
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((resolve, reject) => {
        const signal = init?.signal as AbortSignal
        requests.push({ signal, resolve })
        signal.addEventListener(
          'abort',
          () => reject(new DOMException('aborted', 'AbortError')),
          { once: true },
        )
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const { result, rerender } = renderHook(
      ({ queryKey, url }) =>
        useReadPage<Item>({ url, token: 'token', queryKey, initialLimit: 100 }),
      {
        initialProps: {
          queryKey: 'stage=S1',
          url: '/v1/kb/corpus/entries?stage=S1',
        },
      },
    )
    await waitFor(() => expect(requests).toHaveLength(1))

    rerender({
      queryKey: 'stage=S2',
      url: '/v1/kb/corpus/entries?stage=S2',
    })
    await waitFor(() => expect(requests).toHaveLength(2))
    expect(requests[0].signal.aborted).toBe(true)

    requests[1].resolve(jsonResponse(page('replacement', null)))
    await waitFor(() => expect(result.current.items[0]?.id).toBe('replacement'))
  })

  it('sends the ETag and preserves item identity after 304', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(page('one', null)))
      .mockResolvedValueOnce(new Response(null, { status: 304 }))
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() =>
      useReadPage<Item>({
        url: '/v1/kb/documents/browse?path=%2F',
        token: 'token',
        queryKey: 'root',
      }),
    )
    await waitFor(() => expect(result.current.items).toHaveLength(1))
    const items = result.current.items

    await act(async () => result.current.reload())
    await waitFor(() => expect(result.current.refreshing).toBe(false))

    expect(result.current.items).toBe(items)
    const secondInit = fetchMock.mock.calls[1][1] as RequestInit
    expect(new Headers(secondInit.headers).get('If-None-Match')).toBe(
      '"kb-read-7"',
    )
  })

  it('keeps visible data on 409 until the first-page replacement succeeds', async () => {
    let resolveReplacement!: (response: Response) => void
    const replacement = new Promise<Response>((resolve) => {
      resolveReplacement = resolve
    })
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(page('visible', 'next')))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ detail: { code: 'stale_cursor', revision: 8 } }),
          { status: 409, headers: { 'Content-Type': 'application/json' } },
        ),
      )
      .mockReturnValueOnce(replacement)
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() =>
      useReadPage<Item>({
        url: '/v1/kb/corpus/entries',
        token: 'token',
        queryKey: 'all',
      }),
    )
    await waitFor(() => expect(result.current.items[0]?.id).toBe('visible'))
    const visibleItems = result.current.items

    act(() => result.current.loadNext())
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    expect(result.current.items).toBe(visibleItems)
    expect(result.current.refreshing).toBe(true)

    resolveReplacement(jsonResponse(page('replacement', null, 8), '"kb-read-8"'))
    await waitFor(() => expect(result.current.items[0]?.id).toBe('replacement'))
    expect(result.current.revision).toBe(8)
  })

  it('does not poll while the document is hidden', async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(page('one', null)))
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() =>
      useReadPage<Item>({
        url: '/v1/kb/documents/browse?path=%2F',
        token: 'token',
        queryKey: 'root',
        pollInterval: 1_000,
      }),
    )
    await act(async () => Promise.resolve())
    expect(result.current.items[0]?.id).toBe('one')
    Object.defineProperty(document, 'hidden', {
      configurable: true,
      value: true,
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
