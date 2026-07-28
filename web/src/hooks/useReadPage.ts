'use client'

import * as React from 'react'

import {
  createReadPageState,
  readPageReducer,
} from '@/lib/read-models'
import { apiUrl } from '@/lib/runtime-env'
import type { ReadPage } from '@/lib/types'

const isLocal = process.env.NEXT_PUBLIC_MODE === 'local'
const EMPTY_ITEMS: never[] = []

export interface UseReadPageOptions {
  url: string
  token: string | null
  queryKey: string
  pollInterval?: number
  initialLimit?: number
}

function requestUrl(path: string, cursor: string | null, limit: number): string {
  const result = new URL(path, apiUrl())
  if (!result.searchParams.has('limit')) {
    result.searchParams.set('limit', String(limit))
  }
  if (cursor) {
    result.searchParams.set('cursor', cursor)
  } else {
    result.searchParams.delete('cursor')
  }
  return result.toString()
}

function staleRevision(body: unknown): number | null {
  if (!body || typeof body !== 'object') return null
  const detail = (body as { detail?: unknown }).detail
  if (!detail || typeof detail !== 'object') return null
  const value = detail as { code?: unknown; revision?: unknown }
  return value.code === 'stale_cursor' &&
    typeof value.revision === 'number' &&
    Number.isSafeInteger(value.revision)
    ? value.revision
    : null
}

export function useReadPage<T>({
  url,
  token,
  queryKey,
  pollInterval = 0,
  initialLimit = 100,
}: UseReadPageOptions) {
  const [state, dispatch] = React.useReducer(
    readPageReducer<T>,
    queryKey,
    createReadPageState<T>,
  )
  const [loading, setLoading] = React.useState(true)
  const [refreshing, setRefreshing] = React.useState(false)
  const [error, setError] = React.useState<Error | null>(null)
  const stateRef = React.useRef(state)
  const activeController = React.useRef<AbortController | null>(null)
  const requestSequence = React.useRef(0)
  const etags = React.useRef(new Map<string, string>())
  stateRef.current = state

  const fetchPage = React.useCallback(
    async (initialCursor: string | null) => {
      if (!url || (!isLocal && !token)) return

      activeController.current?.abort()
      const controller = new AbortController()
      activeController.current = controller
      const sequence = ++requestSequence.current
      const hasVisibleData = stateRef.current.pages.length > 0
      setError(null)
      setLoading(!hasVisibleData)
      setRefreshing(hasVisibleData)
      let cursor = initialCursor

      try {
        for (let attempt = 0; attempt < 2; attempt += 1) {
          const endpoint = requestUrl(url, cursor, initialLimit)
          const headers = new Headers({ Accept: 'application/json' })
          if (!isLocal && token) headers.set('Authorization', `Bearer ${token}`)
          const etag = etags.current.get(endpoint)
          if (etag) headers.set('If-None-Match', etag)

          const response = await fetch(endpoint, {
            method: 'GET',
            headers,
            signal: controller.signal,
          })
          if (sequence !== requestSequence.current) return

          if (response.status === 304) {
            dispatch({ type: 'not_modified' })
            return
          }
          if (response.status === 409) {
            const body: unknown = await response.json().catch(() => null)
            const revision = staleRevision(body)
            if (revision !== null && attempt === 0) {
              dispatch({ type: 'stale', revision })
              cursor = null
              continue
            }
          }
          if (!response.ok) {
            throw new Error(`Read model request failed: ${response.status}`)
          }

          const page = (await response.json()) as ReadPage<T>
          const responseEtag = response.headers.get('ETag')
          if (responseEtag) etags.current.set(endpoint, responseEtag)
          dispatch({
            type: 'received',
            cursor,
            page,
            etag: responseEtag,
          })
          return
        }
      } catch (requestError) {
        if (
          requestError instanceof DOMException &&
          requestError.name === 'AbortError'
        ) {
          return
        }
        if (sequence === requestSequence.current) {
          setError(
            requestError instanceof Error
              ? requestError
              : new Error('Read model request failed'),
          )
        }
      } finally {
        if (sequence === requestSequence.current) {
          setLoading(false)
          setRefreshing(false)
        }
      }
    },
    [initialLimit, token, url],
  )

  React.useEffect(() => {
    if (!url || (!isLocal && !token)) {
      activeController.current?.abort()
      setLoading(false)
      setRefreshing(false)
      return
    }
    dispatch({ type: 'query_changed', queryKey })
    void fetchPage(null)
    return () => {
      requestSequence.current += 1
      activeController.current?.abort()
      activeController.current = null
    }
  }, [fetchPage, queryKey, token, url])

  React.useEffect(() => {
    if (pollInterval <= 0 || !url || (!isLocal && !token)) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const schedule = () => {
      timer = setTimeout(async () => {
        if (stopped) return
        if (!document.hidden) {
          const current = stateRef.current.pages[stateRef.current.currentIndex]
          await fetchPage(current?.cursor ?? null)
        }
        if (!stopped) schedule()
      }, pollInterval)
    }
    const onVisible = () => {
      if (!document.hidden) {
        const current = stateRef.current.pages[stateRef.current.currentIndex]
        void fetchPage(current?.cursor ?? null)
      }
    }
    schedule()
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      stopped = true
      if (timer) clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [fetchPage, pollInterval, token, url])

  const current = state.pages[state.currentIndex]
  const items = current?.page.items ?? (EMPTY_ITEMS as T[])
  const loadNext = React.useCallback(() => {
    const latest = stateRef.current
    const active = latest.pages[latest.currentIndex]
    if (active?.page.next_cursor) void fetchPage(active.page.next_cursor)
  }, [fetchPage])
  const reload = React.useCallback(() => {
    const latest = stateRef.current
    const active = latest.pages[latest.currentIndex]
    return fetchPage(active?.cursor ?? null)
  }, [fetchPage])

  return {
    items,
    loading,
    refreshing,
    error,
    hasNext: Boolean(current?.page.next_cursor),
    loadNext,
    reload,
    revision: state.staleRevision ?? current?.page.revision ?? null,
  }
}
