import type { ReadPage } from '@/lib/types'

export interface CachedReadPage<T> {
  cursor: string | null
  page: ReadPage<T>
  etag: string | null
}

export interface ReadPageState<T> {
  queryKey: string
  pages: CachedReadPage<T>[]
  currentIndex: number
  cursorHistory: Array<string | null>
  restartPending: boolean
  staleRevision: number | null
}

export type ReadPageAction<T> =
  | {
      type: 'received'
      cursor: string | null
      page: ReadPage<T>
      etag: string | null
    }
  | { type: 'not_modified' }
  | { type: 'stale'; revision: number }
  | { type: 'query_changed'; queryKey: string }
  | { type: 'navigate'; index: number }

export function createReadPageState<T>(queryKey: string): ReadPageState<T> {
  return {
    queryKey,
    pages: [],
    currentIndex: 0,
    cursorHistory: [],
    restartPending: false,
    staleRevision: null,
  }
}

export function readPageReducer<T>(
  state: ReadPageState<T>,
  action: ReadPageAction<T>,
): ReadPageState<T> {
  if (action.type === 'not_modified') return state

  if (action.type === 'navigate') {
    if (action.index < 0 || action.index >= state.pages.length) return state
    return { ...state, currentIndex: action.index }
  }

  if (action.type === 'stale') {
    return {
      ...state,
      cursorHistory: [],
      restartPending: true,
      staleRevision: action.revision,
    }
  }

  if (action.type === 'query_changed') {
    if (action.queryKey === state.queryKey) return state
    return {
      ...state,
      queryKey: action.queryKey,
      cursorHistory: [],
      restartPending: true,
      staleRevision: null,
    }
  }

  const entry: CachedReadPage<T> = {
    cursor: action.cursor,
    page: action.page,
    etag: action.etag,
  }
  if (
    action.cursor === null ||
    state.restartPending ||
    state.cursorHistory.length === 0
  ) {
    return {
      ...state,
      pages: [entry],
      currentIndex: 0,
      cursorHistory: [null],
      restartPending: false,
      staleRevision: null,
    }
  }

  const existingIndex = state.pages.findIndex(
    (cached) => cached.cursor === action.cursor,
  )
  if (existingIndex >= 0) {
    const pages = state.pages.slice()
    pages[existingIndex] = entry
    return {
      ...state,
      pages,
      currentIndex: existingIndex,
      restartPending: false,
      staleRevision: null,
    }
  }

  const appended = [...state.pages, entry]
  const pages = appended.slice(-3)
  const lastCursor = state.cursorHistory.at(-1)
  return {
    ...state,
    pages,
    currentIndex: pages.length - 1,
    cursorHistory:
      lastCursor === action.cursor
        ? state.cursorHistory
        : [...state.cursorHistory, action.cursor],
    restartPending: false,
    staleRevision: null,
  }
}

export function chunkManifest<T>(items: readonly T[], bound: number): T[][] {
  if (!Number.isInteger(bound)) {
    throw new RangeError('manifest chunk bound must be an integer')
  }
  if (bound <= 0) {
    throw new RangeError('manifest chunk bound must be positive')
  }
  const chunks: T[][] = []
  for (let start = 0; start < items.length; start += bound) {
    chunks.push(items.slice(start, start + bound))
  }
  return chunks
}
