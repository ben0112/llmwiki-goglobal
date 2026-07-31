import { describe, expect, it } from 'vitest'

import {
  chunkManifest,
  createReadPageState,
  readPageReducer,
} from '@/lib/read-models'
import type { ReadPage } from '@/lib/types'

type Item = { id: string }

function page(id: string, nextCursor: string | null): ReadPage<Item> {
  return {
    revision: 7,
    items: [{ id }],
    next_cursor: nextCursor,
    total_count: 4,
  }
}

describe('read page cache', () => {
  it('keeps only the current page and its two nearest payloads', () => {
    let state = createReadPageState<Item>('all')
    for (const [cursor, id, next] of [
      [null, 'one', 'c2'],
      ['c2', 'two', 'c3'],
      ['c3', 'three', 'c4'],
      ['c4', 'four', null],
    ] as const) {
      state = readPageReducer(state, {
        type: 'received',
        cursor,
        page: page(id, next),
        etag: '"kb-read-7"',
      })
    }

    expect(state.pages.map((entry) => entry.page.items[0].id)).toEqual([
      'two',
      'three',
      'four',
    ])
    expect(state.cursorHistory).toEqual([null, 'c2', 'c3', 'c4'])
  })

  it('keeps state and item identity on 304', () => {
    const received = readPageReducer(createReadPageState<Item>('all'), {
      type: 'received',
      cursor: null,
      page: page('one', null),
      etag: '"kb-read-7"',
    })

    const unchanged = readPageReducer(received, { type: 'not_modified' })
    expect(unchanged).toBe(received)
    expect(unchanged.pages[0].page.items).toBe(received.pages[0].page.items)
  })

  it('restarts after 409 without clearing visible items', () => {
    const received = readPageReducer(createReadPageState<Item>('all'), {
      type: 'received',
      cursor: null,
      page: page('visible', 'c2'),
      etag: '"kb-read-7"',
    })
    const conflicted = readPageReducer(received, {
      type: 'stale',
      revision: 8,
    })

    expect(conflicted.pages[conflicted.currentIndex].page.items).toBe(
      received.pages[received.currentIndex].page.items,
    )
    expect(conflicted.cursorHistory).toEqual([])
    expect(conflicted.restartPending).toBe(true)
  })

  it('changes query key while retaining data until page one replaces it', () => {
    const received = readPageReducer(createReadPageState<Item>('all'), {
      type: 'received',
      cursor: null,
      page: page('visible', null),
      etag: '"kb-read-7"',
    })
    const changed = readPageReducer(received, {
      type: 'query_changed',
      queryKey: 'stage=S2',
    })

    expect(changed.queryKey).toBe('stage=S2')
    expect(changed.pages[0].page.items).toBe(received.pages[0].page.items)
    expect(changed.restartPending).toBe(true)
  })
})

it('chunks an unlimited folder manifest into bounded requests', () => {
  const files = Array.from({ length: 40_000 }, (_, index) => ({
    key: String(index),
  }))
  const chunks = chunkManifest(files, 200)
  expect(chunks).toHaveLength(200)
  expect(Math.max(...chunks.map((chunk) => chunk.length))).toBe(200)
  expect(chunks.flat()).toHaveLength(40_000)
  expect(chunks[0]).not.toBe(files)
})

it('rejects invalid manifest bounds', () => {
  expect(() => chunkManifest([1], 0)).toThrow(/positive/)
  expect(() => chunkManifest([1], 1.5)).toThrow(/integer/)
})
