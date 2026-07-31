import { describe, expect, it, vi } from 'vitest'

import {
  reconcileDocumentStatuses,
  runUploadPreflight,
} from '@/lib/upload-preflight'
import type {
  ReadDocumentStatus,
  UploadPreflightItem,
} from '@/lib/types'

function candidate(index: number, content = String(index)) {
  return {
    file: {
      name: `${index}.md`,
      size: content.length,
      arrayBuffer: async () => new TextEncoder().encode(content).buffer,
    } as File,
    path: '/',
  }
}

describe('runUploadPreflight', () => {
  it('bounds 40,000 descriptors to 200 calls with two concurrent requests', async () => {
    const calls: UploadPreflightItem[][] = []
    let active = 0
    let maxActive = 0
    const preflight = vi.fn(async (items: UploadPreflightItem[]) => {
      calls.push(items)
      active += 1
      maxActive = Math.max(maxActive, active)
      await Promise.resolve()
      active -= 1
      return items.map((item) => ({ ...item, accepted: true, code: 'accepted' as const }))
    })

    const result = await runUploadPreflight(
      Array.from({ length: 40_000 }, (_, index) => candidate(index)),
      { preflight, hashFile: false },
    )

    expect(calls).toHaveLength(200)
    expect(calls.every((call) => call.length <= 200)).toBe(true)
    expect(maxActive).toBe(2)
    expect(result.accepted).toHaveLength(40_000)
  }, 20_000)

  it('deduplicates destination names and hashes across chunks with two hash workers', async () => {
    let activeHashes = 0
    let maxHashes = 0
    const hash = vi.fn(async (file: File) => {
      activeHashes += 1
      maxHashes = Math.max(maxHashes, activeHashes)
      await Promise.resolve()
      activeHashes -= 1
      return file.name.startsWith('same-content') ? 'a'.repeat(64) : file.name.padEnd(64, '0').slice(0, 64)
    })
    const entries = Array.from({ length: 201 }, (_, index) => candidate(index))
    entries.push({ ...candidate(999), file: { ...candidate(999).file, name: '1.md' } as File })
    entries[0] = { ...entries[0], file: { ...entries[0].file, name: 'same-content-a.md' } as File }
    entries[200] = { ...entries[200], file: { ...entries[200].file, name: 'same-content-b.md' } as File }

    const result = await runUploadPreflight(entries, {
      hash,
      preflight: async (items) => items.map((item) => ({ ...item, accepted: true, code: 'accepted' })),
    })

    expect(maxHashes).toBe(2)
    expect(result.skipped.some((item) => item.reason === 'duplicate_name')).toBe(true)
    expect(result.skipped.some((item) => item.reason === 'duplicate_content')).toBe(true)
  })

  it('stops unsent chunks after cancellation', async () => {
    const controller = new AbortController()
    let calls = 0
    const result = await runUploadPreflight(
      Array.from({ length: 1_000 }, (_, index) => candidate(index)),
      {
        signal: controller.signal,
        hashFile: false,
        preflight: async (items) => {
          calls += 1
          controller.abort()
          return items.map((item) => ({ ...item, accepted: true, code: 'accepted' }))
        },
      },
    )

    expect(calls).toBeLessThanOrEqual(2)
    expect(result.cancelled).toBe(true)
  })
})

it('reconciles active upload identities in batches of 200', async () => {
  const calls: string[][] = []
  const statuses = await reconcileDocumentStatuses(
    Array.from({ length: 401 }, (_, index) => `doc-${index}`),
    async (ids) => {
      calls.push(ids)
      return ids.map<ReadDocumentStatus>((id) => ({
        id,
        document_number: null,
        status: 'processing',
        error_message: null,
        version: 1,
      }))
    },
  )

  expect(calls.map((call) => call.length)).toEqual([200, 200, 1])
  expect(statuses).toHaveLength(401)
})
