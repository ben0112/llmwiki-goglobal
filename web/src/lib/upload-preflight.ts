import { chunkManifest } from '@/lib/read-models'
import type {
  ReadDocumentStatus,
  UploadPreflightCode,
  UploadPreflightItem,
} from '@/lib/types'

const PREFLIGHT_BOUND = 200
const WORKER_COUNT = 2

export interface UploadManifestCandidate {
  file: File
  path: string
}

export interface UploadSkip {
  candidate: UploadManifestCandidate
  reason: Exclude<UploadPreflightCode, 'accepted'>
}

export interface UploadAccepted {
  candidate: UploadManifestCandidate
  decision: UploadPreflightItem
}

export interface UploadPreflightResult {
  accepted: UploadAccepted[]
  skipped: UploadSkip[]
  cancelled: boolean
}

interface UploadPreflightOptions {
  preflight: (
    items: UploadPreflightItem[],
    signal?: AbortSignal,
  ) => Promise<UploadPreflightItem[]>
  onAccepted?: (items: UploadAccepted[]) => void | Promise<void>
  hash?: (file: File) => Promise<string>
  hashFile?: boolean
  shouldHash?: (candidate: UploadManifestCandidate) => boolean
  signal?: AbortSignal
}

function createSemaphore(limit: number) {
  let active = 0
  const queue: Array<() => void> = []
  const acquire = async () => {
    if (active < limit) {
      active += 1
      return
    }
    await new Promise<void>((resolve) => queue.push(resolve))
    active += 1
  }
  const release = () => {
    active -= 1
    queue.shift()?.()
  }
  return async <T>(operation: () => Promise<T>): Promise<T> => {
    await acquire()
    try {
      return await operation()
    } finally {
      release()
    }
  }
}

async function sha256(file: File): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('')
}

function nameKey(candidate: UploadManifestCandidate): string {
  return `${candidate.path}\0${candidate.file.name.toLocaleLowerCase()}`
}

export async function runUploadPreflight(
  candidates: readonly UploadManifestCandidate[],
  options: UploadPreflightOptions,
): Promise<UploadPreflightResult> {
  const accepted: UploadAccepted[] = []
  const skipped: UploadSkip[] = []
  const names = new Set<string>()
  const unique: UploadManifestCandidate[] = []
  for (const candidate of candidates) {
    const key = nameKey(candidate)
    if (names.has(key)) {
      skipped.push({ candidate, reason: 'duplicate_name' })
    } else {
      names.add(key)
      unique.push(candidate)
    }
  }

  const chunks = chunkManifest(unique, PREFLIGHT_BOUND)
  const withHashPermit = createSemaphore(WORKER_COUNT)
  const digests = new Set<string>()
  const hash = options.hashFile === false
    ? null
    : options.hash ?? (typeof crypto !== 'undefined' && crypto.subtle ? sha256 : null)
  let nextChunk = 0

  const worker = async () => {
    while (!options.signal?.aborted) {
      const chunkIndex = nextChunk
      nextChunk += 1
      const chunk = chunks[chunkIndex]
      if (!chunk) return

      const prepared = await Promise.all(
        chunk.map(async (candidate) => {
          if (
            !hash ||
            options.signal?.aborted ||
            options.shouldHash?.(candidate) === false
          ) {
            return { candidate, sha256: null as string | null }
          }
          try {
            const digest = await withHashPermit(() => hash(candidate.file))
            if (options.signal?.aborted) return null
            if (digests.has(digest)) {
              skipped.push({ candidate, reason: 'duplicate_content' })
              return null
            }
            digests.add(digest)
            return { candidate, sha256: digest }
          } catch {
            return { candidate, sha256: null }
          }
        }),
      )
      if (options.signal?.aborted) return
      const uploadable = prepared.filter(
        (value): value is NonNullable<typeof value> => value !== null,
      )
      if (uploadable.length === 0) continue
      const descriptors = uploadable.map<UploadPreflightItem>(
        ({ candidate, sha256: digest }) => ({
          path: candidate.path,
          filename: candidate.file.name,
          size: candidate.file.size,
          sha256: digest,
        }),
      )
      const decisions = await options.preflight(descriptors, options.signal)
      if (options.signal?.aborted) return
      const acceptedChunk: UploadAccepted[] = []
      for (let index = 0; index < uploadable.length; index += 1) {
        const candidate = uploadable[index].candidate
        const decision = decisions[index]
        if (decision?.accepted && decision.code === 'accepted') {
          const value = { candidate, decision }
          accepted.push(value)
          acceptedChunk.push(value)
        } else {
          skipped.push({
            candidate,
            reason:
              decision?.code && decision.code !== 'accepted'
                ? decision.code
                : 'unsupported',
          })
        }
      }
      if (acceptedChunk.length > 0) {
        await options.onAccepted?.(acceptedChunk)
      }
    }
  }

  await Promise.all(Array.from({ length: WORKER_COUNT }, () => worker()))
  return { accepted, skipped, cancelled: Boolean(options.signal?.aborted) }
}

export async function reconcileDocumentStatuses(
  identities: readonly string[],
  fetchStatuses: (
    ids: string[],
    signal?: AbortSignal,
  ) => Promise<ReadDocumentStatus[]>,
  signal?: AbortSignal,
): Promise<ReadDocumentStatus[]> {
  const statuses: ReadDocumentStatus[] = []
  for (const ids of chunkManifest(identities, PREFLIGHT_BOUND)) {
    if (signal?.aborted) break
    statuses.push(...(await fetchStatuses(ids, signal)))
  }
  return statuses
}
