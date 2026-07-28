'use client'

import * as React from 'react'

import { apiUrl } from '@/lib/runtime-env'
import type { ReadDocument } from '@/lib/types'

const isLocal = process.env.NEXT_PUBLIC_MODE === 'local'
const resolved = new Map<string, ReadDocument | null>()
const inFlight = new Map<string, Promise<ReadDocument | null>>()
const allControllers = new Set<AbortController>()

interface ResolverOptions {
  kbId: string
  token: string | null
  revision: number | null
  navigationKey: string
}

type ResolverKey =
  | { documentNumber: number; logicalReference?: never }
  | { documentNumber?: never; logicalReference: string }

function cacheKey(kbId: string, revision: number, key: ResolverKey): string {
  return key.documentNumber !== undefined
    ? `${kbId}|${revision}|number|${key.documentNumber}`
    : `${kbId}|${revision}|reference|${key.logicalReference}`
}

export function clearDocumentResolverCache() {
  for (const controller of allControllers) controller.abort()
  allControllers.clear()
  inFlight.clear()
  resolved.clear()
}

export function useDocumentResolver({
  kbId,
  token,
  revision,
  navigationKey,
}: ResolverOptions) {
  const ownedControllers = React.useRef(new Set<AbortController>())

  React.useEffect(
    () => () => {
      for (const controller of ownedControllers.current) controller.abort()
      ownedControllers.current.clear()
    },
    [navigationKey],
  )

  const resolve = React.useCallback(
    async (key: ResolverKey, signal?: AbortSignal): Promise<ReadDocument | null> => {
      if (!kbId || revision === null || (!isLocal && !token)) return null
      const normalized: ResolverKey = key.documentNumber !== undefined
        ? { documentNumber: key.documentNumber }
        : { logicalReference: key.logicalReference.trim() }
      const keyValue = cacheKey(kbId, revision, normalized)
      if (resolved.has(keyValue)) return resolved.get(keyValue) ?? null
      const existing = inFlight.get(keyValue)
      if (existing) return existing

      const controller = new AbortController()
      ownedControllers.current.add(controller)
      allControllers.add(controller)
      const abort = () => controller.abort(signal?.reason)
      if (signal?.aborted) abort()
      else signal?.addEventListener('abort', abort, { once: true })
      const params = new URLSearchParams()
      if (normalized.documentNumber !== undefined) {
        params.set('document_number', String(normalized.documentNumber))
      } else {
        params.set('logical_reference', normalized.logicalReference)
      }
      const headers = new Headers({ Accept: 'application/json' })
      if (!isLocal && token) headers.set('Authorization', `Bearer ${token}`)
      const request = fetch(
        `${apiUrl()}/v1/knowledge-bases/${kbId}/documents/resolve?${params.toString()}`,
        { headers, signal: controller.signal },
      ).then(async (response) => {
        if (response.status === 404) {
          resolved.set(keyValue, null)
          return null
        }
        if (!response.ok) {
          throw new Error(`Document resolution failed: ${response.status}`)
        }
        const document = (await response.json()) as ReadDocument
        resolved.set(keyValue, document)
        return document
      }).finally(() => {
        signal?.removeEventListener('abort', abort)
        ownedControllers.current.delete(controller)
        allControllers.delete(controller)
        inFlight.delete(keyValue)
      })
      inFlight.set(keyValue, request)
      return request
    },
    [kbId, revision, token],
  )

  return { resolve }
}
