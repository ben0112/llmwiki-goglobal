import { apiUrl } from './runtime-env'
const isLocal = process.env.NEXT_PUBLIC_MODE === 'local'

/** Thrown by apiFetch on non-2xx responses. Callers can branch on `.status`
 *  for clean retry logic (e.g. 409 conflict reconciliation). */
export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, message: string, detail: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError
}

function withRequestTimeout(signal: AbortSignal | null | undefined, timeoutMs: number) {
  const controller = new AbortController()
  let timedOut = false
  let timeoutId: ReturnType<typeof setTimeout> | undefined

  const abortFromCaller = () => controller.abort(signal?.reason)

  if (signal?.aborted) {
    abortFromCaller()
  } else {
    signal?.addEventListener('abort', abortFromCaller, { once: true })
    timeoutId = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, timeoutMs)
  }

  return {
    signal: controller.signal,
    didTimeout: () => timedOut,
    cleanup: () => {
      if (timeoutId) clearTimeout(timeoutId)
      signal?.removeEventListener('abort', abortFromCaller)
    },
  }
}

export async function apiFetch<T>(
  path: string,
  token: string,
  options?: RequestInit,
): Promise<T> {
  const { signal: callerSignal, ...fetchOptions } = options ?? {}
  const timeout = withRequestTimeout(callerSignal, 15000)
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers as Record<string, string>,
  }

  // In local mode, skip Authorization header (API doesn't check it)
  if (!isLocal && token) {
    headers.Authorization = `Bearer ${token}`
  }

  try {
    const res = await fetch(`${apiUrl()}${path}`, {
      ...fetchOptions,
      headers,
      signal: timeout.signal,
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      const message =
        typeof body?.detail === 'string'
          ? body.detail
          : `API error: ${res.status}`
      throw new ApiError(res.status, message, body)
    }
    if (res.status === 204) return undefined as T
    return res.json()
  } catch (err) {
    if (timeout.didTimeout()) throw new Error('API request timeout')
    throw err
  } finally {
    timeout.cleanup()
  }
}

export function getDocumentsWsUrl(kbId: string): string {
  return `${apiUrl().replace(/^http/, 'ws')}/v1/ws/documents/${kbId}`
}
