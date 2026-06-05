import { create } from 'zustand'
import type { DocumentListItem } from '@/lib/types'

export type UploadPhase = 'uploading' | 'processing' | 'ready' | 'failed'

export interface UploadItem {
  id: string
  filename: string
  kbId: string
  kbSlug: string
  path: string
  progress: number
  phase: UploadPhase
  documentNumber: number | null
  error: string | null
}

interface OpenDocRequest {
  kbId: string
  documentNumber: number
  nonce: number
}

interface NewUpload {
  id: string
  filename: string
  kbId: string
  kbSlug: string
  path: string
}

interface UploadState {
  items: UploadItem[]
  openRequest: OpenDocRequest | null
  addUpload: (upload: NewUpload) => void
  setProgress: (id: string, progress: number) => void
  markProcessing: (id: string) => void
  markFailed: (id: string, error?: string | null) => void
  reconcileDocuments: (kbId: string, documents: DocumentListItem[]) => void
  dismiss: (id: string) => void
  clearFinished: () => void
  requestOpenDocument: (kbId: string, documentNumber: number) => void
  consumeOpenRequest: () => void
}

function matchDocument(item: UploadItem, documents: DocumentListItem[]): DocumentListItem | undefined {
  const name = item.filename.toLowerCase()
  return documents.find((d) => !d.archived && d.path === item.path && d.filename.toLowerCase() === name)
}

export const useUploadStore = create<UploadState>((set) => ({
  items: [],
  openRequest: null,

  addUpload: (upload) =>
    set((state) => ({
      items: [
        { ...upload, progress: 0, phase: 'uploading', documentNumber: null, error: null },
        ...state.items,
      ],
    })),

  setProgress: (id, progress) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id && item.phase === 'uploading' ? { ...item, progress } : item,
      ),
    })),

  markProcessing: (id) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id ? { ...item, phase: 'processing', progress: 1 } : item,
      ),
    })),

  markFailed: (id, error = null) =>
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id ? { ...item, phase: 'failed', error } : item,
      ),
    })),

  reconcileDocuments: (kbId, documents) =>
    set((state) => {
      let changed = false
      const items = state.items.map((item) => {
        if (item.kbId !== kbId || item.phase !== 'processing') return item
        const doc = matchDocument(item, documents)
        if (!doc) return item
        const phase: UploadPhase =
          doc.status === 'ready' ? 'ready' : doc.status === 'failed' ? 'failed' : 'processing'
        const error = doc.status === 'failed' ? doc.error_message : item.error
        if (phase === item.phase && doc.document_number === item.documentNumber && error === item.error) {
          return item
        }
        changed = true
        return { ...item, phase, documentNumber: doc.document_number, error }
      })
      return changed ? { items } : state
    }),

  dismiss: (id) => set((state) => ({ items: state.items.filter((item) => item.id !== id) })),

  clearFinished: () =>
    set((state) => ({
      items: state.items.filter((item) => item.phase === 'uploading' || item.phase === 'processing'),
    })),

  requestOpenDocument: (kbId, documentNumber) =>
    set((state) => ({ openRequest: { kbId, documentNumber, nonce: (state.openRequest?.nonce ?? 0) + 1 } })),

  consumeOpenRequest: () => set({ openRequest: null }),
}))
