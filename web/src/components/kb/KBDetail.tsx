'use client'

import * as React from 'react'
import { useSearchParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload as UploadIcon, BookOpen, ArrowUpRight, Loader2 } from 'lucide-react'
import { useUserStore, useUploadStore } from '@/stores'
import { useKBDocuments } from '@/hooks/useKBDocuments'
import { apiFetch } from '@/lib/api'
import { apiUrl } from '@/lib/runtime-env'
import { collectDroppedFiles } from '@/lib/dropFiles'
import { toast } from 'sonner'
import { KBSidenav } from '@/components/kb/KBSidenav'
import { SelectionActionBar } from '@/components/kb/SelectionActionBar'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { WikiContent } from '@/components/wiki/WikiContent'
import type { DocumentListItem, WikiNode } from '@/lib/types'
import type { ViewMode } from '@/components/kb/viewMode'


const FilesGrid = dynamic(() => import('@/components/kb/FilesGrid').then((mod) => mod.FilesGrid), {
  ssr: false,
  loading: () => <DetailPaneLoading />,
})

const GraphViewer = dynamic(() => import('@/components/kb/GraphViewer').then((mod) => mod.GraphViewer), {
  ssr: false,
  loading: () => <DetailPaneLoading />,
})

function DetailPaneLoading() {
  return (
    <div className="flex h-full items-center justify-center">
      <Loader2 className="size-5 animate-spin text-muted-foreground" />
    </div>
  )
}

function buildTreeFromDocs(docs: DocumentListItem[]): WikiNode[] {
  const sorted = [...docs].sort((a, b) => (a.sort_order ?? 999) - (b.sort_order ?? 999))
  const topLevel: Array<{ title: string; path: string; slug: string; docNumber: number | null }> = []
  const childPages = new Map<string, Array<{ title: string; path: string; docNumber: number | null }>>()

  for (const doc of sorted) {
    const relative = (doc.path + doc.filename).replace(/^\/wiki\/?/, '')
    const parts = relative.split('/')
    const title =
      doc.title ||
      parts[parts.length - 1].replace(/\.(md|txt|json)$/, '').replace(/[-_]/g, ' ')

    if (parts.length === 1) {
      const slug = parts[0].replace(/\.(md|txt|json)$/, '')
      topLevel.push({ title, path: relative, slug, docNumber: doc.document_number })
    } else {
      const folder = parts[0]
      if (!childPages.has(folder)) childPages.set(folder, [])
      childPages.get(folder)!.push({ title, path: relative, docNumber: doc.document_number })
    }
  }

  const tree: WikiNode[] = []
  const usedFolders = new Set<string>()

  for (const parent of topLevel) {
    const children = childPages.get(parent.slug)
    if (children && children.length > 0) {
      usedFolders.add(parent.slug)
      tree.push({
        title: parent.title, path: parent.path, docNumber: parent.docNumber,
        children: children.map((c) => ({ title: c.title, path: c.path, docNumber: c.docNumber })),
      })
    } else {
      tree.push({ title: parent.title, path: parent.path, docNumber: parent.docNumber })
    }
  }

  for (const [folder, children] of childPages) {
    if (usedFolders.has(folder)) continue
    const folderTitle = folder.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
    tree.push({ title: folderTitle, children: children.map((c) => ({ title: c.title, path: c.path, docNumber: c.docNumber })) })
  }

  const slug = (n: WikiNode) => n.path?.replace(/\.(md|txt|json)$/, '').split('/')[0] ?? ''
  tree.sort((a, b) => {
    const sa = slug(a), sb = slug(b)
    if (sa === 'overview') return -1
    if (sb === 'overview') return 1
    if (sa === 'log') return 1
    if (sb === 'log') return -1
    return a.title.localeCompare(b.title)
  })

  return tree
}

function enrichTreeWithDocNumbers(tree: WikiNode[], docs: DocumentListItem[]): WikiNode[] {
  const pathToDocNumber = new Map<string, number | null>()
  for (const doc of docs) {
    const relative = (doc.path + doc.filename).replace(/^\/wiki\/?/, '')
    pathToDocNumber.set(relative, doc.document_number)
  }
  function enrich(nodes: WikiNode[]): WikiNode[] {
    return nodes.map((node) => ({
      ...node,
      docNumber: node.path ? (pathToDocNumber.get(node.path) ?? null) : null,
      children: node.children ? enrich(node.children) : undefined,
    }))
  }
  return enrich(tree)
}

function findFirstPath(nodes: WikiNode[]): { path: string; docNumber: number | null } | null {
  for (const node of nodes) {
    if (node.path) return { path: node.path, docNumber: node.docNumber ?? null }
    if (node.children) {
      const found = findFirstPath(node.children)
      if (found) return found
    }
  }
  return null
}

type Props = {
  kbId: string
  kbSlug: string
  kbName: string
  viewMode: ViewMode
  routeFilesPath: string
}

export function KBDetail({ kbId, kbSlug, kbName, viewMode, routeFilesPath }: Props) {
  const searchParams = useSearchParams()
  const token = useUserStore((s) => s.accessToken)
  const userId = useUserStore((s) => s.user?.id)
  const { documents, setDocuments, loading } = useKBDocuments(kbId)

  // ─── Upload tracking (global progress panel) ─────────────────
  const addUpload = useUploadStore((s) => s.addUpload)
  const setUploadProgress = useUploadStore((s) => s.setProgress)
  const markUploadProcessing = useUploadStore((s) => s.markProcessing)
  const markUploadReady = useUploadStore((s) => s.markReady)
  const markUploadFailed = useUploadStore((s) => s.markFailed)
  const reconcileUploads = useUploadStore((s) => s.reconcileDocuments)
  const processingUploads = useUploadStore(
    (s) => s.items.filter((i) => i.kbId === kbId && i.phase === 'processing').length,
  )
  const openRequest = useUploadStore((s) => s.openRequest)
  const consumeOpenRequest = useUploadStore((s) => s.consumeOpenRequest)

  // ─── URL helpers ─────────────────────────────────────────────
  // Search param updates are instant (no Next.js route recompilation).
  // Path changes only happen on view-mode switches (rare).

  const updateParam = React.useCallback((key: string, value: string | null) => {
    const url = new URL(window.location.href)
    if (value != null) url.searchParams.set(key, value)
    else url.searchParams.delete(key)
    window.history.replaceState(window.history.state, '', url.pathname + url.search)
  }, [])

  const navigateToView = React.useCallback((view: ViewMode, opts?: { filesPath?: string; searchParams?: Record<string, string> }) => {
    let url = `/wikis/${kbSlug}`
    if (view === 'files') {
      const path = opts?.filesPath ?? '/'
      const clean = path === '/' ? '' : path.replace(/^\//, '').replace(/\/$/, '')
      url += clean ? `/files/${encodeURI(clean)}` : '/files'
    } else if (view === 'graph') {
      url += '/graph'
    }
    if (opts?.searchParams) {
      const sp = new URLSearchParams(opts.searchParams)
      url += '?' + sp.toString()
    }
    window.history.pushState(window.history.state, '', url)
  }, [kbSlug])

  // ─── Document splits ─────────────────────────────────────────
  const wikiDocs = React.useMemo(
    () => documents.filter((d) => (d.path === '/wiki/' || d.path.startsWith('/wiki/')) && !d.archived && d.file_type === 'md'),
    [documents],
  )
  const sourceDocs = React.useMemo(
    () => documents.filter((d) => !d.path.startsWith('/wiki/') && !d.archived),
    [documents],
  )

  // ─── View state ──────────────────────────────────────────────
  // activeView tracks the current tab. Initialized from the viewMode prop
  // (path segment) and kept in sync when the prop changes (back/forward).
  const [activeView, setActiveView] = React.useState<ViewMode | 'doc'>(viewMode)
  React.useEffect(() => { setActiveView(viewMode) }, [viewMode])

  const filesViewActive = activeView === 'files' || activeView === 'doc'
  const graphViewActive = activeView === 'graph'

  // ─── Wiki page selection (from ?p= search param) ─────────────
  const pParam = searchParams.get('p')
  const urlWikiDocNumber = pParam ? parseInt(pParam, 10) : null

  const [wikiActivePath, setWikiActivePath] = React.useState<string | null>(null)
  const lastWikiDocNumberRef = React.useRef<number | null>(urlWikiDocNumber)

  // Initialize wikiActivePath from ?p= on mount and when ?p= changes
  React.useEffect(() => {
    if (urlWikiDocNumber == null) return
    if (!documents.length) return
    const doc = documents.find((d) => d.document_number === urlWikiDocNumber)
    if (doc) {
      const path = (doc.path + doc.filename).replace(/^\/wiki\/?/, '')
      setWikiActivePath(path)
      lastWikiDocNumberRef.current = urlWikiDocNumber
    }
  }, [urlWikiDocNumber, documents])

  // ─── Source doc selection ────────────────────────────────────
  // Read ?doc= only on mount (for bookmarked URLs / browser back-forward)
  const initialDocParam = React.useRef(searchParams.get('doc'))
  const initialDocNumber = initialDocParam.current ? parseInt(initialDocParam.current, 10) : null

  const [activeSourceDocId, setActiveSourceDocId] = React.useState<string | null>(() => {
    if (initialDocNumber == null) return null
    const doc = documents.find((d) => d.document_number === initialDocNumber)
    return doc?.id ?? null
  })

  // Resolve initial ?doc= once documents load
  React.useEffect(() => {
    if (initialDocNumber == null || activeSourceDocId) return
    const doc = documents.find((d) => d.document_number === initialDocNumber)
    if (doc) {
      setActiveSourceDocId(doc.id)
      setActiveView('doc')
    }
  }, [initialDocNumber, documents, activeSourceDocId])

  const [filesInitialPage, setFilesInitialPage] = React.useState<number | undefined>()

  // ─── Graph state ─────────────────────────────────────────────
  const [graphFocusNodeId, setGraphFocusNodeId] = React.useState<string | null>(null)

  // ─── Wiki tree ───────────────────────────────────────────────
  const indexDoc = wikiDocs.find((d) => d.filename === 'index.json' && d.path === '/wiki/')
  const SCAFFOLD_FILES = new Set(['index.json', 'overview.md', 'log.md'])
  const hasNavigableWiki = React.useMemo(
    () => wikiDocs.some((d) => d.path === '/wiki/' ? !SCAFFOLD_FILES.has(d.filename) : true),
    [wikiDocs],
  )
  const [wikiTree, setWikiTree] = React.useState<WikiNode[]>([])
  const [indexLoaded, setIndexLoaded] = React.useState(false)

  const wikiDocIds = React.useMemo(() => wikiDocs.map((d) => d.id).join(), [wikiDocs])

  React.useEffect(() => {
    let cancelled = false
    setIndexLoaded(false)
    if (indexDoc && token) {
      apiFetch<{ content: string }>(`/v1/documents/${indexDoc.id}/content`, token)
        .then((res) => {
          if (cancelled) return
          try {
            const parsed = JSON.parse(res.content)
            setWikiTree(enrichTreeWithDocNumbers(parsed.tree || [], wikiDocs))
          } catch {
            setWikiTree(buildTreeFromDocs(wikiDocs.filter((d) => d.id !== indexDoc.id)))
          }
          setIndexLoaded(true)
        })
        .catch(() => {
          if (cancelled) return
          setWikiTree(buildTreeFromDocs(wikiDocs.filter((d) => d.id !== indexDoc.id)))
          setIndexLoaded(true)
        })
    } else {
      setWikiTree(buildTreeFromDocs(wikiDocs))
      setIndexLoaded(true)
    }
    return () => { cancelled = true }
  }, [indexDoc?.id, token, wikiDocIds, wikiDocs])

  // Auto-select first wiki page when none is selected
  React.useEffect(() => {
    if (indexLoaded && activeView === 'wiki' && !wikiActivePath && urlWikiDocNumber == null && wikiTree.length && !loading) {
      const first = findFirstPath(wikiTree)
      if (first) {
        setWikiActivePath(first.path)
        lastWikiDocNumberRef.current = first.docNumber
        if (first.docNumber != null) updateParam('p', String(first.docNumber))
      }
    }
  }, [indexLoaded, wikiTree, wikiActivePath, activeView, urlWikiDocNumber, loading, updateParam])

  // ─── Wiki content loading ────────────────────────────────────
  const [pageContent, setPageContent] = React.useState('')
  const [pageTitle, setPageTitle] = React.useState('')
  const [pageLoading, setPageLoading] = React.useState(false)
  const [pageLoadedPath, setPageLoadedPath] = React.useState<string | null>(null)

  const activeWikiDoc = React.useMemo(() => {
    if (!wikiActivePath) return null
    return wikiDocs.find((d) => {
      const relative = (d.path + d.filename).replace(/^\/wiki\/?/, '')
      return relative === wikiActivePath
    }) ?? null
  }, [wikiActivePath, wikiDocs])

  const activeWikiVersion = activeWikiDoc?.version ?? -1
  const activeWikiDocId = activeWikiDoc?.id ?? null

  React.useEffect(() => {
    if (!wikiActivePath || !token) {
      setPageLoadedPath(null)
      return
    }
    if (!activeWikiDoc) {
      setPageContent(`页面不存在: ${wikiActivePath}`)
      setPageTitle('')
      setPageLoadedPath(wikiActivePath)
      return
    }
    setPageTitle(activeWikiDoc.title || activeWikiDoc.filename.replace(/\.(md|txt)$/, ''))
    const isLiveUpdate = pageLoadedPath === wikiActivePath
    if (!isLiveUpdate) {
      setPageLoading(true)
      setPageLoadedPath(null)
    }
    const controller = new AbortController()
    apiFetch<{ content: string }>(`/v1/documents/${activeWikiDoc.id}/content`, token, { signal: controller.signal })
      .then((res) => {
        if (!controller.signal.aborted) setPageContent(res.content || '')
      })
      .catch((err) => {
        if (!controller.signal.aborted) setPageContent('页面内容加载失败。')
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPageLoading(false)
          setPageLoadedPath(wikiActivePath)
        }
      })
    return () => controller.abort()
  }, [wikiActivePath, token, activeWikiDocId, activeWikiVersion])

  // ─── Token helper ────────────────────────────────────────────
  const getToken = () => {
    const t = useUserStore.getState().accessToken
    if (!t) { toast.error('未登录'); return null }
    return t
  }

  // ─── Multi-selection ─────────────────────────────────────────
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set())
  const lastSelectedIdRef = React.useRef<string | null>(null)
  const sourceDocIds = React.useMemo(() => sourceDocs.map((d) => d.id), [sourceDocs])

  const handleSelect = React.useCallback((docId: string, e: React.MouseEvent) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (e.shiftKey && lastSelectedIdRef.current) {
        const lastIdx = sourceDocIds.indexOf(lastSelectedIdRef.current)
        const currIdx = sourceDocIds.indexOf(docId)
        if (lastIdx !== -1 && currIdx !== -1) {
          const [start, end] = lastIdx < currIdx ? [lastIdx, currIdx] : [currIdx, lastIdx]
          for (let i = start; i <= end; i++) next.add(sourceDocIds[i])
        } else {
          next.add(docId)
        }
      } else if (e.metaKey || e.ctrlKey) {
        if (next.has(docId)) next.delete(docId)
        else next.add(docId)
      } else {
        next.clear()
        next.add(docId)
      }
      lastSelectedIdRef.current = docId
      return next
    })
  }, [sourceDocIds])

  const clearSelection = React.useCallback(() => {
    setSelectedIds(new Set())
    lastSelectedIdRef.current = null
  }, [])

  React.useEffect(() => {
    if (selectedIds.size === 0) return
    const handleKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') clearSelection() }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [selectedIds.size, clearSelection])

  // ─── 删除(带引用影响预估 + 受影响维基页面自动重生成)────────
  const [deleteRequest, setDeleteRequest] = React.useState<{
    ids: string[]
    names: string[]
    impact: { id: string; title: string | null; path: string; filename: string }[]
    fromSelection: boolean
  } | null>(null)
  const [deleting, setDeleting] = React.useState(false)

  const requestDelete = React.useCallback(async (ids: string[], fromSelection = false) => {
    const t = getToken()
    if (!t || ids.length === 0) return
    const names = ids
      .map((id) => documents.find((d) => d.id === id)?.filename)
      .filter((n): n is string => !!n)
    let impact: { id: string; title: string | null; path: string; filename: string }[] = []
    try {
      const res = await apiFetch<{ pages: typeof impact; count: number }>(
        '/v1/documents/delete-impact', t,
        { method: 'POST', body: JSON.stringify({ ids }) },
      )
      impact = res.pages ?? []
    } catch { /* 影响预估失败不阻断删除流程 */ }
    setDeleteRequest({ ids, names, impact, fromSelection })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents])

  // 删除后后台重生成的完成提示:先见 running=true 再等落回 false;
  // 任务太快结束时用 finished_at(同机时钟,容忍 60s 偏差)兜底。
  const pollRegenCompletion = React.useCallback((expectedPages: number) => {
    if (process.env.NEXT_PUBLIC_MODE !== 'local') return
    const t = getToken()
    if (!t) return
    const startedAt = Date.now()
    let sawRunning = false
    const timer = window.setInterval(async () => {
      if (Date.now() - startedAt > 5 * 60_000) { window.clearInterval(timer); return }
      try {
        const s = await apiFetch<{ running: boolean; total: number; done: number; failed: number; finished_at: number | null }>(
          '/v1/documents/regen-status', t)
        if (s.running) { sawRunning = true; return }
        const justFinished = s.finished_at != null && s.finished_at * 1000 >= startedAt - 60_000
        if (sawRunning || justFinished) {
          window.clearInterval(timer)
          if (s.failed > 0) {
            toast.warning(`维基页面重新生成完成:成功 ${s.total - s.failed} 个,失败 ${s.failed} 个`)
          } else {
            toast.success(`已自动重新生成 ${s.total || expectedPages} 个引用维基页面`)
          }
        }
      } catch { /* 轮询失败静默重试 */ }
    }, 2000)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const performDelete = React.useCallback(async () => {
    if (!deleteRequest) return
    const t = getToken()
    if (!t) return
    setDeleting(true)
    const { ids, impact, fromSelection } = deleteRequest
    try {
      if (ids.length === 1) {
        await apiFetch(`/v1/documents/${ids[0]}`, t, { method: 'DELETE' })
      } else {
        await apiFetch('/v1/documents/bulk-delete', t, { method: 'POST', body: JSON.stringify({ ids }) })
      }
      setDocuments((prev) => prev.filter((d) => !ids.includes(d.id)))
      if (impact.length > 0 && process.env.NEXT_PUBLIC_MODE === 'local') {
        toast.info(`已删除 ${ids.length} 个文件;${impact.length} 个引用维基页面正在后台重新生成`)
        pollRegenCompletion(impact.length)
      }
      if (fromSelection) clearSelection()
      setDeleteRequest(null)
    } catch {
      toast.error('删除文件失败')
    } finally {
      setDeleting(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deleteRequest, clearSelection, pollRegenCompletion, setDocuments])

  const handleDeleteSelected = () => { requestDelete(Array.from(selectedIds), true) }

  // ─── Navigation handlers ─────────────────────────────────────

  // Wiki page click: only updates ?p= search param (instant, no route change)
  const handleWikiSelect = React.useCallback((path: string, docNumber?: number | null) => {
    setActiveView('wiki')
    setWikiActivePath(path)
    const num = docNumber ?? wikiDocs.find((d) => {
      const relative = (d.path + d.filename).replace(/^\/wiki\/?/, '')
      return relative === path
    })?.document_number ?? null
    lastWikiDocNumberRef.current = num
    if (num != null) updateParam('p', String(num))
  }, [updateParam, wikiDocs])

  const handleFilesToggle = React.useCallback(() => {
    if (activeView === 'doc') {
      // Doc is open — close it, go to root file browser
      setActiveSourceDocId(null)
      setActiveView('files')
      navigateToView('files')
    } else if (activeView === 'files') {
      // Already browsing files — toggle back to wiki
      const sp = lastWikiDocNumberRef.current != null
        ? { p: String(lastWikiDocNumberRef.current) }
        : undefined
      setActiveView('wiki')
      navigateToView('wiki', { searchParams: sp })
    } else {
      // From wiki/graph — switch to files
      setActiveView('files')
      navigateToView('files')
    }
  }, [activeView, navigateToView])

  const handleGraphToggle = React.useCallback(() => {
    if (graphViewActive) {
      const sp = lastWikiDocNumberRef.current != null
        ? { p: String(lastWikiDocNumberRef.current) }
        : undefined
      navigateToView('wiki', { searchParams: sp })
    } else {
      setActiveView('graph')
      setGraphFocusNodeId(null)
      navigateToView('graph')
    }
  }, [graphViewActive, navigateToView])

  const handleGraphNodeClick = React.useCallback((docId: string, sourceKind: string) => {
    const doc = documents.find((d) => d.id === docId)
    if (!doc) return
    if (sourceKind === 'wiki') {
      const wikiPath = (doc.path + doc.filename).replace(/^\/wiki\/?/, '')
      setActiveView('wiki')
      setWikiActivePath(wikiPath)
      lastWikiDocNumberRef.current = doc.document_number
      navigateToView('wiki', { searchParams: doc.document_number != null ? { p: String(doc.document_number) } : undefined })
      return
    }
    setActiveSourceDocId(doc.id)
    setActiveView('doc')
    navigateToView('files', { searchParams: doc.document_number != null ? { doc: String(doc.document_number) } : undefined })
  }, [documents, navigateToView])

  const handleOpenSourceDoc = React.useCallback((docId: string) => {
    const doc = documents.find((d) => d.id === docId)
    if (!doc) return
    setActiveSourceDocId(doc.id)
    setActiveView('doc')
    if (doc.document_number != null) {
      navigateToView('files', { searchParams: { doc: String(doc.document_number) } })
    }
  }, [documents, navigateToView])

  const handleCitationSourceClick = React.useCallback((filename: string, page?: number) => {
    const lower = filename.toLowerCase()
    const match = sourceDocs.find((d) => {
      const fn = d.filename.toLowerCase()
      const title = (d.title || '').toLowerCase()
      return fn === lower || title === lower || fn === lower + '.md' || fn.replace(/\.md$/, '') === lower
    })
    if (!match) return
    setActiveSourceDocId(match.id)
    setActiveView('doc')
    setFilesInitialPage(page)
    if (match.document_number != null) {
      navigateToView('files', { searchParams: { doc: String(match.document_number) } })
    }
  }, [sourceDocs, navigateToView])

  const handlePageGraphClick = React.useCallback(() => {
    if (!activeWikiDocId) return
    setActiveView('graph')
    setGraphFocusNodeId(activeWikiDocId)
    navigateToView('graph')
  }, [activeWikiDocId, navigateToView])

  const wikiPathSet = React.useMemo(() => {
    const set = new Set<string>()
    for (const d of wikiDocs) {
      const relative = (d.path + d.filename).replace(/^\/wiki\/?/, '')
      set.add(relative)
    }
    return set
  }, [wikiDocs])

  const handleWikiNavigate = React.useCallback(
    (path: string) => {
      let nextPath = path
      if (path.startsWith('/wiki/')) {
        nextPath = path.replace(/^\/wiki\/?/, '')
      } else if (path.startsWith('/')) {
        nextPath = path.slice(1)
      } else if (wikiPathSet.has(path)) {
        nextPath = path
      } else if (wikiActivePath) {
        const dir = wikiActivePath.includes('/')
          ? wikiActivePath.substring(0, wikiActivePath.lastIndexOf('/'))
          : ''
        let resolved = path.startsWith('./')
          ? (dir ? dir + '/' : '') + path.slice(2)
          : (dir ? dir + '/' : '') + path
        while (resolved.includes('../')) {
          resolved = resolved.replace(/[^/]*\/\.\.\//, '')
        }
        nextPath = resolved
      }
      setWikiActivePath(nextPath)
      const doc = wikiDocs.find((d) => {
        const relative = (d.path + d.filename).replace(/^\/wiki\/?/, '')
        return relative === nextPath
      })
      lastWikiDocNumberRef.current = doc?.document_number ?? null
      if (doc?.document_number != null) updateParam('p', String(doc.document_number))
    },
    [wikiActivePath, wikiPathSet, updateParam, wikiDocs],
  )


  // ─── Document CRUD ───────────────────────────────────────────
  const handleCreateNote = async (targetPath: string = '/') => {
    const t = getToken()
    if (!t || !userId) return
    try {
      const data = await apiFetch<DocumentListItem>(`/v1/knowledge-bases/${kbId}/documents/note`, t, {
        method: 'POST',
        body: JSON.stringify({ filename: '无标题.md', path: targetPath }),
      })
      setDocuments((prev) => [data, ...prev])
      if (!filesViewActive) {
        setActiveView('files')
        navigateToView('files')
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '创建笔记失败')
    }
  }

  const handleCreateFolder = (folderName: string, parentPath: string = '/') => {
    const t = getToken()
    if (!t || !userId) return
    const path = parentPath.replace(/\/$/, '') + '/' + folderName + '/'
    apiFetch<DocumentListItem>(`/v1/knowledge-bases/${kbId}/documents/note`, t, {
      method: 'POST',
      body: JSON.stringify({ filename: '无标题.md', path }),
    })
      .then((data) => {
        setDocuments((prev) => [data, ...prev])
        if (!filesViewActive) {
          setActiveView('files')
          navigateToView('files')
        }
      })
      .catch((err: Error) => toast.error(err.message || '创建文件夹失败'))
  }

  const handleMoveDocument = async (docId: string, targetPath: string) => {
    const t = getToken()
    if (!t) return
    try {
      await apiFetch(`/v1/documents/${docId}`, t, { method: 'PATCH', body: JSON.stringify({ path: targetPath }) })
      setDocuments((prev) => prev.map((d) => d.id === docId ? { ...d, path: targetPath } : d))
    } catch { toast.error('移动文档失败') }
  }

  const handleDeleteDocument = async (docId: string) => {
    await requestDelete([docId])
  }

  const handleRenameDocument = async (docId: string, newTitle: string) => {
    const t = getToken()
    if (!t) return
    try {
      await apiFetch(`/v1/documents/${docId}`, t, { method: 'PATCH', body: JSON.stringify({ title: newTitle }) })
      setDocuments((prev) => prev.map((d) => d.id === docId ? { ...d, title: newTitle } : d))
    } catch { toast.error('重命名文档失败') }
  }

  // ─── File upload ─────────────────────────────────────────────
  const uploadPathRef = React.useRef('/')
  const handleUploadClick = (targetPath: string = '/') => {
    uploadPathRef.current = targetPath
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.md,.txt,.pdf,.pptx,.ppt,.docx,.doc,.png,.jpg,.jpeg,.webp,.gif,.svg,.xlsx,.xls,.csv,.html,.htm,.zip,.tar,.tgz,.gz'
    input.multiple = true
    input.onchange = () => { if (input.files) uploadFiles(Array.from(input.files), uploadPathRef.current) }
    input.click()
  }

  // 选择目录并递归上传其下全部文件(含子文件夹;webkitRelativePath 保留结构)
  const handleUploadFolderClick = (targetPath: string = '/') => {
    uploadPathRef.current = targetPath
    const input = document.createElement('input')
    input.type = 'file'
    ;(input as HTMLInputElement & { webkitdirectory: boolean }).webkitdirectory = true
    input.onchange = () => { if (input.files) uploadFiles(Array.from(input.files), uploadPathRef.current) }
    input.click()
  }

  const tusUploadFile = React.useCallback(async (file: File, targetPath: string = '/'): Promise<void> => {
    const t = getToken()
    if (!t) return Promise.reject(new Error('未登录'))
    const uploadId = crypto.randomUUID()
    addUpload({ id: uploadId, filename: file.name, kbId, kbSlug, path: targetPath })
    const { Upload } = await import('tus-js-client')
    return new Promise((resolve, reject) => {
      const upload = new Upload(file, {
        endpoint: `${apiUrl()}/v1/uploads`,
        retryDelays: [0, 1000, 3000, 5000],
        // 50MB 分块:1GB 上限下单个 PATCH 不至于撑爆反向代理 body 限制
        chunkSize: 50 * 1024 * 1024,
        metadata: { filename: file.name, knowledge_base_id: kbId, path: targetPath },
        headers: { Authorization: `Bearer ${t}` },
        onProgress: (sent, total) => setUploadProgress(uploadId, total > 0 ? sent / total : 0),
        onError: (error) => { markUploadFailed(uploadId); reject(error) },
        onSuccess: () => { markUploadProcessing(uploadId); resolve() },
      })
      upload.start()
    })
  }, [kbId, kbSlug, addUpload, setUploadProgress, markUploadProcessing, markUploadFailed])

  const uploadFiles = React.useCallback(async (
    files: File[], targetPath: string = '/', relMap?: Map<File, string>,
  ) => {
    const t = getToken()
    if (!t || !userId) return

    // 目录上传/文件夹拖放:webkitRelativePath 或 relMap 带出相对路径,
    // 逐文件保留子目录结构(目标 = targetPath + 相对目录/)。
    const relOf = (file: File): string =>
      relMap?.get(file) ?? (file as File & { webkitRelativePath?: string }).webkitRelativePath ?? ''
    const destOf = (file: File): string => {
      const rel = relOf(file)
      if (!rel || !rel.includes('/')) return targetPath
      // 剥离最外层文件夹名:保留内部结构、不额外新建顶层目录——
      // 内容落在当前所在文件夹(或根)下,避免重复嵌套(如 A/A/…)
      const dir = rel.split('/').slice(1, -1).join('/')
      return dir ? targetPath.replace(/\/$/, '') + '/' + dir + '/' : targetPath
    }
    const JUNK = new Set(['.ds_store', 'thumbs.db', 'desktop.ini'])
    const isJunk = (file: File): boolean =>
      JUNK.has(file.name.toLowerCase()) || file.name.startsWith('.')
      || relOf(file).split('/').some((p) => p.startsWith('.') || p === '__MACOSX')
    const isArchive = (name: string): boolean => /\.(zip|tar|tgz)$/i.test(name) || /\.tar\.gz$/i.test(name)

    // 双层去重,自动跳过(不打断上传,仅在完成报告中提及):
    // 第一层:目标位置同名文件(不区分大小写);
    // 第二层:文件内容 SHA-256 与库内任意既有文件相同(本地模式;
    // 后端 content_hash 即原始字节哈希),同批内重复内容也只传一份。
    // 压缩包本身不入库,跳过去重(包内条目由服务端按同口径去重)。
    const namesByPath = new Map<string, Set<string>>()
    for (const d of documents) {
      if (d.archived) continue
      const set = namesByPath.get(d.path) ?? new Set<string>()
      set.add(d.filename.toLowerCase())
      namesByPath.set(d.path, set)
    }
    const existingHashes = new Set(
      documents.filter((d) => !d.archived && d.content_hash).map((d) => d.content_hash as string),
    )
    const MAX_UPLOAD_BYTES = 1024 * 1024 * 1024      // 1 GiB,与后端一致
    const HASH_MAX_BYTES = 64 * 1024 * 1024          // 超过则跳过浏览器内容哈希(避免整读进内存)
    const canHash = process.env.NEXT_PUBLIC_MODE === 'local'
      && typeof crypto !== 'undefined' && !!crypto.subtle
    const skippedByName: string[] = []
    const skippedByContent: string[] = []
    const oversizeNames: string[] = []
    const batchHashes = new Set<string>()
    const batchNames = new Set<string>()
    const toUpload: File[] = []
    for (const file of files) {
      if (isJunk(file)) continue // 系统垃圾文件静默丢弃
      if (file.size > MAX_UPLOAD_BYTES) {
        oversizeNames.push(file.name)
        continue
      }
      if (isArchive(file.name)) {
        toUpload.push(file)
        continue
      }
      const dest = destOf(file)
      const nameKey = dest + ' ' + file.name.toLowerCase()
      if (namesByPath.get(dest)?.has(file.name.toLowerCase()) || batchNames.has(nameKey)) {
        skippedByName.push(file.name)
        continue
      }
      if (canHash && file.size <= HASH_MAX_BYTES) {
        try {
          const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
          const hex = Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('')
          if (existingHashes.has(hex) || batchHashes.has(hex)) {
            skippedByContent.push(file.name)
            continue
          }
          batchHashes.add(hex)
        } catch { /* 哈希失败不阻断上传 */ }
      }
      batchNames.add(nameKey)
      toUpload.push(file)
    }
    const skippedTotal = skippedByName.length + skippedByContent.length
    if (toUpload.length === 0) {
      if (skippedTotal > 0 || oversizeNames.length > 0) {
        const names = [...skippedByName, ...skippedByContent, ...oversizeNames]
        const parts = [
          skippedByName.length > 0 ? `重名 ${skippedByName.length} 个` : '',
          skippedByContent.length > 0 ? `内容相同 ${skippedByContent.length} 个` : '',
          oversizeNames.length > 0 ? `超过 1GB 上限 ${oversizeNames.length} 个` : '',
        ].filter(Boolean).join(',')
        toast.info(`未上传:${names.length} 个文件被跳过(${parts})`, {
          description: names.slice(0, 5).join('、') + (names.length > 5 ? ` 等 ${names.length} 个` : ''),
        })
      }
      return
    }

    // 断点续传:大文件分块上传,中断后凭 upload_id 查询进度续传;
    // upload_id 存 localStorage,刷新页面后重传同一文件也能接着传
    const RESUMABLE_THRESHOLD = 32 * 1024 * 1024
    const uploadResumable = async (file: File, dest: string, uploadId: string): Promise<DocumentListItem> => {
      const storeKey = `lwup:${kbId}:${dest}:${file.name}:${file.size}:${file.lastModified}`
      let id: string | null = localStorage.getItem(storeKey)
      let offset = 0
      if (id) {
        try {
          offset = (await apiFetch<{ offset: number }>(`/v1/upload/resumable/${id}`, t)).offset
        } catch { id = null }
      }
      if (!id) {
        const init = await apiFetch<{ upload_id: string; offset: number }>('/v1/upload/resumable', t, {
          method: 'POST',
          body: JSON.stringify({ filename: file.name, path: dest, size: file.size }),
        })
        id = init.upload_id
        offset = init.offset
        localStorage.setItem(storeKey, id)
      }
      const CHUNK = 8 * 1024 * 1024
      let failures = 0
      while (offset < file.size) {
        try {
          const res = await fetch(`${apiUrl()}/v1/upload/resumable/${id}?offset=${offset}`, {
            method: 'PATCH',
            body: file.slice(offset, offset + CHUNK),
          })
          if (res.status === 409) { offset = (await res.json()).offset; continue }
          if (!res.ok) throw new Error(`分块上传失败:${res.status}`)
          offset = (await res.json()).offset
          failures = 0
          setUploadProgress(uploadId, (offset / file.size) * 0.98)
        } catch (err) {
          failures += 1
          if (failures > 5) throw err
          // 网络抖动:退避后向服务端查询已收 offset,从断点继续
          await new Promise((resolve) => setTimeout(resolve, 1000 * failures))
          try {
            offset = (await apiFetch<{ offset: number }>(`/v1/upload/resumable/${id}`, t)).offset
          } catch { /* 查询失败则按本地 offset 重试 */ }
        }
      }
      const doc = await apiFetch<DocumentListItem>(`/v1/upload/resumable/${id}/complete`, t, { method: 'POST' })
      localStorage.removeItem(storeKey)
      return doc
    }

    const supportedTypes = new Set(['pdf', 'pptx', 'ppt', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'webp', 'gif', 'xlsx', 'xls', 'csv', 'html', 'htm'])
    const results: { name: string; ok: boolean }[] = []
    const unsupportedNames: string[] = []
    let extractedCreated = 0
    let archiveSkipped = 0

    const uploadOne = async (file: File): Promise<boolean | null> => {
      const dest = destOf(file)
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (isArchive(file.name)) {
        if (process.env.NEXT_PUBLIC_MODE !== 'local') {
          unsupportedNames.push(file.name) // 托管模式暂不支持服务端解压
          return null
        }
        // 压缩包:上传后服务端解压入库(zip/tar[.gz],以包名建文件夹)
        const uploadId = crypto.randomUUID()
        addUpload({ id: uploadId, filename: file.name, kbId, kbSlug, path: dest })
        const formData = new FormData()
        formData.append('file', file)
        formData.append('path', dest)
        try {
          const data = await new Promise<{
            created: number; skipped_duplicate: number; skipped_unsupported: number
            documents: DocumentListItem[]
          }>((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.open('POST', `${apiUrl()}/v1/upload/archive`)
            xhr.upload.onprogress = (e) => {
              // 上传占 90%,余下留给服务端解压阶段
              if (e.lengthComputable) setUploadProgress(uploadId, (e.loaded / e.total) * 0.9)
            }
            xhr.onload = () => {
              if (xhr.status >= 200 && xhr.status < 300) {
                try { resolve(JSON.parse(xhr.responseText)) } catch { reject(new Error('响应解析失败')) }
              } else {
                let detail = `上传失败:${xhr.status}`
                try { detail = JSON.parse(xhr.responseText).detail || detail } catch { /* keep default */ }
                reject(new Error(detail))
              }
            }
            xhr.onerror = () => reject(new Error('网络错误'))
            xhr.send(formData)
          })
          setDocuments((prev) => [...data.documents, ...prev])
          extractedCreated += data.created
          archiveSkipped += data.skipped_duplicate + data.skipped_unsupported
          markUploadReady(uploadId)
          return true
        } catch (err) {
          markUploadFailed(uploadId, err instanceof Error ? err.message : null)
          return false
        }
      }
      if (ext === 'md' || ext === 'txt') {
        // 文本文件走笔记导入,也进上传面板,让批量上传有统一进度
        const uploadId = crypto.randomUUID()
        addUpload({ id: uploadId, filename: file.name, kbId, kbSlug, path: dest })
        try {
          const content = await file.text()
          const title = file.name.replace(/\.(md|txt)$/i, '')
          const data = await apiFetch<DocumentListItem>(`/v1/knowledge-bases/${kbId}/documents/note`, t, {
            method: 'POST',
            body: JSON.stringify({ filename: file.name, title, content, path: dest }),
          })
          setDocuments((prev) => [data, ...prev])
          setUploadProgress(uploadId, 1)
          markUploadProcessing(uploadId)
          return true
        } catch {
          markUploadFailed(uploadId, '导入失败')
          return false
        }
      }
      if (!ext || !supportedTypes.has(ext)) {
        unsupportedNames.push(file.name)
        return null
      }
      if (process.env.NEXT_PUBLIC_MODE === 'local') {
        const uploadId = crypto.randomUUID()
        addUpload({ id: uploadId, filename: file.name, kbId, kbSlug, path: dest })
        try {
          let data: DocumentListItem
          if (file.size > RESUMABLE_THRESHOLD) {
            // 大文件走断点续传(分块 + 断点恢复)
            data = await uploadResumable(file, dest, uploadId)
          } else {
            // 小文件走单请求 XHR:浏览器上传进度事件驱动面板进度条
            const formData = new FormData()
            formData.append('file', file)
            formData.append('path', dest)
            data = await new Promise<DocumentListItem>((resolve, reject) => {
              const xhr = new XMLHttpRequest()
              xhr.open('POST', `${apiUrl()}/v1/upload`)
              xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) setUploadProgress(uploadId, e.loaded / e.total)
              }
              xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                  try { resolve(JSON.parse(xhr.responseText)) } catch { reject(new Error('响应解析失败')) }
                } else reject(new Error(`上传失败:${xhr.status}`))
              }
              xhr.onerror = () => reject(new Error('网络错误'))
              xhr.send(formData)
            })
          }
          setDocuments((prev) => [data, ...prev])
          markUploadProcessing(uploadId)
          return true
        } catch (err) {
          markUploadFailed(uploadId, err instanceof Error ? err.message : null)
          return false
        }
      }
      try {
        await tusUploadFile(file, dest)
        return true
      } catch {
        return false
      }
    }

    // 并发池:目录上传可能带来上百个文件,限流避免同时打开过多请求
    const queue = [...toUpload]
    const workers = Array.from({ length: Math.min(5, queue.length) }, async () => {
      for (;;) {
        const file = queue.shift()
        if (!file) return
        const ok = await uploadOne(file)
        if (ok !== null) results.push({ name: file.name, ok })
      }
    })
    await Promise.all(workers)

    // 完成汇总:成功/失败数量确认(+ 解压导入与自动跳过的重复/不支持)
    const okCount = results.filter((r) => r.ok).length
    const failed = results.filter((r) => !r.ok).map((r) => r.name)
    const extras = [
      extractedCreated > 0 ? `解压导入 ${extractedCreated} 个` : '',
      archiveSkipped > 0 ? `压缩包内跳过 ${archiveSkipped} 个` : '',
      skippedByName.length > 0 ? `跳过重名 ${skippedByName.length} 个` : '',
      skippedByContent.length > 0 ? `跳过内容相同 ${skippedByContent.length} 个` : '',
      oversizeNames.length > 0 ? `超过 1GB ${oversizeNames.length} 个` : '',
      unsupportedNames.length > 0 ? `不支持 ${unsupportedNames.length} 个` : '',
    ].filter(Boolean).join(',')
    const suffix = extras ? `(${extras})` : ''
    const skippedNames = [...skippedByName, ...skippedByContent, ...oversizeNames, ...unsupportedNames]
    const skippedDesc = skippedNames.length > 0
      ? `已跳过:${skippedNames.slice(0, 5).join('、')}${skippedNames.length > 5 ? ` 等 ${skippedNames.length} 个` : ''}`
      : undefined
    if (results.length > 0) {
      if (failed.length === 0) {
        toast.success(`上传完成:${okCount} 个文件全部成功${suffix}`, { description: skippedDesc })
      } else {
        toast.error(`上传完成:成功 ${okCount} 个,失败 ${failed.length} 个${suffix}`, {
          description: [
            failed.slice(0, 5).join('、') + (failed.length > 5 ? ` 等 ${failed.length} 个` : ''),
            skippedDesc,
          ].filter(Boolean).join(';'),
          duration: 10000,
        })
      }
    } else if (unsupportedNames.length > 0) {
      toast.info(`未上传:${unsupportedNames.length} 个文件类型不支持`, {
        description: unsupportedNames.slice(0, 5).join('、')
          + (unsupportedNames.length > 5 ? ` 等 ${unsupportedNames.length} 个` : ''),
      })
    }

    // Navigate to files view after first upload
    if (sourceDocs.length === 0 && okCount > 0) {
      setActiveView('files')
      navigateToView('files')
    }
  }, [kbId, kbSlug, userId, tusUploadFile, documents, sourceDocs.length, navigateToView, addUpload, setUploadProgress, markUploadProcessing, markUploadReady, markUploadFailed])

  React.useEffect(() => {
    reconcileUploads(kbId, documents)
  }, [kbId, documents, processingUploads, reconcileUploads])

  React.useEffect(() => {
    if (!openRequest || openRequest.kbId !== kbId) return
    const doc = documents.find((d) => d.document_number === openRequest.documentNumber)
    if (!doc) return
    setActiveSourceDocId(doc.id)
    setActiveView('doc')
    navigateToView('files', { searchParams: { doc: String(openRequest.documentNumber) } })
    consumeOpenRequest()
  }, [openRequest, kbId, documents, navigateToView, consumeOpenRequest])

  // ─── Drag-and-drop ───────────────────────────────────────────
  const [fileDragOver, setFileDragOver] = React.useState(false)
  const dragCounterRef = React.useRef(0)

  const handleFileDragEnter = (e: React.DragEvent) => {
    if (filesViewActive) return
    if (e.dataTransfer.types.includes('application/x-llmwiki-item')) return
    e.preventDefault()
    dragCounterRef.current++
    if (dragCounterRef.current === 1) setFileDragOver(true)
  }
  const handleFileDragLeave = (e: React.DragEvent) => {
    if (filesViewActive) return
    e.preventDefault()
    dragCounterRef.current--
    if (dragCounterRef.current === 0) setFileDragOver(false)
  }
  const handleFileDragOver = (e: React.DragEvent) => {
    if (filesViewActive) return
    if (e.dataTransfer.types.includes('application/x-llmwiki-item')) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }
  const handleFileDrop = (e: React.DragEvent) => {
    if (filesViewActive) return
    if (e.dataTransfer.types.includes('application/x-llmwiki-item')) return
    e.preventDefault()
    dragCounterRef.current = 0
    setFileDragOver(false)
    // 文件夹拖放:递归遍历目录树,保留相对路径(必须在事件内同步启动)
    collectDroppedFiles(e.dataTransfer).then(({ files, relMap }) => {
      if (files.length > 0) uploadFiles(files, '/', relMap)
    })
  }

  // ─── FilesGrid URL-sync callbacks ────────────────────────────
  const handleFilesPathChange = React.useCallback((path: string) => {
    const clean = path === '/' ? '' : path.replace(/^\//, '').replace(/\/$/, '')
    const url = `/wikis/${kbSlug}` + (clean ? `/files/${encodeURI(clean)}` : '/files')
    // pushState so each folder is a back-button stop; FilesGrid already holds the path.
    window.history.pushState(window.history.state, '', url)
  }, [kbSlug])

  const handleFilesDocOpen = React.useCallback((docNumber: number | null) => {
    if (docNumber == null) return
    const doc = documents.find((d) => d.document_number === docNumber)
    if (doc) {
      setActiveSourceDocId(doc.id)
      setActiveView('doc')
      updateParam('doc', String(docNumber))
    }
  }, [documents, updateParam])

  const handleFilesDocClose = React.useCallback(() => {
    setActiveSourceDocId(null)
    setActiveView('files')
    updateParam('doc', null)
  }, [updateParam])

  // ─── Loading state ───────────────────────────────────────────
  const showMainLoading =
    loading ||
    (!filesViewActive && !graphViewActive && hasNavigableWiki && !wikiActivePath) ||
    (!filesViewActive && !graphViewActive && !!wikiActivePath && pageLoadedPath !== wikiActivePath)

  // ─── Render ──────────────────────────────────────────────────
  return (
    <div
      className="flex flex-col h-full relative"
      onDragEnter={handleFileDragEnter}
      onDragLeave={handleFileDragLeave}
      onDragOver={handleFileDragOver}
      onDrop={handleFileDrop}
    >
      <AnimatePresence>
        {fileDragOver && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center pointer-events-none"
          >
            <div className="flex flex-col items-center gap-3 border-2 border-dashed border-primary rounded-xl px-12 py-10">
              <UploadIcon className="size-8 text-primary" />
              <p className="text-sm font-medium text-primary">松开鼠标即可上传</p>
              <p className="text-xs text-muted-foreground">支持 PDF、Word、PowerPoint、图片等格式</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 overflow-hidden flex">
        <div className="shrink-0">
          <KBSidenav
            kbId={kbId}
            kbName={kbName}
            wikiTree={wikiTree}
            wikiActivePath={filesViewActive || graphViewActive ? null : wikiActivePath}
            onWikiNavigate={handleWikiSelect}
            sourceDocs={sourceDocs}
            wikiDocs={wikiDocs}
            hasWiki={hasNavigableWiki}
            loading={loading}
            onUpload={() => handleUploadClick()}
            filesViewActive={filesViewActive}
            onFilesToggle={handleFilesToggle}
            graphViewActive={graphViewActive}
            onGraphToggle={handleGraphToggle}
            onOpenSourceDoc={handleOpenSourceDoc}
          />
        </div>
        <div className="flex-1 min-w-0">
          <AnimatePresence mode="wait">
            {showMainLoading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.1 }}
                className="flex items-center justify-center h-full"
              >
                <Loader2 className="size-5 animate-spin text-muted-foreground" />
              </motion.div>
            ) : graphViewActive ? (
              <motion.div
                key="graph"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15, ease: [0.25, 0.1, 0.25, 1] }}
                className="h-full"
              >
                <GraphViewer
                  kbId={kbId}
                  focusNodeId={graphFocusNodeId}
                  onNavigateToDoc={handleGraphNodeClick}
                />
              </motion.div>
            ) : filesViewActive ? (
              <motion.div
                key="files"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15, ease: [0.25, 0.1, 0.25, 1] }}
                className="h-full"
              >
                <FilesGrid
                  key={kbId}
                  documents={documents}
                  onDeleteDocument={handleDeleteDocument}
                  onRenameDocument={handleRenameDocument}
                  onUpload={handleUploadClick}
                  onUploadFolder={handleUploadFolderClick}
                  onCreateNote={handleCreateNote}
                  onCreateFolder={handleCreateFolder}
                  onMoveDocument={handleMoveDocument}
                  onUploadFiles={uploadFiles}
                  initialDocId={activeSourceDocId}
                  initialPage={filesInitialPage}
                  initialPath={routeFilesPath}
                  onPathChange={handleFilesPathChange}
                  onDocOpen={handleFilesDocOpen}
                  onDocClose={handleFilesDocClose}
                />
              </motion.div>
            ) : pageLoading ? (
              <motion.div
                key="wiki-loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.1 }}
                className="flex items-center justify-center h-full"
              >
                <Loader2 className="size-5 animate-spin text-muted-foreground" />
              </motion.div>
            ) : hasNavigableWiki && wikiActivePath ? (
              <motion.div
                key={`wiki-${wikiActivePath}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15, ease: [0.25, 0.1, 0.25, 1] }}
                className="h-full"
              >
                <WikiContent
                  content={pageContent}
                  title={pageTitle}
                  path={wikiActivePath ?? undefined}
                  documentId={activeWikiDocId}
                  onNavigate={handleWikiNavigate}
                  onSourceClick={handleCitationSourceClick}
                  onGraphClick={handlePageGraphClick}
                  documents={documents}
                />
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
                className="flex flex-col items-center justify-center h-full gap-4 px-6"
              >
                <BookOpen className="size-10 text-muted-foreground/20" />
                <div className="text-center max-w-sm">
                  <h3 className="text-base font-medium mb-1.5">还没有维基</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    先添加一些资料,然后让 Claude 从中编译出一个维基。
                  </p>
                </div>
                <div className="flex items-center gap-3 mt-2">
                  <button
                    onClick={() => handleUploadClick()}
                    className="inline-flex items-center gap-2 rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
                  >
                    <UploadIcon className="size-3.5 opacity-60" />
                    上传源文件
                  </button>
                  <a
                    href="https://claude.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 rounded-full border border-border px-5 py-2 text-sm font-medium hover:bg-accent transition-colors"
                  >
                    打开 Claude
                    <ArrowUpRight className="size-3.5 opacity-60" />
                  </a>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <Dialog open={!!deleteRequest} onOpenChange={(open) => { if (!open && !deleting) setDeleteRequest(null) }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除文件</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>
              将永久删除{' '}
              {deleteRequest?.ids.length === 1 ? (
                <strong className="text-foreground">{deleteRequest.names[0] ?? '该文件'}</strong>
              ) : (
                <>选中的 <strong className="text-foreground">{deleteRequest?.ids.length}</strong> 个文件</>
              )}
              ,此操作不可恢复。
            </p>
            {deleteRequest && deleteRequest.impact.length > 0 && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-amber-700 dark:text-amber-400">
                <p className="font-medium">
                  {deleteRequest.impact.length} 个维基页面引用了这些文件,删除后将自动重新生成:
                </p>
                <ul className="mt-1 list-disc pl-4">
                  {deleteRequest.impact.slice(0, 5).map((p) => (
                    <li key={p.id}>{p.title || p.filename}</li>
                  ))}
                  {deleteRequest.impact.length > 5 && <li>…等 {deleteRequest.impact.length} 个页面</li>}
                </ul>
              </div>
            )}
          </div>
          <DialogFooter>
            <button
              onClick={() => setDeleteRequest(null)}
              disabled={deleting}
              className="rounded-lg border border-input px-4 py-2 text-sm font-medium hover:bg-accent cursor-pointer"
            >
              取消
            </button>
            <button
              onClick={performDelete}
              disabled={deleting}
              className="rounded-lg bg-destructive px-4 py-2 text-sm font-medium text-destructive-foreground hover:opacity-90 disabled:opacity-50 cursor-pointer"
            >
              {deleting ? '删除中…' : '删除'}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <SelectionActionBar
        count={selectedIds.size}
        onDelete={handleDeleteSelected}
        onClear={clearSelection}
      />
    </div>
  )
}
