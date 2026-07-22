'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { useKBStore, useUserStore } from '@/stores'
import {
  Plus, Loader2, LogOut, Moon, Sun, BookOpen, AlertCircle, RefreshCcw,
  EllipsisVertical, Pencil, Trash2,
} from 'lucide-react'
import { toast } from 'sonner'
import type { KnowledgeBase } from '@/lib/types'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu'
import { useTheme } from 'next-themes'
const isLocal = process.env.NEXT_PUBLIC_MODE === 'local'

function wikiHref(slug: string): string {
  return `/wikis/${slug}`
}

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} 小时前`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days} 天前`
  const months = Math.floor(days / 30)
  if (months < 12) return `${months} 个月前`
  return `${Math.floor(months / 12)} 年前`
}

export default function WikisPage() {
  const router = useRouter()
  const knowledgeBases = useKBStore((s) => s.knowledgeBases)
  const loading = useKBStore((s) => s.loading)
  const error = useKBStore((s) => s.error)
  const retryFetchKBs = useKBStore((s) => s.fetchKBs)
  const createKB = useKBStore((s) => s.createKB)
  const user = useUserStore((s) => s.user)
  const [creating, setCreating] = React.useState(false)
  const [dialogOpen, setDialogOpen] = React.useState(false)
  const [name, setName] = React.useState('')
  const [kind, setKind] = React.useState<'wiki' | 'course'>('wiki')
  const [openingSlug, setOpeningSlug] = React.useState<string | null>(null)
  const [, startNavigation] = React.useTransition()

  const openWiki = React.useCallback((slug: string) => {
    setOpeningSlug(slug)
    startNavigation(() => {
      router.push(wikiHref(slug))
    })
  }, [router])

  const handleQuickCreate = async () => {
    setCreating(true)
    try {
      const email = user?.email || 'My'
      const displayName = email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1)
      const kb = await createKB(`${displayName} 的维基`)
      openWiki(kb.slug)
    } catch (err) {
      console.error('Failed to create KB:', err)
    } finally {
      setCreating(false)
    }
  }

  const handleCreate = async () => {
    if (!name.trim()) return
    setCreating(true)
    try {
      const kb = await createKB(name.trim(), undefined, kind)
      setDialogOpen(false)
      setName('')
      setKind('wiki')
      openWiki(kb.slug)
    } catch (err) {
      console.error('Failed to create KB:', err)
    } finally {
      setCreating(false)
    }
  }

  const handleDialogOpenChange = (open: boolean) => {
    setDialogOpen(open)
    if (!open) {
      setName('')
      setKind('wiki')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full flex flex-col">
        <PageHeader onNew={() => setDialogOpen(true)} />
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-sm rounded-xl border border-border bg-card p-5 shadow-sm">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md bg-destructive/10 text-destructive">
                <AlertCircle className="size-4" />
              </div>
              <div className="min-w-0">
                <h1 className="text-sm font-semibold text-foreground">维基加载失败</h1>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  {error}
                </p>
              </div>
            </div>
            <button
              onClick={() => retryFetchKBs()}
              className="mt-4 inline-flex h-9 w-full items-center justify-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              <RefreshCcw className="size-4" />
              重试
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (knowledgeBases.length === 0) {
    return (
      <div className="h-full flex flex-col">
        <PageHeader onNew={() => setDialogOpen(true)} />
        <div className="flex-1 flex flex-col items-center justify-center p-8">
          <div className="w-full max-w-2xl">
            <div className="text-center mb-12">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-foreground mb-6">
                <BookOpen size={24} className="text-background" />
              </div>
              <h1 className="text-3xl font-bold tracking-tight">
                创建您的第一个维基
              </h1>
              <p className="mt-3 text-base text-muted-foreground leading-relaxed max-w-md mx-auto">
                上传资料、连接 AI 助手,让它自动编纂出结构化的维基。
              </p>
            </div>

            <div className="grid sm:grid-cols-3 gap-4 mb-10">
              {[
                {
                  step: '1',
                  title: '创建维基',
                  desc: '为您的知识空间命名,数量不限。',
                },
                {
                  step: '2',
                  title: '添加资料',
                  desc: '上传 PDF、笔记、会议记录 — 任何想让 AI 学习的内容。',
                },
                {
                  step: '3',
                  title: '交给 AI',
                  desc: 'AI 助手阅读您的资料,编纂出带交叉引用与摘要的维基。',
                },
              ].map((item, i) => (
                <motion.div
                  key={item.step}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.08, ease: [0.25, 0.1, 0.25, 1] }}
                  className="rounded-xl border border-border p-5 bg-card"
                >
                  <div className="flex items-center justify-center w-7 h-7 rounded-full bg-foreground text-background text-xs font-bold mb-3">
                    {item.step}
                  </div>
                  <h3 className="text-sm font-semibold mb-1">{item.title}</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">{item.desc}</p>
                </motion.div>
              ))}
            </div>

            <div className="flex flex-col items-center gap-3">
              <button
                onClick={handleQuickCreate}
                disabled={creating}
                className="inline-flex items-center justify-center gap-2.5 rounded-full bg-foreground text-background px-8 py-3 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer disabled:opacity-50"
              >
                {creating ? (
                  <><Loader2 size={15} className="animate-spin" /> 初始化中...</>
                ) : (
                  <><Plus size={15} /> 开始使用</>
                )}
              </button>
              <button
                onClick={() => setDialogOpen(true)}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              >
                或自定义名称创建
              </button>
            </div>
          </div>
        </div>

        <CreateWikiDialog
          open={dialogOpen}
          onOpenChange={handleDialogOpenChange}
          name={name}
          onNameChange={setName}
          kind={kind}
          onKindChange={setKind}
          creating={creating}
          onCreate={handleCreate}
        />
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      <PageHeader onNew={() => setDialogOpen(true)} />

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-8 py-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {knowledgeBases.map((kb, index) => (
              <WikiCard
                key={kb.id}
                kb={kb}
                index={index}
                isOpening={openingSlug === kb.slug}
                onOpen={() => openWiki(kb.slug)}
              />
            ))}

            <button
              onClick={() => setDialogOpen(true)}
              className="flex flex-col items-center justify-center gap-2 p-5 rounded-xl border border-dashed border-border hover:border-primary/50 hover:bg-accent/30 transition-colors cursor-pointer min-h-[112px]"
            >
              <Plus size={16} className="text-muted-foreground" />
              <span className="text-xs text-muted-foreground">新建维基</span>
            </button>
          </div>
        </div>
      </div>

      <CreateWikiDialog
        open={dialogOpen}
        onOpenChange={handleDialogOpenChange}
        name={name}
        onNameChange={setName}
        kind={kind}
        onKindChange={setKind}
        creating={creating}
        onCreate={handleCreate}
      />
    </div>
  )
}

function WikiCard({
  kb,
  index,
  isOpening,
  onOpen,
}: {
  kb: KnowledgeBase
  index: number
  isOpening: boolean
  onOpen: () => void
}) {
  const renameKB = useKBStore((s) => s.renameKB)
  const deleteKB = useKBStore((s) => s.deleteKB)
  const [renameOpen, setRenameOpen] = React.useState(false)
  const [deleteOpen, setDeleteOpen] = React.useState(false)
  const [renameName, setRenameName] = React.useState(kb.name)
  const [busy, setBusy] = React.useState(false)

  const stats: string[] = []
  if (kb.source_count > 0) stats.push(`${kb.source_count} 份资料`)
  if (kb.wiki_page_count > 0) stats.push(`${kb.wiki_page_count} 个页面`)

  const handleRename = async () => {
    const next = renameName.trim()
    if (!next || next === kb.name || busy) return
    setBusy(true)
    try {
      await renameKB(kb.id, next)
      setRenameOpen(false)
    } catch {
      toast.error('重命名维基失败')
    } finally {
      setBusy(false)
    }
  }

  const handleDelete = async () => {
    if (busy) return
    setBusy(true)
    try {
      await deleteKB(kb.id)
      setDeleteOpen(false)
    } catch {
      toast.error('删除维基失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: index * 0.05, ease: [0.25, 0.1, 0.25, 1] }}
        role="button"
        tabIndex={0}
        onClick={onOpen}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onOpen()
          }
        }}
        className="flex flex-col items-start gap-3 p-5 rounded-xl border border-border bg-card hover:bg-accent/50 transition-colors cursor-pointer text-left group overflow-hidden"
      >
        <div className="flex items-center gap-3 min-w-0 w-full">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-muted group-hover:bg-accent transition-colors flex-shrink-0">
            {isOpening ? (
              <Loader2 size={16} className="animate-spin text-muted-foreground" />
            ) : (
              <BookOpen size={16} className="text-muted-foreground group-hover:text-foreground transition-colors" />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-medium text-foreground truncate">{kb.name}</h2>
            {kb.description && (
              <p className="text-xs text-muted-foreground mt-0.5 truncate">{kb.description}</p>
            )}
          </div>
          {/* Menu events must not reach the card's open handler. */}
          <span
            className="shrink-0 -mr-2"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.stopPropagation()}
          >
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  aria-label="维基操作"
                  className="flex items-center justify-center size-7 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-accent opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 transition-opacity cursor-pointer"
                >
                  <EllipsisVertical className="size-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-36">
                <DropdownMenuItem
                  onSelect={() => {
                    setRenameName(kb.name)
                    setRenameOpen(true)
                  }}
                >
                  <Pencil />
                  重命名
                </DropdownMenuItem>
                <DropdownMenuItem variant="destructive" onSelect={() => setDeleteOpen(true)}>
                  <Trash2 />
                  删除
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground/50 w-full">
          {stats.length > 0 ? (
            <span>{stats.join(' · ')}</span>
          ) : (
            <span className="text-muted-foreground/30">暂无资料</span>
          )}
          <span className="ml-auto text-muted-foreground/30 shrink-0">
            {relativeTime(kb.updated_at)}
          </span>
        </div>
      </motion.div>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>重命名维基</DialogTitle>
          </DialogHeader>
          <Input
            value={renameName}
            onChange={(e) => setRenameName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleRename()}
            autoFocus
          />
          <DialogFooter>
            <Button onClick={handleRename} disabled={busy || !renameName.trim() || renameName.trim() === kb.name}>
              {busy ? '重命名中…' : '重命名'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除维基</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            将永久删除 <strong>{kb.name}</strong> 及其全部文档,此操作不可恢复。
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              取消
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={busy}>
              {busy ? '删除中…' : '删除'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function PageHeader({ onNew }: { onNew?: () => void }) {
  return (
    <div className="shrink-0 flex items-center justify-between px-6 h-12 border-b border-border">
      <span className="text-sm font-medium text-foreground tracking-tight">LLM Wiki</span>
      <div className="flex items-center gap-1">
        {onNew && (
          <button
            onClick={onNew}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors cursor-pointer"
          >
            <Plus className="size-3" />
            新建
          </button>
        )}
        <UserMenu />
      </div>
    </div>
  )
}

function UserMenu() {
  const router = useRouter()
  const { theme, setTheme } = useTheme()
  const user = useUserStore((s) => s.user)
  const signOutLocal = useUserStore((s) => s.signOut)
  const [mounted, setMounted] = React.useState(false)
  React.useEffect(() => { setMounted(true) }, [])

  const handleSignOut = async () => {
    if (!isLocal) {
      const { createClient } = await import('@/lib/supabase/client')
      const supabase = createClient()
      await supabase.auth.signOut()
    }
    signOutLocal()
    if (isLocal) return
    router.push('/login')
  }

  if (!user) return null
  const initials = user.email.slice(0, 2).toUpperCase()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="h-6 w-6 bg-muted border border-border rounded-full flex items-center justify-center cursor-pointer hover:bg-accent transition-colors">
          <span className="text-[9px] font-medium text-muted-foreground">{initials}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <div className="px-2 py-1.5 text-xs text-muted-foreground truncate">
          {user.email}
        </div>
        <DropdownMenuSeparator />
        {mounted && (
          <DropdownMenuItem onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? (
              <><Sun className="mr-2 h-4 w-4" />浅色模式</>
            ) : (
              <><Moon className="mr-2 h-4 w-4" />深色模式</>
            )}
          </DropdownMenuItem>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleSignOut}>
          <LogOut className="mr-2 h-4 w-4" />
          退出登录
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function CreateWikiDialog({
  open,
  onOpenChange,
  name,
  onNameChange,
  kind,
  onKindChange,
  creating,
  onCreate,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  name: string
  onNameChange: (name: string) => void
  kind: 'wiki' | 'course'
  onKindChange: (kind: 'wiki' | 'course') => void
  creating: boolean
  onCreate: () => void
}) {
  const isCourse = kind === 'course'
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>新建{isCourse ? '课程' : '维基'}</DialogTitle>
        </DialogHeader>
        <Input
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onCreate()}
          placeholder={isCourse ? '强化学习入门' : '我的研究'}
          autoFocus
        />
        {isCourse && (
          <p className="-mt-1 text-xs leading-relaxed text-muted-foreground">
            课程会将您的材料组织为有序课时,支持进度跟踪与续学,而非自由形式的维基。
          </p>
        )}
        <DialogFooter className="items-center gap-3 sm:justify-between">
          <button
            type="button"
            onClick={() => onKindChange(isCourse ? 'wiki' : 'course')}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            {isCourse ? '改为维基' : '改为课程'}
          </button>
          <Button onClick={onCreate} disabled={creating || !name.trim()}>
            {creating ? '创建中…' : `创建${isCourse ? '课程' : '维基'}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
