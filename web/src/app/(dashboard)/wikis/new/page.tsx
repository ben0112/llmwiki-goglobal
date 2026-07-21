'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { useKBStore } from '@/stores'
import { Loader2 } from 'lucide-react'

export default function NewKnowledgeBasePage() {
  const [name, setName] = React.useState('')
  const [description, setDescription] = React.useState('')
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState('')
  const router = useRouter()
  const createKB = useKBStore((s) => s.createKB)

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return

    setLoading(true)
    setError('')

    try {
      const kb = await createKB(trimmed, description.trim() || undefined)
      router.push(`/wikis/${kb.slug}`)
    } catch (err) {
      setError((err as Error).message || '创建维基失败')
      setLoading(false)
    }
  }

  return (
    <div className="max-w-md mx-auto p-8">
      <h1 className="text-xl font-semibold tracking-tight">新建维基</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        维基是由大模型基于您的原始资料编纂并持续维护的知识库。
      </p>

      <form onSubmit={handleCreate} className="mt-6 space-y-4">
        <div>
          <label htmlFor="kb-name" className="block text-sm font-medium mb-1.5">
            名称
          </label>
          <input
            id="kb-name"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background"
            placeholder="我的研究"
            required
          />
        </div>

        <div>
          <label htmlFor="kb-description" className="block text-sm font-medium mb-1.5">
            Description <span className="text-muted-foreground font-normal">(optional)</span>
          </label>
          <textarea
            id="kb-description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background resize-none"
            placeholder="关于……的笔记与文献"
          />
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        <button
          type="submit"
          disabled={loading || !name.trim()}
          className="w-full rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              创建中...
            </>
          ) : (
            '创建'
          )}
        </button>
      </form>
    </div>
  )
}
