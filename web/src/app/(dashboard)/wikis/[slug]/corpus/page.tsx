'use client'

import * as React from 'react'
import { useParams } from 'next/navigation'
import { Loader2 } from 'lucide-react'
import { useKBStore, useUserStore } from '@/stores'
import { CorpusView } from '@/components/corpus/CorpusView'

export default function CorpusPage() {
  const params = useParams<{ slug: string }>()
  const knowledgeBases = useKBStore((s) => s.knowledgeBases)
  const kbLoading = useKBStore((s) => s.loading)
  const user = useUserStore((s) => s.user)

  const kb = React.useMemo(
    () => knowledgeBases.find((k) => k.slug === params.slug),
    [knowledgeBases, params.slug],
  )

  if (kbLoading || !user) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!kb) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 bg-background">
        <h1 className="text-lg font-medium">未找到维基</h1>
        <p className="text-sm text-muted-foreground">
          维基 &ldquo;{params.slug}&rdquo; 不存在,或您没有访问权限。
        </p>
      </div>
    )
  }

  return <CorpusView key={kb.id} kbId={kb.id} kbSlug={kb.slug} kbName={kb.name} />
}
