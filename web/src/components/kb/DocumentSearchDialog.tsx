'use client'

import * as React from 'react'
import { FileText, Folder, NotepadText, Upload } from 'lucide-react'

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command'
import { useReadPage } from '@/hooks/useReadPage'
import type { ReadDocument, WikiNode } from '@/lib/types'

interface DocumentSearchDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  kbId: string
  token: string | null
  wikiTree: WikiNode[]
  onWikiNavigate: (path: string, docNumber?: number | null) => void
  onOpenSourceDoc: (document: ReadDocument) => void
  onFilesToggle: () => void
  onUpload: () => void
}

function flattenWiki(nodes: WikiNode[]): WikiNode[] {
  const result: WikiNode[] = []
  for (const node of nodes) {
    if (node.path) result.push(node)
    if (node.children) result.push(...flattenWiki(node.children))
  }
  return result
}

export function DocumentSearchDialog({
  open,
  onOpenChange,
  kbId,
  token,
  wikiTree,
  onWikiNavigate,
  onOpenSourceDoc,
  onFilesToggle,
  onUpload,
}: DocumentSearchDialogProps) {
  const [input, setInput] = React.useState('')
  const [query, setQuery] = React.useState('')
  React.useEffect(() => {
    const timer = setTimeout(() => setQuery(input.trim()), 250)
    return () => clearTimeout(timer)
  }, [input])
  React.useEffect(() => {
    if (!open) {
      setInput('')
      setQuery('')
    }
  }, [open])
  const url = query
    ? `/v1/knowledge-bases/${kbId}/documents/browse?query=${encodeURIComponent(query)}&sort=name&direction=asc`
    : ''
  const source = useReadPage<ReadDocument>({
    url,
    token,
    queryKey: `${kbId}|${query}`,
    initialLimit: 100,
  })
  const wikiItems = React.useMemo(() => flattenWiki(wikiTree), [wikiTree])
  const close = () => onOpenChange(false)

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="跳转到页面、源文件或操作..."
        aria-label="搜索页面与源文件"
        value={input}
        onValueChange={setInput}
      />
      <CommandList>
        <CommandEmpty>{query && source.loading ? '搜索中…' : '未找到结果。'}</CommandEmpty>
        {wikiItems.length > 0 && (
          <CommandGroup heading="维基">
            {wikiItems.map((item) => (
              <CommandItem
                key={`wiki-${item.path}`}
                value={`${item.title} ${item.path ?? ''}`}
                onSelect={() => {
                  close()
                  if (item.path) onWikiNavigate(item.path, item.docNumber)
                }}
              >
                <FileText className="mr-2 size-3.5 opacity-50" />
                <span className="truncate">{item.title}</span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
        {query && source.items.length > 0 && (
          <CommandGroup heading="源文件">
            {source.items.slice(0, 100).map((document) => (
              <CommandItem
                key={`source-${document.id}`}
                value={`${document.title ?? ''} ${document.filename} ${document.path}`}
                onSelect={() => {
                  close()
                  onOpenSourceDoc(document)
                }}
              >
                <NotepadText className="mr-2 size-3.5 opacity-50" />
                <span className="truncate">{document.title || document.filename}</span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
        <CommandSeparator />
        <CommandGroup heading="操作">
          <CommandItem onSelect={() => { close(); onFilesToggle() }}>
            <Folder className="mr-2 size-3.5 opacity-50" />
            浏览文件
          </CommandItem>
          <CommandItem onSelect={() => { close(); onUpload() }}>
            <Upload className="mr-2 size-3.5 opacity-50" />
            上传文件
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
