'use client'

import * as React from 'react'
import { Copy, Check, ArrowLeft, KeyRound, Trash2 } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils'
import { apiFetch } from '@/lib/api'
import { buildApiKeyMcpConfig, MCP_URL } from '@/lib/mcp'
import { useUserStore } from '@/stores'

interface Usage {
  total_pages: number
  total_storage_bytes: number
  document_count: number
  max_pages: number
  max_storage_bytes: number
}

interface ApiKey {
  id: string
  name: string | null
  key_prefix: string
  created_at: string
  last_used_at: string | null
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / Math.pow(1024, i)
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[i]}`
}

export default function SettingsPage() {
  const router = useRouter()
  const token = useUserStore((s) => s.accessToken)
  const [usage, setUsage] = React.useState<Usage | null>(null)
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    if (!token) return
    apiFetch<Usage>('/v1/usage', token)
      .then((u) => setUsage(u))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [token])

  return (
    <div className="max-w-2xl mx-auto p-8">
      <div className="flex items-center gap-3 mb-8">
        <button
          onClick={() => router.back()}
          className="p-1 rounded-md hover:bg-accent transition-colors cursor-pointer text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
        </button>
        <h1 className="text-xl font-semibold tracking-tight">Settings</h1>
      </div>

      {/* Usage */}
      {usage && (
        <section>
          <h2 className="text-base font-medium">Usage</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {usage.document_count} document{usage.document_count !== 1 ? 's' : ''} uploaded
          </p>
          <div className="mt-4 space-y-4">
            <div>
              <div className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-muted-foreground">Storage</span>
                <span className="font-mono text-xs">
                  {formatBytes(usage.total_storage_bytes)} / {formatBytes(usage.max_storage_bytes)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all',
                    usage.total_storage_bytes / usage.max_storage_bytes > 0.9
                      ? 'bg-destructive'
                      : usage.total_storage_bytes / usage.max_storage_bytes > 0.7
                        ? 'bg-yellow-500'
                        : 'bg-primary'
                  )}
                  style={{ width: `${Math.min(100, (usage.total_storage_bytes / usage.max_storage_bytes) * 100)}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-muted-foreground">OCR Pages</span>
                <span className="font-mono text-xs">
                  {usage.total_pages.toLocaleString()} / {usage.max_pages.toLocaleString()}
                </span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all',
                    usage.total_pages / usage.max_pages > 0.9
                      ? 'bg-destructive'
                      : usage.total_pages / usage.max_pages > 0.7
                        ? 'bg-yellow-500'
                        : 'bg-primary'
                  )}
                  style={{ width: `${Math.min(100, (usage.total_pages / usage.max_pages) * 100)}%` }}
                />
              </div>
            </div>
          </div>
        </section>
      )}

      {usage && <div className="h-px bg-border my-8" />}

      {/* MCP Config */}
      {process.env.NEXT_PUBLIC_MODE === 'local' ? (
        <section>
          <h2 className="text-base font-medium">Connect Claude</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Run this command to get the Claude Desktop / Claude Code MCP config for this workspace:
          </p>
          <pre className="mt-4 rounded-lg bg-muted border border-border p-4 text-sm font-mono overflow-x-auto text-foreground">
            llmwiki mcp-config &lt;workspace-path&gt;
          </pre>
        </section>
      ) : (
        <ApiKeysSection />
      )}
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false)
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setCopied(true)
          setTimeout(() => setCopied(false), 2000)
        } catch {
          console.error('Failed to copy')
        }
      }}
      className={cn(
        'absolute top-3 right-3 flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs transition-colors cursor-pointer',
        copied
          ? 'bg-green-500/10 text-green-600 dark:text-green-400'
          : 'bg-background border border-border text-muted-foreground hover:text-foreground hover:bg-accent'
      )}
    >
      {copied ? <><Check size={12} />Copied</> : <><Copy size={12} />Copy</>}
    </button>
  )
}

// API keys are the MCP/API credential: they work as Bearer tokens anywhere a
// session JWT does, and need no OAuth-capable auth server — which keeps
// self-hosted deployments working (see docs/self-hosting.md).
function ApiKeysSection() {
  const token = useUserStore((s) => s.accessToken)
  const [keys, setKeys] = React.useState<ApiKey[]>([])
  const [name, setName] = React.useState('')
  const [creating, setCreating] = React.useState(false)
  const [newKey, setNewKey] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(() => {
    if (!token) return
    apiFetch<ApiKey[]>('/v1/api-keys', token)
      .then(setKeys)
      .catch(() => {})
  }, [token])

  React.useEffect(() => {
    refresh()
  }, [refresh])

  const createKey = async () => {
    if (!token || creating) return
    setCreating(true)
    setError(null)
    try {
      const res = await apiFetch<ApiKey & { key: string }>('/v1/api-keys', token, {
        method: 'POST',
        body: JSON.stringify({ name: name.trim() || 'Default' }),
      })
      setNewKey(res.key)
      setName('')
      refresh()
    } catch {
      setError('Failed to create API key')
    } finally {
      setCreating(false)
    }
  }

  const revokeKey = async (id: string) => {
    if (!token) return
    try {
      await apiFetch(`/v1/api-keys/${id}`, token, { method: 'DELETE' })
      refresh()
    } catch {
      setError('Failed to revoke API key')
    }
  }

  return (
    <section>
      <h2 className="text-base font-medium">Connect Claude (MCP)</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        Create an API key and add the configuration below to your MCP client
        (Claude Desktop, Claude Code, or claude.ai custom connectors). The key
        authenticates as you — treat it like a password and revoke it when no
        longer needed.
      </p>

      <div className="mt-4 flex items-center gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Key name (e.g. claude-desktop)"
          className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring"
        />
        <button
          onClick={createKey}
          disabled={creating}
          className="flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent transition-colors cursor-pointer disabled:opacity-50"
        >
          <KeyRound size={14} />
          Create key
        </button>
      </div>
      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      {newKey && (
        <div className="mt-4">
          <p className="text-xs text-amber-600 dark:text-amber-500">
            This key is shown once — copy the config now and store it safely.
          </p>
          <div className="relative mt-2">
            <pre className="rounded-lg bg-muted border border-border p-4 text-xs font-mono overflow-x-auto text-foreground">
              {buildApiKeyMcpConfig(newKey)}
            </pre>
            <CopyButton text={buildApiKeyMcpConfig(newKey)} />
          </div>
        </div>
      )}

      {keys.length > 0 && (
        <div className="mt-5 divide-y divide-border rounded-lg border border-border">
          {keys.map((k) => (
            <div key={k.id} className="flex items-center gap-3 px-3 py-2 text-sm">
              <span className="font-medium">{k.name || 'Default'}</span>
              <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">{k.key_prefix}…</code>
              <span className="flex-1 text-xs text-muted-foreground">
                created {k.created_at.slice(0, 10)}
                {k.last_used_at ? ` · last used ${k.last_used_at.slice(0, 10)}` : ' · never used'}
              </span>
              <button
                onClick={() => revokeKey(k.id)}
                title="Revoke"
                className="p-1 rounded-md text-muted-foreground hover:text-destructive hover:bg-accent transition-colors cursor-pointer"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      <p className="mt-3 text-xs text-muted-foreground">
        MCP URL:{' '}
        <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">{MCP_URL}</code>
      </p>
    </section>
  )
}
