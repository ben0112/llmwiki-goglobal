'use client'

import * as React from 'react'
import { Copy, Check, ArrowLeft, KeyRound, Trash2 } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils'
import { apiFetch } from '@/lib/api'
import { buildApiKeyMcpConfig, MCP_URL } from '@/lib/mcp'
import { runtimeMcpUrl } from '@/lib/runtime-env'
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
        <h1 className="text-xl font-semibold tracking-tight">设置</h1>
      </div>

      {/* Usage */}
      {usage && (
        <section>
          <h2 className="text-base font-medium">用量</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            已上传 {usage.document_count} 个文档
          </p>
          <div className="mt-4 space-y-4">
            <div>
              <div className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-muted-foreground">存储</span>
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
                <span className="text-muted-foreground">OCR 页数</span>
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
        <>
          <LocalMcpSection />
          <div className="h-px bg-border my-8" />
          <CorpusPipelineSection />
        </>
      ) : (
        <ApiKeysSection />
      )}
    </div>
  )
}

interface PipelineConfig {
  base_url: string
  model: string
  api_key_masked: string
  concurrency: number
  effective_concurrency: number
  enable_thinking: boolean
  is_local_endpoint: boolean
  auto: { enabled: boolean; interval: number }
}

// 语料分类流水线设置(仅本地模式):LLM 端点前端可配,测试连接,自动分类开关。
// 优先级:此处保存的值 > 部署环境变量 > 内置默认。
function CorpusPipelineSection() {
  const token = useUserStore((s) => s.accessToken)
  const [cfg, setCfg] = React.useState<PipelineConfig | null>(null)
  const [baseUrl, setBaseUrl] = React.useState('')
  const [model, setModel] = React.useState('')
  const [apiKey, setApiKey] = React.useState('')
  const [concurrency, setConcurrency] = React.useState('')
  const [autoInterval, setAutoInterval] = React.useState('30')
  const [saving, setSaving] = React.useState(false)
  const [testing, setTesting] = React.useState(false)
  const [notice, setNotice] = React.useState<{ ok: boolean; text: string } | null>(null)

  const load = React.useCallback(() => {
    if (!token) return
    apiFetch<PipelineConfig>('/v1/corpus/llm-config', token)
      .then((c) => {
        setCfg(c)
        setBaseUrl(c.base_url)
        setModel(c.model)
        setConcurrency(String(c.concurrency || ''))
        setAutoInterval(String(c.auto.interval))
      })
      .catch(() => {})
  }, [token])

  React.useEffect(() => { load() }, [load])

  const save = async (extra: Record<string, unknown> = {}) => {
    if (!token || saving) return
    setSaving(true)
    setNotice(null)
    try {
      const body: Record<string, unknown> = {
        base_url: baseUrl.trim(), model: model.trim(),
        concurrency: concurrency === '' ? 0 : Math.max(0, parseInt(concurrency, 10) || 0),
        auto_interval: Math.max(30, parseInt(autoInterval, 10) || 30),
        ...extra,
      }
      if (apiKey !== '') body.api_key = apiKey
      const c = await apiFetch<PipelineConfig>('/v1/corpus/llm-config', token, {
        method: 'PUT', body: JSON.stringify(body),
      })
      setCfg(c)
      setApiKey('')
      setAutoInterval(String(c.auto.interval))
      setNotice({ ok: true, text: '设置已保存' })
    } catch {
      setNotice({ ok: false, text: '保存失败' })
    } finally {
      setSaving(false)
    }
  }

  const test = async () => {
    if (!token || testing) return
    setTesting(true)
    setNotice(null)
    try {
      const r = await apiFetch<{ ok: boolean; latency_ms: number; detail: string }>(
        '/v1/corpus/llm-config/test', token, { method: 'POST' })
      setNotice({ ok: r.ok, text: r.ok ? `连接正常(${r.latency_ms}ms)· ${r.detail}` : `连接失败:${r.detail}` })
    } catch {
      setNotice({ ok: false, text: '测试请求失败' })
    } finally {
      setTesting(false)
    }
  }

  if (!cfg) return null

  return (
    <section>
      <h2 className="text-base font-medium">语料分类流水线</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        新语料的审核与八维标注所用的 LLM 端点(OpenAI 兼容)。并发数为同时向端点
        发起的请求数,留空或填 0 时按端点自动取默认:本地端点 2,云端 8;上限 256
        (可经环境变量 CORPUS_LLM_MAX_CONCURRENCY 放宽至 1024)。大并发需端点配额
        支撑——遇 429/5xx 会自动指数退避重试,不计入语料失败。每轮处理全部待分类语料。
      </p>

      <div className="mt-4 space-y-3">
        <div className="flex items-center gap-2">
          <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://host.docker.internal:8000/v1"
            className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm font-mono outline-none focus:ring-1 focus:ring-ring" />
          <button onClick={test} disabled={testing}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent transition-colors cursor-pointer disabled:opacity-50 whitespace-nowrap">
            {testing ? '测试中…' : '测试连接'}
          </button>
        </div>
        <div className="flex items-center gap-2">
          <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="模型名称"
            className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring" />
          <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} type="password"
            placeholder={cfg.api_key_masked ? `API Key(现值 ${cfg.api_key_masked},留空沿用)` : 'API Key(可选)'}
            className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring" />
          <input value={concurrency} onChange={(e) => setConcurrency(e.target.value)} inputMode="numeric"
            placeholder={`LLM 并发数(默认 ${cfg.effective_concurrency})`}
            className="w-44 rounded-md border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring" />
        </div>
        <label className="flex items-center gap-2 text-sm select-none cursor-pointer">
          <input type="checkbox" checked={cfg.enable_thinking}
            onChange={(e) => save({ enable_thinking: e.target.checked })} />
          <span>思考模式</span>
          <span className="text-xs text-muted-foreground">
            模型先推理再作答:标注更审慎,但更慢、更耗 token;仅对支持
            enable_thinking 的推理栈(vLLM/MLX 等)生效,其他端点忽略
          </span>
        </label>
        <div className="flex items-center gap-3">
          <button onClick={() => save()} disabled={saving}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent transition-colors cursor-pointer disabled:opacity-50">
            保存设置
          </button>
          <div className="flex items-center gap-1.5 text-sm select-none">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={cfg.auto.enabled}
                onChange={(e) => save({ auto_enabled: e.target.checked })} />
              自动分类,每
            </label>
            <input value={autoInterval} onChange={(e) => setAutoInterval(e.target.value)}
              onBlur={() => {
                const v = Math.max(30, parseInt(autoInterval, 10) || 30)
                setAutoInterval(String(v))
                if (v !== cfg.auto.interval) save({ auto_interval: v })
              }}
              inputMode="numeric"
              className="w-14 rounded-md border border-border bg-background px-2 py-0.5 text-sm text-center outline-none focus:ring-1 focus:ring-ring" />
            <span>s 检查待分类队列,有积压即自动处理(最低 30s)</span>
          </div>
        </div>
        {notice && (
          <p className={`text-xs ${notice.ok ? 'text-emerald-600' : 'text-destructive'}`}>{notice.text}</p>
        )}
      </div>
    </section>
  )
}

// 本地模式的"连接 AI 助手":Docker 部署时入口脚本会经 /__llmwiki_env.js 注入
// MCP 的 HTTP 地址(runtimeMcpUrl),直接展示可粘贴配置;源码运行时回退到
// CLI 命令(stdio)。用 effect 读取以避免 SSR/客户端首帧不一致。
function LocalMcpSection() {
  const [mcpUrl, setMcpUrl] = React.useState<string | null>(null)
  React.useEffect(() => {
    setMcpUrl(runtimeMcpUrl())
  }, [])

  if (mcpUrl) {
    const config = JSON.stringify({ mcpServers: { llmwiki: { url: mcpUrl } } }, null, 2)
    return (
      <section>
        <h2 className="text-base font-medium">连接 AI 助手 (MCP)</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          本实例的 MCP 服务已通过 HTTP 暴露,任何支持 Streamable HTTP 的 MCP
          客户端都可直接连接,无需认证:
        </p>
        <div className="relative mt-4">
          <pre className="rounded-lg bg-muted border border-border p-4 text-xs font-mono overflow-x-auto text-foreground">
            {config}
          </pre>
          <CopyButton text={config} />
        </div>

      </section>
    )
  }

  return (
    <section>
      <h2 className="text-base font-medium">连接 AI 助手 (MCP)</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        运行以下命令获取此工作区的 MCP 客户端配置:
      </p>
      <pre className="mt-4 rounded-lg bg-muted border border-border p-4 text-sm font-mono overflow-x-auto text-foreground">
        llmwiki mcp-config &lt;workspace-path&gt;
      </pre>
    </section>
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
      {copied ? <><Check size={12} />已复制</> : <><Copy size={12} />复制</>}
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
        body: JSON.stringify({ name: name.trim() || '默认' }),
      })
      setNewKey(res.key)
      setName('')
      refresh()
    } catch {
      setError('创建 API 密钥失败')
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
      setError('吊销 API 密钥失败')
    }
  }

  return (
    <section>
      <h2 className="text-base font-medium">连接 AI 助手 (MCP)</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        创建 API 密钥,并将下方配置添加到您的 MCP 客户端(桌面端或网页端的
        自定义连接器均可)。该密钥以您的身份进行认证——请像密码一样妥善
        保管,不再需要时及时吊销。
      </p>

      <div className="mt-4 flex items-center gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="密钥名称(如 my-desktop)"
          className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring"
        />
        <button
          onClick={createKey}
          disabled={creating}
          className="flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent transition-colors cursor-pointer disabled:opacity-50"
        >
          <KeyRound size={14} />
          创建密钥
        </button>
      </div>
      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      {newKey && (
        <div className="mt-4">
          <p className="text-xs text-amber-600 dark:text-amber-500">
            此密钥仅显示一次——请立即复制配置并妥善保存。
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
              <span className="font-medium">{k.name || '默认'}</span>
              <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">{k.key_prefix}…</code>
              <span className="flex-1 text-xs text-muted-foreground">
                创建于 {k.created_at.slice(0, 10)}
                {k.last_used_at ? ` · 最近使用 ${k.last_used_at.slice(0, 10)}` : ' · 从未使用'}
              </span>
              <button
                onClick={() => revokeKey(k.id)}
                title="吊销"
                className="p-1 rounded-md text-muted-foreground hover:text-destructive hover:bg-accent transition-colors cursor-pointer"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      <p className="mt-3 text-xs text-muted-foreground">
        MCP 地址:{' '}
        <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">{MCP_URL}</code>
      </p>
    </section>
  )
}
