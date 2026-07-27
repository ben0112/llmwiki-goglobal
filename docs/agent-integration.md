# 第三方 LLM 智能体接入指南(MCP 配置与认证)

LLM Wiki 通过 **MCP(Model Context Protocol)** 提供交互式读写、检索与治理
工具；Hosted REST 另行提供文档任务、取消和可选的服务端 RAG。任何支持 MCP
的智能体——桌面端助手、OpenAI Codex CLI、Hermes、OpenClaw，以及其他 MCP
客户端——都可以接入。整体边界见[总体架构](architecture/overview.md)。

本指南覆盖:两种部署形态下的连接方式、API 密钥的创建与认证流程、各主流客户端的具体配置、通用配方(适配未列出的客户端)、验证与排错。

---

## 1. 两种形态，一张能力矩阵

| Surface | Local | Hosted | Purpose |
|---|---|---|---|
| MCP tools | yes：stdio | yes：Streamable HTTP | 交互式搜索、读取、写入与治理 |
| REST document/job APIs | no | yes | 上传、持久任务状态与取消 |
| REST server-RAG APIs | no | gated，默认关闭 | 有界的服务端 wiki 构建 |
| `scripts.rag` CLI | no | gated，默认关闭 | 通过 REST 操作服务端 RAG |

本地 MCP 由客户端拉起进程，进程与工作区构成信任边界。Hosted MCP URL 是
`https://mcp.example.com/mcp`，使用 Streamable HTTP；Hosted MCP 与 REST
都接受 `sv_` API 密钥形式的 `Authorization: Bearer` 认证。仅支持 stdio 的
客户端可通过 `mcp-remote` 桥接 Hosted MCP。

> 托管模式同时接受 Supabase 会话 JWT 作为 Bearer(Web 端内部使用),但**给智能体配的应当是 API 密钥**:长期有效、可单独吊销、有使用审计,不随会话过期。

---

## 2. 准备工作

### 2.1 确定 MCP URL

- **本地模式**:无 URL,走 stdio(见 §3.1)。
- **托管模式**:`https://mcp.example.com/mcp`(即部署时的 `MCP_URL`,见 `docs/self-hosting.md`)。Web 端 **设置** 页底部会显示当前部署的 MCP 地址。

健康检查:`curl -fsS https://mcp.example.com/health` 应返回 ok。

### 2.2 创建 API 密钥(仅托管模式)

**方式一:Web 界面(推荐)**

1. 登录 Web 应用 → 右上角 **设置** → **连接 AI 助手 (MCP)**;
2. 填写密钥名称(建议一个客户端一个密钥,如 `my-desktop`、`codex-cli`、`openclaw`),点击 **创建密钥**;
3. 页面会展示一段现成的 MCP 配置 JSON(含密钥)。**密钥仅显示这一次**,立即复制保存:

```json
{
  "mcpServers": {
    "llmwiki": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer sv_xxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**方式二:命令行(无浏览器环境 / 自动化)**

API 密钥接口需要一个已登录用户的 JWT。先用邮箱密码从自部署 Supabase(GoTrue)换取 access token,再调用密钥接口:

```bash
# 1) 换取会话 JWT(ANON_KEY 为部署时生成的 anon key)
TOKEN=$(curl -sS "https://supabase.example.com/auth/v1/token?grant_type=password" \
  -H "apikey: $ANON_KEY" -H "Content-Type: application/json" \
  -d '{"email":"agent-admin@example.com","password":"..."}' | jq -r .access_token)

# 2) 创建 API 密钥(响应中的 key 字段仅返回这一次)
curl -sS -X POST "https://api.example.com/v1/api-keys" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"codex-cli"}' | jq .

# 3) 管理:列出 / 吊销
curl -sS "https://api.example.com/v1/api-keys" -H "Authorization: Bearer $TOKEN" | jq .
curl -sS -X DELETE "https://api.example.com/v1/api-keys/<id>" -H "Authorization: Bearer $TOKEN"
```

**认证过程**(便于排错时理解):客户端在每个 HTTP 请求上带 `Authorization: Bearer sv_…`;MCP 服务端计算密钥的 SHA-256 与 `api_keys` 表比对(库中只存哈希),命中且未吊销即放行,并同步更新 `last_used_at`(设置页可见,用于审计)。同一个密钥也通行 REST API(`https://api.example.com/v1/…`),权限等同于该用户本人。

### 2.3 Hosted 持久任务状态与取消

文档上传或其他 Hosted 写入返回 `job_id` 后，客户端可使用同一 Bearer 凭据
观察和取消任务：

```bash
curl -sS "https://api.example.com/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $LLMWIKI_ACCESS_TOKEN" | jq .
curl -sS -X POST "https://api.example.com/v1/jobs/$JOB_ID/cancel" \
  -H "Authorization: Bearer $LLMWIKI_ACCESS_TOKEN" | jq .
```

任务状态由 Postgres 持久账本拥有；轮询任意 API 副本得到同一结果。取消在
worker 检查点生效，不保证撤回已经原子提交的结果。租约、重试和恢复合同见
[durable-jobs.md](architecture/durable-jobs.md)。

### 2.4 可选服务端 RAG

当管理员在 API 与 worker 上启用 `SERVER_RAG_ENABLED=true` 后，通过认证的
REST 提供：

- `POST /v1/rag/build-wiki`；
- `GET /v1/rag/runs/{run_id}`；
- `GET /v1/rag/runs/{run_id}/steps`；
- `POST /v1/rag/runs/{run_id}/resume`；以及
- 通用的 `POST /v1/jobs/{job_id}/cancel`。

`PYTHONPATH=api python -m scripts.rag` 提供 `build-wiki`、`status`、`steps`
和 `resume`，它只调用 REST。客户端只提交服务器允许的 profile 名称；服务端
**不接受 provider URL、模型 ID 或 provider credential**。
`LLMWIKI_ACCESS_TOKEN` 是
LLMWiki Bearer 凭据，不是模型供应商密钥。预算、dry-run、部分提交和恢复合同
见 [server-rag.md](architecture/server-rag.md)。

---

## 3. 各客户端配置

### 3.1 桌面端 MCP 客户端(mcpServers 配置)

**本地模式(stdio)** — 在仓库目录执行:

```bash
./llmwiki mcp-config ~/goglobal-ws
```

把打印出的 JSON 合并进客户端的 MCP 配置文件(`mcpServers` 段),形如:

```json
{
  "mcpServers": {
    "llmwiki-goglobal-ws": {
      "command": "/path/to/llmwiki-goglobal/llmwiki",
      "args": ["mcp", "/home/you/goglobal-ws"]
    }
  }
}
```

一个工作区一个 server 条目;多工作区就加多条。重启客户端生效。

若本地部署跑在 Docker 里(README 的"Docker 一键运行",容器名 `llmwiki`),容器还直接以 **Streamable HTTP** 暴露 MCP(默认 `http://localhost:8080/mcp`,本机回环、无需认证)——支持 HTTP 传输的客户端首选它;启动日志(`docker compose logs llmwiki`)与 Web 设置页都会给出带真实端口的可粘贴配置:

```json
{"mcpServers": {"llmwiki": {"url": "http://localhost:8080/mcp"}}}
```

仅支持 stdio 的客户端改为经容器执行:

```json
{
  "mcpServers": {
    "llmwiki": {
      "command": "docker",
      "args": ["exec", "-i", "llmwiki", "/app/llmwiki", "mcp", "/workspace"]
    }
  }
}
```

**托管模式** — 仅能拉起本地进程的桌面客户端连远程服务需用 [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) 桥接(stdio ↔ Streamable HTTP,并注入认证头):

```json
{
  "mcpServers": {
    "llmwiki": {
      "command": "npx",
      "args": [
        "-y", "mcp-remote",
        "https://mcp.example.com/mcp",
        "--header", "Authorization: Bearer sv_xxxxxxxxxxxxxxxxxxxxxxxx"
      ]
    }
  }
}
```

需要本机有 Node.js 20+。若不想把密钥写进配置文件,可写成 `"Authorization: Bearer ${LLMWIKI_API_KEY}"` 并在 `env` 字段传入(mcp-remote 支持环境变量展开)。

### 3.2 支持 Streamable HTTP 的 CLI 智能体

**托管模式** — 原生支持 Streamable HTTP + 自定义请求头的 CLI 客户端,直接把 §2.2 生成的 JSON 放进其 MCP 配置(团队共享时建议用环境变量替代明文密钥):

```json
{
  "mcpServers": {
    "llmwiki": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer ${LLMWIKI_API_KEY}" }
    }
  }
}
```

**本地模式** — 与桌面端相同的 stdio 条目,或直接指向 `http://localhost:8080/mcp`(Docker 部署时)。

### 3.3 网页端智能体(自定义连接器)

网页端智能体通过其 **设置 → 连接器(Connectors)→ 添加自定义连接器** 接入远程 MCP,填入 `https://mcp.example.com/mcp`。注意:

- 部分网页端自定义连接器的认证以 **OAuth 为主**;本项目为保证可自部署性**移除了 OAuth 依赖**,采用静态 API 密钥。若你的网页端在添加连接器时不提供"自定义请求头"选项,将无法直连——改用桌面端 / CLI 接入(§3.1 / §3.2),体验一致。
- 你的 MCP 端点是公网可达的:不要为了绕过认证而部署无鉴权的 MCP 服务。

### 3.4 OpenAI Codex CLI

Codex 的 MCP 配置在 `~/.codex/config.toml`。**通用可靠的方式**是经 `mcp-remote` 桥接(stdio 对所有 Codex 版本可用):

```toml
[mcp_servers.llmwiki]
command = "npx"
args = ["-y", "mcp-remote", "https://mcp.example.com/mcp",
        "--header", "Authorization: Bearer sv_xxxxxxxxxxxxxxxxxxxxxxxx"]
```

较新的 Codex 版本原生支持 Streamable HTTP 服务器,可免去桥接(具体键名以你所装版本的官方文档为准):

```toml
[mcp_servers.llmwiki]
url = "https://mcp.example.com/mcp"
http_headers = { "Authorization" = "Bearer sv_xxxxxxxxxxxxxxxxxxxxxxxx" }
```

本地模式同理拉起 stdio 进程:

```toml
[mcp_servers.llmwiki_local]
command = "/path/to/llmwiki-goglobal/llmwiki"
args = ["mcp", "/home/you/goglobal-ws"]
```

配置后运行 `codex`,用 `/mcp` 命令(或让模型调用 `guide` 工具)确认服务器已挂载。

### 3.5 Hermes、OpenClaw 及其他智能体

这类框架迭代快,配置键名各异,但接入 LLM Wiki 只需回答两个问题——**它支持哪种 MCP 传输?去哪里填配置?**

- **支持 Streamable HTTP(远程 MCP)**:在其 MCP 配置节填两项——服务器 URL `https://mcp.example.com/mcp`,自定义请求头 `Authorization: Bearer sv_…`。等价于 §2.2 的 JSON,把键名对应到该框架的字段即可。
- **仅支持 stdio(本地进程 MCP)**:用桥接命令,任何框架通用:

  ```bash
  npx -y mcp-remote https://mcp.example.com/mcp \
      --header "Authorization: Bearer sv_xxxxxxxxxxxxxxxxxxxxxxxx"
  ```

  把它注册为一个 stdio MCP 服务器(`command: npx`,`args: [-y, mcp-remote, <url>, --header, ...]`)。OpenClaw 的 MCP/工具配置、Hermes 的连接器设置均按此模式填写,具体位置以各自文档的 MCP 章节为准。
- **本地模式**:注册 stdio 服务器 `command: /path/to/llmwiki`,`args: [mcp, <工作区路径>]`。

> 给每个智能体单独发一把密钥(§2.2),不要复用。某个智能体失控或密钥泄露时,单独吊销即可,不影响其他客户端。

---

## 4. 验证与排错

### 4.1 不依赖客户端,先用 curl 验证

托管 MCP 是无状态 Streamable HTTP,可直接握手(注意 `Accept` 必须同时含两种类型):

```bash
curl -sS https://mcp.example.com/mcp \
  -H "Authorization: Bearer sv_xxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

返回含 `"serverInfo"` 的 JSON 即认证与服务均正常。把密钥改错重试应得到 401——这能区分"网络/代理问题"与"密钥问题"。

### 4.2 常见故障

| 现象 | 原因与处理 |
|---|---|
| `401 Unauthorized` | 密钥拼写错误、已吊销,或头名写错(必须是 `Authorization: Bearer sv_…`)。在设置页看该密钥的"最近使用"是否更新:从未更新说明请求根本没带对头。 |
| `406 Not Acceptable` | 客户端/curl 未同时接受 `application/json` 与 `text/event-stream`。正规 MCP 客户端不会触发;手工测试补 `Accept` 头。 |
| 连接超时 / 流中断 | 反向代理未放宽读超时。MCP 是流式响应,代理需 `proxy_read_timeout 300s`(见 `docs/self-hosting.md` 的 nginx 示例)。 |
| Desktop/Codex 里服务器"未连接" | 桥接场景先单独跑一遍 §3.5 的 `npx mcp-remote …` 命令看报错;常见是本机无 Node、公司代理拦截、或 URL 少了末尾 `/mcp`。 |
| 本地模式工具报错"workspace not initialized" | 先 `./llmwiki init <工作区>`(或直接 `./llmwiki open` 一次)。 |
| 密钥泄露 | 设置页(或 `DELETE /v1/api-keys/<id>`)立即吊销,重发新钥。库中只存 SHA-256 哈希,吊销即刻生效。 |

---

## 5. 接入后:让智能体正确使用语料库

无论哪个客户端,建议的第一句话都是:**"先调用 `guide` 工具"** ——它向模型说明知识库结构、写作规范(frontmatter、脚注引用)与工具用法。之后的典型指令:

```text
# 分面检索(八维)
用 search 在 goglobal-corpus 里查"数据出境",facets 用 {"domain":"Z1","country":"IDN","timeliness":"M2"}。

# 治理巡检
对 goglobal-corpus 跑 lint,汇报八维完备率、覆盖率账本的空格,以及所有过期未复审条目。

# 复审工作清单(适合夜间例行任务)
search(mode="references", query="due") 取回复审到期清单,逐条核实时效性并更新条目,最后在 wiki/log.md 记一笔。

# 关系层维护
用 relate 给"ODI备案"与"境外落地设立"建 next 边,给印尼数据条目与商务委部门页建 governed_by 边。
```

把最后两条挂到你所用智能体的定时任务上(智能体的 Routines / cron 定时拉起均可),语料库即可自我保鲜:复审到期自动出工单、维基随新语料持续编译。

---

## 6. 安全实践清单

- **一客户端一密钥**,命名可辨识(`codex-cli`、`hermes-nightly`),定期在设置页核对 `last_used_at`,清理不再使用的密钥。
- 密钥即账号:持有者等同于该用户本人(读写其全部知识库)。给自动化智能体建议**单独注册一个语料账号**,不要用管理员账号的密钥。
- 密钥**不进 git、不进镜像**:配置文件用环境变量引用;CI/容器经 secret 注入。
- MCP 端点务必走 TLS(反向代理终止),`/mcp` 之外不要暴露 converter 与数据库(见 `docs/self-hosting.md` 安全注记)。
