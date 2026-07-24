# 共享内核与数据不变量

本文记录共享内核里程碑完成后的实际边界。它覆盖当前 API、MCP、SQLite、
Postgres 和本地文件系统实现，不定义后续持久任务、混合检索或服务端 RAG
接口。

## 依赖方向

允许的依赖方向是：API 路由和 MCP 工具依赖各自的应用服务与存储适配器；
应用服务和存储适配器可以依赖 `llmwiki_core`；`llmwiki_core` 不得反向依赖
`api`、`mcp` 或基础设施包。

`llmwiki_core` 是无框架、无数据库的纯 Python 包。它不能导入 FastAPI、MCP、
`asyncpg`、`aiosqlite`、S3 客户端或服务配置。数据库事务、SQL、对象存储、
本地文件 I/O、认证和进程生命周期仍分别属于 `api/` 与 `mcp/`。这一边界由
`tests/unit/core/test_import_boundaries.py` 和所有 Python 镜像的安装测试约束。

## 内核职责

- `llmwiki_core.documents`：文档种类、状态迁移、三元身份范围
  `(user_id, knowledge_base_id, document_id)` 和逻辑路径规范化。
- `llmwiki_core.chunking`：API 与 MCP 共用的分块参数、分块结果和纯文本/分页
  分块算法。
- `llmwiki_core.facets`：facet 名称校验，以及引用语料元数据到 wiki 页面的
  rollup 规则。
- `llmwiki_core.references`：引用、wiki 链接解析，查找表构建和内容派生边的
  去重。
- `llmwiki_core.search`：与后端无关的查询和命中值对象；命中身份为
  `(document_id, document_version, chunk_index)`。
- `llmwiki_core.wiki`：一次 wiki 修订所需的 `WikiWriteBundle`、引用边和
  `VersionConflict` 乐观并发契约。

内核只表达可在各运行时一致执行的值和规则，不负责持久化策略。

## 生命周期与归档语义

普通处理路径只允许以下状态迁移：

```text
pending -> processing -> ready
pending -> failed
processing -> failed
failed -> pending
```

其中 `pending -> failed` 和 `processing -> failed` 都合法；`failed -> pending`
用于显式重试。`ready` 是普通业务路径的终态，只有系统检测到派生数据缺失或
版本漂移时，才可以用 `for_repair=True` 将它重置为 `pending`。状态规则由
`llmwiki_core.documents.assert_status_transition` 定义。

归档不是上述处理状态之一。Hosted Postgres 使用 `documents.archived` 软删除，
所有活动文档查询和唯一性判断都必须排除归档行。本地 SQLite 以工作区文件为
事实来源，不保留归档历史；“归档”会删除文档及级联/显式删除的派生行。两种
适配器可以采用不同保存策略，但对调用者都表现为文档不再出现在活动集合中。

## `ready` 与 `document_version` 不变量

`documents.version` 表示已发布的文档修订。每个 `document_pages` 和
`document_chunks` 行都携带 `document_version`，且可检索命中必须保留该版本。

文档变为 `ready` 时必须满足：

1. 本次发布产生的页面和分块全部使用将要写入 `documents.version` 的同一版本；
2. 旧页面和旧分块已经被替换，不能把不同修订的派生行混合发布；
3. 对达到分块阈值的非 asset 文本，至少存在一个当前版本分块；
4. 文档内容、页面、分块、版本和 `ready` 状态必须在同一数据库事务中可见。

短于分块下限的文本可以合法地没有分块，asset 文档也不受文本分块要求约束。
`api/infra/db/derived_documents.py` 会扫描已就绪但版本不一致或缺失必要分块的
Hosted 文档，并在恢复处理开始前原子重置为 `pending`。

## 本地文件系统修复

本地源文件由工作区文件系统提供原始内容，SQLite 保存索引和派生状态。
`api/domain/local_processor.py` 在启动对账时处理四类中断：派生版本漂移、卡在
`processing`、丢失的 `pending` 任务，以及未超过重试上限的 `failed` 文档。
发现 `ready` 文档的页面或分块版本不一致时，先重置为 `pending`，再从当前源
文件重新提取；反复在处理中崩溃的文档会进入失败隔离，避免每次启动重复拖垮
进程。

本地 wiki 写盘使用同目录临时文件、`fsync` 和 `os.replace`，所以单个文件替换
是原子的，但文件系统与 SQLite 无法组成同一个事务。如果进程在两者之间中断，
`mcp.vaultfs.sqlite.SqliteVaultFS._reconcile_wiki_files` 以已经成功替换的磁盘内容
为准，通过新的 `WikiWriteBundle` 重建内容、版本、分块、内容派生引用和 facet
rollup，使索引最终收敛。

## Hosted 事务边界

Hosted 文档提取由 `api/infra/db/derived_documents.replace_derived_content` 在一个
Postgres 事务中锁定文档，替换页面、分块和派生 asset，递增版本，最后发布
内容与 `ready` 状态。任何一步失败都会回滚整个派生集合。

Hosted wiki 写入由 `mcp.vaultfs.postgres.PostgresVaultFS.write_wiki_bundle` 在一个
事务中完成文档创建或 compare-and-swap 更新、分块替换、内容派生引用替换、
反向页面 stale 传播和 facet rollup。创建路径使用事务级 advisory lock 串行化
同一逻辑路径；更新路径要求 `expected_version` 命中，否则抛出
`VersionConflict`。本地 SQLite 适配器在 `BEGIN IMMEDIATE` 与进程内写锁下提供
同一 bundle 语义。

对象字节写入不属于上述 Postgres 事务；数据库事务只发布已经准备好的派生
数据和引用关系。

## 兼容门面退出条件

`api/services/chunker.py`、`mcp/services/chunker.py`、
`api/services/references.py`、`mcp/tools/references.py` 和
`mcp/vaultfs/facet_rollup.py` 暂时保留旧导入路径，但纯定义只能存在于
`llmwiki_core`。

只有同时满足以下条件才可删除某个兼容门面：仓库内生产代码与测试均已直接
导入 `llmwiki_core`；所有已发布入口和镜像都安装共享包；没有仍受支持的外部
插件或调用方依赖旧路径；删除经过一次明确的弃用周期并有回归测试覆盖。删除
门面不得把数据库或框架依赖迁入共享内核。
