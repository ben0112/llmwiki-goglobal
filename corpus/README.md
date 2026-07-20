# 出海智能体语料分类体系 · LLM Wiki 集成 (Phase 1)

Implements the eight-dimension corpus classification layer (spec v2026.06) on top of LLM Wiki: versioned code tables, entry metadata parsing/validation, and an importer for the LLM 标注工具包 output.

本目录把《出海智能体语料分类体系(v2026.06)》的**八维"身份证"**落到 LLM Wiki 上,是分期实施的**第一期**:

| 模块 | 作用 |
|---|---|
| `codetables/v2026_06.json` | 受控码表(阶段/服务大类/体裁/隐性规则/证据强度/归口部门/来源域/国别 ISO 3166/行业 GB\/T 4754/形态/时效/生命周期/业务视图 7类27场景),独立版本号,**只增版本、不改历史** |
| `codetable.py` | 码表加载与取值归一化(别名、U 码、ISO 转换、置信度、复审周期) |
| `schema.py` | 八维条目模型 `EntryRecord`:解析 `标注明细.csv` 两种口径(流水线扁平列 / 试标注样本装饰值,如 `S2④(副S3⑤)`、`E2/商务委U3`),主/副标签拆分、entry_id 校验与重派、review_due 推算;**公理一**——任何行都能落位,码表外取值走兜底并记录问题 |
| `import_annotations.py` | 导入 CLI:标注明细 → 工作区 markdown 条目 + SQLite 索引行 + 导入报告/复核队列 |

## 用法

上游流程不变(先审后标,见工具包 README):

```
原始语料 →〔审核 收/不收〕→ 收录/ →〔classify 八维标注〕→ 标注明细.csv
                                        →〔derive 业务派生〕→ 标注明细_业务视图.csv
                                                                    │
                                                          本导入器(新增一步)
                                                                    ↓
                                                          LLM Wiki 工作区
```

```bash
# 1) 初始化工作区(一次)
./llmwiki init ~/goglobal-ws

# 2) 校验(不写入,报告出在 CSV 同目录 corpus_import_dryrun/)
python3 -m corpus.import_annotations \
    --csv 标注结果/标注明细_业务视图.csv \
    --workspace ~/goglobal-ws --dry-run

# 3) 导入(--raw 提供收录语料目录时,正文一并入条目)
python3 -m corpus.import_annotations \
    --csv 标注结果/标注明细_业务视图.csv \
    --workspace ~/goglobal-ws \
    --raw 审核结果_deepseek/收录

# 4) 打开工作区(reconcile 会自动为新条目建全文检索分块)
./llmwiki open ~/goglobal-ws
```

`标注明细.csv`(未派生业务视图)同样可导;业务四列缺省即空。

## 导入后的形态

- **文件即真相源**:每条语料一个 markdown 文件,按货架落位
  `corpus/<主阶段>-<主大类>/<entry_id>.md`(如 `corpus/S2-G1/S2-G1-政策-GEN-3F2A1.md`),
  YAML frontmatter 携带完整八维 + 业务视图元数据,可直接被 MCP `read`/`search`(tags)消费。
- **SQLite 索引**:`documents.metadata` 存结构化八维记录(第二期分面检索读这里),
  `tags` 存分面标签(现有按 tag 过滤立即可用),`parser=NULL` 交给应用 reconcile 自动分块。
- **报告**(存 `.llmwiki/corpus_import/`,不入索引):
  - `导入报告.md` — 覆盖率账本(主阶段×大类层)、空格清单(补采罗盘)、校验明细;
  - `复核队列.csv` — X9 / 低置信 / 校验错误条目,供部门人工校准(法律 C1、数据出境 R1 类按规范全量复核)。
- **幂等**:entry_id 流水号 = relpath 稳定短哈希(与 classify_pipeline 一致),
  重跑同一 CSV 不产生重复;内容变化才 bump version 并重建分块。

## 码表迭代

按规范 §5.2:新版本 = 新增 `codetables/vYYYY.MM.json`,旧版本不动;
导入时 `--codetable vYYYY.MM` 指定。X9 积累到阈值 → 码表评审 → 升版 → 按"理由"字段批量重标。

## 分期路线

- **第一期(本目录)**:码表 + 八维校验 + 导入。✅
- 第二期:分面检索(VaultFS/API/MCP `search` 增加 metadata 过滤;本地 FTS5 换 CJK 分词)、`lint` 八维完备率检查、覆盖率账本 API。
- 第三期:Web 分面筛选/货架矩阵/业务视图导航/生命周期标记。
- 第四期:关系层五类边(上下位/前后置/路径衔接/归口映射/阶段服务包)、review_due 到期驱动与 KPI 仪表盘。
