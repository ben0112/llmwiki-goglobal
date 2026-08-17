"""CJK 二元组切词 —— 二字中文搜索伴生索引(chunks_fts_bi)共用工具。

trigram 分词器覆盖 ≥3 字查询;二字业务词(税务/备案)靠本模块:
正文的每段连续 CJK 以相邻二元组展开、空格连接,unicode61 按空格分词
后即可精确 MATCH。查询侧把 ≥2 字的纯 CJK token 转成 bigram 短语
(相邻位置约束保证等价于子串匹配)。

与 api/services/cjk_bigram.py 保持逐字一致(两包独立部署,不共享导入)。
"""


def _is_cjk(ch: str) -> bool:
    return "㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿"


def bigramize(text: str) -> str:
    """正文 → bigram 流(仅 CJK;其余字符只作分段,不入本索引)。

    "税务总局ABC通知" → "税务 务总 总局 通知"(单字段落原样保留)。
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) == 1:
            out.append(run[0])
        else:
            out.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
        run.clear()

    for ch in text:
        if _is_cjk(ch):
            run.append(ch)
        elif run:
            flush()
    if run:
        flush()
    return " ".join(out)


def build_bi_match(query: str) -> str | None:
    """查询 → chunks_fts_bi 的 MATCH 表达式;无法表达时返回 None(回落 LIKE)。

    每个 token 须为 ≥2 字纯 CJK:展开为相邻 bigram 短语("税务总局" →
    "税务 务总 总局" 短语),token 间 AND。含单字/非 CJK token 的查询
    交还 LIKE 路径。
    """
    tokens = [t for t in query.split() if t]
    if not tokens:
        return None
    parts = []
    for t in tokens:
        if len(t) < 2 or not all(_is_cjk(c) for c in t):
            return None
        grams = [t[i] + t[i + 1] for i in range(len(t) - 1)]
        parts.append('"' + " ".join(grams) + '"')
    return " AND ".join(parts)
