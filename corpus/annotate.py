"""审核 + 八维标注(标注工具包 classify_pipeline.py 的库化移植)。

口径的真相源在 corpus/prompts/*.txt —— 调判定规则改提示词,不改本模块。
输出为「标注明细.csv 行」形状的中文键字典,直接喂 schema.parse_row。
mock 模式移植自工具包的规则桩,供无端点自测与单元测试使用。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .llm import LLMConfig, chat_json

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

VALID_STAGE = {"S0", "S1", "S2", "S3", "S4"}
VALID_DOMAIN = ({f"G{i}" for i in range(1, 5)} | {f"C{i}" for i in range(1, 6)}
                | {f"O{i}" for i in range(1, 7)} | {f"Z{i}" for i in range(1, 5)} | {"X9"})
GENRE_SHORT = {"政策法规": "政策", "办事指南": "办事", "国别研究": "国别", "案例经验": "案例",
               "风险预警": "预警", "实操指引": "指引", "知识问答": "问答", "数据指标": "数据",
               "资源名录": "名录", "规则提炼": "规则", "其他": "其他"}
COUNTRY_CODE = {"印尼": "IDN", "马来西亚": "MYS", "泰国": "THA", "越南": "VNM", "新加坡": "SGP",
                "沙特": "SAU", "阿联酋": "UAE", "巴西": "BRA", "秘鲁": "PER", "墨西哥": "MEX",
                "美国": "USA", "德国": "DEU", "法国": "FRA", "英国": "GBR", "哥伦比亚": "COL",
                "孟加拉": "BGD", "俄罗斯": "RUS", "日本": "JPN", "韩国": "KOR", "印度": "IND",
                "欧盟": "EU"}
REGION_CODE = {"东盟": "ASN", "欧盟": "EU", "中东": "MEA", "拉美": "LAT", "非洲": "AFR",
               "北美": "NAM", "中亚": "CAS", "RCEP": "RCEP", "一带一路": "BNR",
               "多边": "GLB", "全球": "GLB"}


_TITLE_HDR = re.compile(r"^(?:标题|Title)[：:]\s*(.+)$", re.I | re.M)


def extract_title(content: str) -> str:
    """从正文头部的 `标题:` 行提取标题(与工具包 parse_doc 对齐);无则空串。"""
    m = _TITLE_HDR.search(content[:500])
    return m.group(1).strip() if m else ""


def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


# ---------------- 文档 → 提示词 ----------------

def effective_body(body: str) -> str:
    return "\n".join(ln.strip() for ln in body.split("\n") if len(ln.strip()) >= 10)


def build_classify_prompt(title: str, body: str, max_chars: int = 6000) -> str:
    body = effective_body(body)
    if len(body) > max_chars:
        body = body[: int(max_chars * 0.8)] + "\n……[截断]……\n" + body[-int(max_chars * 0.2):]
    return (f"请给以下【已收录】语料打八维标签。\n标题：{title}"
            f"\n正文：\n{body}\n\n只输出一个扁平JSON。")


def build_audit_prompt(title: str, body: str, max_chars: int = 4000) -> str:
    body = effective_body(body)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n……[截断]……"
    return f"候选语料如下。\n标题：{title}\n正文:\n{body}\n\n只输出一个扁平JSON。"


# ---------------- entry_id(与工具包/schema 同源算法) ----------------

def _first(s, valid=None, default=""):
    head = re.split(r"[（(]", str(s))[0].strip()
    if valid:
        m = re.search(r"[A-Z]\d|X9", head)
        if m and (m.group(0) in valid):
            return m.group(0)
        for tok in re.split(r"[·,，、/ ]", head):
            mt = re.search(r"[A-Z]\d|X9", tok)
            if mt and mt.group(0) in valid:
                return mt.group(0)
        return default
    return head or default


def geo_code(geo) -> str:
    g = str(geo)
    if "通用" in g or not g.strip():
        return "GEN"
    for k, v in COUNTRY_CODE.items():
        if k in g:
            return v
    for k, v in REGION_CODE.items():
        if k in g:
            return v
    return "OTH"


def make_entry_id(rec: dict, relpath: str) -> str:
    st = _first(rec.get("阶段", ""), VALID_STAGE, "S0")
    dm = _first(rec.get("服务大类", ""), VALID_DOMAIN, "X9")
    gz = GENRE_SHORT.get(str(rec.get("体裁", "")).strip(), "其他")
    gc = geo_code(rec.get("国别区域", ""))
    serial = hashlib.md5(relpath.encode("utf-8")).hexdigest()[:5].upper()
    return f"{st}-{dm}-{gz}-{gc}-{serial}"


# ---------------- LLM 两级判断 ----------------

async def audit(config: LLMConfig, title: str, body: str) -> tuple[bool, str]:
    """准入审核:返回 (是否收录, 理由)。mock 模式按正文长度规则桩。"""
    if config.is_mock:
        substantial = len(effective_body(body)) >= 200
        return substantial, "[mock]正文长度规则桩"
    obj = await chat_json(config, load_prompt("audit_system_prompt.txt"),
                          build_audit_prompt(title, body), required_keys=("收录",))
    include = str(obj.get("收录", "")).strip().startswith("是")
    return include, str(obj.get("理由", "")).strip()


async def classify(config: LLMConfig, title: str, body: str, relpath: str) -> dict:
    """八维标注:返回「标注明细」行形状的中文键字典(含 entry_id)。"""
    if config.is_mock:
        rec = _mock_classify(title, body)
    else:
        rec = await chat_json(config, load_prompt("classify_system_prompt.txt"),
                              build_classify_prompt(title, body),
                              required_keys=("服务大类", "阶段"))
    rec = dict(rec)
    rec["relpath"] = relpath
    rec["title"] = title
    rec["entry_id"] = make_entry_id(rec, relpath)
    return rec


def _mock_classify(title: str, body: str) -> dict:
    """规则桩(移植自工具包 --mock),测试与无端点自测用。"""
    t = f"{title} {body[:200]}"

    def has(*ks):
        return any(k in t for k in ks)

    if has("反倾销", "反补贴", "保障措施"):
        dm, gen, rule, ev, dep, tm, st = "G3", "风险预警", "R5", "E2", "贸促会", "M1", "S4"
    elif has("制裁", "出口管制", "管制清单", "两用物项"):
        dm, gen, rule, ev, dep, tm, st = "G1", "政策法规", "R1", "E0", "商务委(海关)", "M1", "S2"
    elif has("认证", "标准", "检测", "标签"):
        dm, gen, rule, ev, dep, tm, st = "C5", "国别研究", "R2", "E2", "市监", "M2", "S3"
    elif has("数据", "隐私", "个人信息"):
        dm, gen, rule, ev, dep, tm, st = "Z1", "国别研究", "R2", "E3", "商务委", "M2", "S3"
    elif has("税", "转让定价", "协定"):
        dm, gen, rule, ev, dep, tm, st = "C2", "政策法规", "R1", "E1", "税务", "M2", "S2"
    elif has("EPC", "承包工程", "国际工程"):
        dm, gen, rule, ev, dep, tm, st = "Z2", "实操指引", "R4", "E2", "商务委", "M3", "S1"
    elif has("备案", "核准", "境外投资", "对外投资", "ODI"):
        dm, gen, rule, ev, dep, tm, st = "G1", "政策法规", "R0", "E1", "商务委", "M2", "S2"
    elif has("劳务", "用工", "派遣"):
        dm, gen, rule, ev, dep, tm, st = "O4", "普通资讯", "R0", "E3", "人社(商务委)", "M3", "S2"
    else:
        dm, gen, rule, ev, dep, tm, st = "X9", "普通资讯", "R0", "E3", "其他", "M3", "S0"
    geo = "通用"
    for k in list(COUNTRY_CODE) + list(REGION_CODE):
        if k in t:
            geo = k
            break
    origin = "目的地国" if geo != "通用" else "国内"
    conf = "中" if dm == "X9" else "高"
    return {"阶段": st, "服务大类": dm, "体裁": gen, "隐性规则": rule, "证据": ev,
            "来源": origin, "归口": dep, "国别区域": geo, "行业形态": "通用/通用",
            "时效": tm, "置信度": conf, "理由": "[mock]规则桩示意", "建议消费": "出海智询"}
