"""业务视图推导(derive_business_view.py 的库化移植)。

八维 → 业务码(7类27场景):先覆盖规则,再服务大类默认映射。纯函数、确定性;
场景名/需求类名以码表为准(schema.parse_row 会从码表回填),此处只产码。
"""

from __future__ import annotations

import re


def _first_dom(s) -> str:
    m = re.search(r"[GCOZX]\d|X9", str(s))
    return m.group(0) if m else "X9"


def _has(text: str, *ks) -> bool:
    return any(k in text for k in ks)


def derive_business_code(dom, rule, genre, timeliness, stage, text: str) -> str:
    g = str(genre); rl = str(rule); tm = str(timeliness); dm = _first_dom(dom)
    # —— 覆盖规则(优先级高)——
    if "风险预警" in g or tm.startswith("M1"):
        return "B6.25"
    if "R1" in rl and _has(text, "数据", "个人信息", "出境", "隐私", "GDPR", "本地化存储"):
        return "B4.14"
    if "资源名录" in g or "名录" in g:
        return "B7.26"
    if ("案例" in g) or ("规则提炼" in g):
        return "B7.27"
    if "R6" in rl:
        return "B4.15" if _has(text, "用工", "劳动", "社保", "工时", "劳工", "薪酬") else "B4.18"
    if _has(text, "关税", "原产地", "反倾销", "反补贴", "壁垒", "TBT", "保障措施"):
        return "B1.3"
    # —— 服务大类默认映射 ——
    if dm == "G1":
        return "B3.8" if _has(text, "税", "关税", "增值税", "所得税") else "B1.2"
    if dm == "G2":
        if _has(text, "准入", "负面清单", "禁限入", "外资准入"):
            return "B1.1"
        if _has(text, "商机", "项目", "政府采购", "招投标"):
            return "B2.7"
        if _has(text, "产业链", "园区", "配套", "渠道"):
            return "B2.5"
        if _has(text, "营商环境", "政务", "腐败", "透明度", "效率"):
            return "B5.20"
        if _has(text, "双边", "BIT", "外交", "协定"):
            return "B5.21"
        if _has(text, "产业", "行业") and stage and "S1" in str(stage):
            return "B2.6"
        return "B5.19"
    if dm == "G3":
        return "B6.25"
    if dm == "G4":
        return "B6.24"
    if dm == "C1":
        if _has(text, "数据", "隐私", "出境"):
            return "B4.14"
        if _has(text, "仲裁", "争端", "诉讼", "救济"):
            return "B6.24"
        return "B7.26" if "名录" in g else "B4.14"
    if dm == "C2":
        return "B3.8"
    if dm == "C3":
        return "B3.9"
    if dm == "C4":
        return "B4.13"
    if dm == "C5":
        return "B4.12"
    if dm == "O1":
        return "B3.9" if _has(text, "外汇", "汇回", "结算", "换汇") else "B3.10"
    if dm == "O2":
        return "B3.11" if _has(text, "汇率", "金融稳定") else "B6.25"
    if dm == "O3":
        return "B2.5" if _has(text, "产业链", "配套") else "B4.18"
    if dm == "O4":
        return "B4.16" if _has(text, "签证", "居留", "工作许可") else "B4.15"
    if dm in ("O5", "O6"):
        return "B4.18"
    if dm == "Z1":
        return "B2.5"
    if dm == "Z2":
        return "B4.17" if _has(text, "用地", "厂房", "不动产") else "B2.5"
    if dm == "Z3":
        return "B4.12"
    if dm == "Z4":
        return "B6.23"
    return "待定"


def apply_business_view(rec: dict) -> dict:
    """就地为标注行补业务视图列(业务码/业务待定;名称由码表回填)。"""
    text = " ".join(str(rec.get(k, "") or "") for k in ("title", "理由", "国别区域", "行业形态"))
    code = derive_business_code(rec.get("服务大类", ""), rec.get("隐性规则", ""),
                                rec.get("体裁", ""), rec.get("时效", ""),
                                rec.get("阶段", ""), text)
    rec["业务码"] = code
    rec["业务待定"] = "是" if code == "待定" else ""
    return rec
