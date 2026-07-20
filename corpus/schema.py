"""八维"身份证" entry metadata: parsing, validation, and serialization.

Parses annotation rows from the LLM 标注工具包 pipeline output (标注明细.csv /
标注明细_业务视图.csv, flat Chinese columns) as well as the decorated value
style used in the curated 试标注样本 (e.g. "S2④(副S3⑤)", "O1金融(副Z2)",
"E2/丝路基金(U0)"). Every parse yields an EntryRecord plus a list of issues;
per 公理一 nothing is unclassifiable — unknown values fall back to the facet's
兜底 value and are flagged for the manual review queue.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .codetable import CodeTable

ERROR = "error"
WARN = "warn"

_ENTRY_ID_RE = re.compile(r"^S[0-4]-(?:[GCOZ]\d|X9)-[^-]+-[A-Z]{2,4}-[0-9A-Za-z]{4,8}$")
_SPLIT_RE = re.compile(r"[·,，、/;；\s]+")

# Column aliases: curated-sample sheet headers -> pipeline CSV headers.
_HEADER_ALIASES = {
    "标题(节选)": "title",
    "标题": "title",
    "来源/平台": "source",
    "阶段(主/副)": "阶段",
    "服务大类(主/副)": "服务大类",
    "体裁F1": "体裁",
    "隐性规则F2": "隐性规则",
    "证据E/归口部门": "证据",
    "来源域": "来源",
    "国别·区域": "国别区域",
    "行业/出海形态": "行业形态",
    "时效/状态": "时效",
    "置信": "置信度",
    "消费智能体": "建议消费",
    "疑难/裁定要点": "理由",
}


@dataclass
class Issue:
    level: str  # error | warn
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.field}: {self.message}"


@dataclass
class EntryRecord:
    """One corpus entry's eight-dimension metadata (spec §4.2 schema)."""

    entry_id: str = ""
    title: str = ""
    # 1 阶段 (主轴)
    stage: str = "S0"
    stage_ext: list = field(default_factory=list)
    # 2 服务大类 (副轴)
    domain: str = "X9"
    domain_ext: list = field(default_factory=list)
    # 3 体裁 F1
    genre: str = "其他"
    # 4 隐性规则 F2
    rule_type: list = field(default_factory=lambda: ["R0"])
    # 5 来源权威 F3
    evidence: str = "E4"
    origin: str = "混合"
    gov_dept: list = field(default_factory=lambda: ["其他"])
    source_url: str = ""
    source_site: str = ""
    source_relpath: str = ""
    # 6 国别区域 F4
    geo_scope: str = "通用"  # 通用 | 区域 | 单国
    geo_region: list = field(default_factory=list)
    geo_country: list = field(default_factory=list)  # ISO 3166-1 alpha-3
    geo_country_names: list = field(default_factory=list)
    # 7 行业形态 F5
    industry: list = field(default_factory=lambda: ["通用"])
    mode: list = field(default_factory=list)
    # 8 时效状态 F6
    timeliness: str = "M3"
    lifecycle_state: str = "已入库"
    effective_date: str | None = None
    review_due: str | None = None
    # 质检 / 消费
    confidence: float | None = None
    confidence_raw: str = ""
    reason: str = ""
    consumer_agents: list = field(default_factory=list)
    # 业务视图 (derived, optional)
    business_code: str = ""
    business_scene: str = ""
    business_class: str = ""
    business_priority: str = ""
    business_pending: bool = False
    # bookkeeping
    spec_version: str = ""
    issues: list = field(default_factory=list)

    # ---- derived ----------------------------------------------------------

    @property
    def needs_review(self) -> bool:
        """低置信 / X9 / any error → 人工复核队列 (spec §4.4)."""
        if self.domain == "X9":
            return True
        if self.confidence is not None and self.confidence < 0.5:
            return True
        return any(i.level == ERROR for i in self.issues)

    def geo_display(self) -> str:
        parts = self.geo_region + self.geo_country_names
        return "·".join(parts) if parts else "通用"

    # ---- serialization -----------------------------------------------------

    def to_metadata(self) -> dict:
        """The structured record stored in documents.metadata / frontmatter."""
        meta = {
            "spec_version": self.spec_version,
            "entry_id": self.entry_id,
            "stage": self.stage,
            "stage_ext": self.stage_ext,
            "domain": self.domain,
            "domain_ext": self.domain_ext,
            "genre": self.genre,
            "rule_type": self.rule_type,
            "evidence": self.evidence,
            "origin": self.origin,
            "gov_dept": self.gov_dept,
            "geo_scope": self.geo_scope,
            "geo_region": self.geo_region,
            "geo_country": self.geo_country,
            "geo_country_names": self.geo_country_names,
            "industry": self.industry,
            "mode": self.mode,
            "timeliness": self.timeliness,
            "lifecycle_state": self.lifecycle_state,
            "effective_date": self.effective_date,
            "review_due": self.review_due,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "source_site": self.source_site,
            "source_relpath": self.source_relpath,
            "consumer_agents": self.consumer_agents,
            "reason": self.reason,
        }
        if self.business_code:
            meta["business"] = {
                "code": self.business_code,
                "scene": self.business_scene,
                "class": self.business_class,
                "priority": self.business_priority,
                "pending": self.business_pending,
            }
        return meta

    def to_tags(self) -> list:
        """Facet tags so existing tag filtering works before phase-2 search."""
        tags = [self.stage, self.domain, self.genre, self.timeliness, self.origin]
        tags += [r for r in self.rule_type if r != "R0"]
        tags += self.geo_region + self.geo_country_names
        tags += [i for i in self.industry if i != "通用"]
        if self.business_code and self.business_code != "待定":
            tags.append(self.business_code)
        return list(dict.fromkeys(t for t in tags if t))


def normalize_headers(row: dict) -> dict:
    """Map curated-sample column names onto the pipeline CSV dialect."""
    out = {}
    for key, value in row.items():
        k = str(key or "").strip()
        out[_HEADER_ALIASES.get(k, k)] = value
    return out


def compute_review_due(timeliness: str, table: CodeTable, base: date) -> str:
    return (base + timedelta(days=table.review_days(timeliness))).isoformat()


def make_entry_id(rec: EntryRecord, relpath: str, table: CodeTable) -> str:
    """Composite human-readable id 阶段-大类-体裁短码-国别短码-流水 (spec §4.3).

    Serial derivation matches classify_pipeline.make_entry_id: a stable short
    hash of relpath, so re-runs and resumed runs keep the same id.
    """
    genre_short = table.genre_short(rec.genre)
    if rec.geo_country:
        geo = rec.geo_country[0]
    elif rec.geo_region:
        geo = table.region_code(rec.geo_region[0]) or "OTH"
    else:
        geo = "GEN"
    seed = relpath or f"{rec.title}|{rec.source_url}"
    serial = hashlib.md5(seed.encode("utf-8")).hexdigest()[:5].upper()
    return f"{rec.stage}-{rec.domain}-{genre_short}-{geo}-{serial}"


def _parse_identity(rec: EntryRecord, val, issues: list) -> None:
    rec.title = val("title")
    if not rec.title:
        issues.append(Issue(ERROR, "title", "标题缺失"))
    rec.source_url = val("url")
    rec.source_site = val("source")
    rec.source_relpath = val("relpath")
    rec.reason = val("理由")


def _parse_axes(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    """1 阶段 + 2 服务大类 — first code is 主, remainder 副."""
    stages = table.extract_stages(val("阶段"))
    if stages:
        rec.stage, rec.stage_ext = stages[0], stages[1:]
    else:
        issues.append(Issue(ERROR, "阶段", f"无法识别: {val('阶段')!r} → 兜底 S0"))

    domains = table.extract_domains(val("服务大类"))
    if domains:
        rec.domain, rec.domain_ext = domains[0], domains[1:]
    else:
        issues.append(Issue(ERROR, "服务大类", f"无法识别: {val('服务大类')!r} → 兜底 X9"))


def _parse_genre_rules(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    """3 体裁 + 4 隐性规则."""
    genre, aliased = table.normalize_genre(val("体裁"))
    if genre is None:
        issues.append(Issue(ERROR, "体裁", f"码表外取值: {val('体裁')!r} → 兜底 其他"))
    else:
        rec.genre = genre
        if aliased:
            issues.append(Issue(WARN, "体裁", f"别名 {val('体裁')!r} → {genre}"))

    rules = table.extract_rules(val("隐性规则"))
    if rules:
        rec.rule_type = rules
    elif val("隐性规则"):
        issues.append(Issue(WARN, "隐性规则", f"无法识别: {val('隐性规则')!r} → R0"))


def _parse_source_facet(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    """5 来源权威: 证据强度 / 来源域 / 归口部门."""
    evidence_raw = val("证据")
    evidence = table.extract_evidence(evidence_raw)
    if evidence:
        rec.evidence = evidence
    else:
        issues.append(Issue(ERROR, "证据", f"无法识别: {evidence_raw!r} → 兜底 E4"))

    origin = table.normalize_origin(val("来源"))
    if origin:
        rec.origin = origin
    else:
        issues.append(Issue(ERROR, "来源", f"码表外取值: {val('来源')!r} → 兜底 混合"))

    # 归口部门 — "牵头(协同)" style keeps both, lead first. Curated sheets embed
    # 归口 in the 证据 cell ("E1/商务委U3"), so fall back to that cell.
    dept_raw = val("归口") or evidence_raw
    dept_tokens = [t for t in re.split(r"[（(）)·/,，、]+", dept_raw) if t and not table.extract_evidence(t)]
    depts = []
    for tok in dept_tokens:
        canon = table.normalize_dept(tok) or table.normalize_dept(re.sub(r"U\d+$", "", tok))
        if canon and canon not in depts:
            depts.append(canon)
    if depts:
        rec.gov_dept = depts
    elif dept_raw:
        rec.gov_dept = ["其他"]
        issues.append(Issue(WARN, "归口", f"码表外部门: {dept_raw!r} → 其他"))
    else:
        issues.append(Issue(ERROR, "归口", "归口部门缺失 → 其他"))


def _parse_geo(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    """6 国别区域 — "通用" / "印尼" / "拉美·哥伦比亚" / "区域=东盟;国别=印尼"."""
    geo_raw = val("国别区域").replace("区域=", "").replace("国别=", "")
    if not geo_raw or "通用" in geo_raw:
        rec.geo_scope = "通用"
        return
    for tok in [t for t in _SPLIT_RE.split(geo_raw) if t]:
        iso = table.country_code(tok)
        if iso:
            if iso not in rec.geo_country:
                rec.geo_country.append(iso)
                rec.geo_country_names.append(tok)
        elif table.region_code(tok):
            if tok not in rec.geo_region:
                rec.geo_region.append(tok)
        else:
            issues.append(Issue(WARN, "国别区域", f"未识别地理取值: {tok!r} (国别按 ISO 3166 补录)"))
    rec.geo_scope = "单国" if rec.geo_country else ("区域" if rec.geo_region else "通用")


def _collect_normalized(raw: str, normalize, field_name: str, issues: list,
                        skip: tuple = ()) -> list:
    """Split a multi-value cell, normalize each token, keep unknowns with a warn."""
    values = []
    for tok in [t for t in _SPLIT_RE.split(raw) if t and t not in skip]:
        canon = normalize(tok)
        if canon is None:
            issues.append(Issue(WARN, field_name, f"码表外取值: {tok!r}"))
            canon = tok  # keep raw; the referenced national standard may extend
        if canon not in values:
            values.append(canon)
    return values


def _parse_industry_mode(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    """7 行业/出海形态 — "行业/形态" split on the first slash."""
    ind_raw = val("行业形态")
    if not ind_raw:
        return
    ind_part, _, mode_part = ind_raw.partition("/")
    rec.industry = _collect_normalized(
        ind_part, table.normalize_industry, "行业", issues, skip=("通用",)
    ) or ["通用"]
    rec.mode = _collect_normalized(
        mode_part, table.normalize_mode, "出海形态", issues, skip=("通用",)
    )


def _parse_time_state(rec: EntryRecord, val, table: CodeTable, issues: list, today: date) -> None:
    """8 时效/状态 — "M2" or "M3/已发布"."""
    time_raw = val("时效")
    timeliness = table.extract_timeliness(time_raw)
    if timeliness:
        rec.timeliness = timeliness
    else:
        issues.append(Issue(ERROR, "时效", f"无法识别: {time_raw!r} → 兜底 M3"))
    state = next((s for s in table.lifecycle_states if s in time_raw), None)
    rec.lifecycle_state = state or "已入库"
    rec.review_due = compute_review_due(rec.timeliness, table, today)


def _parse_quality(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    rec.confidence_raw = val("置信度")
    rec.confidence = table.confidence_value(rec.confidence_raw)
    if rec.confidence is None and rec.confidence_raw:
        issues.append(Issue(WARN, "置信度", f"无法解析: {rec.confidence_raw!r}"))

    rec.consumer_agents = [t for t in _SPLIT_RE.split(val("建议消费")) if t]
    for agent in rec.consumer_agents:
        if agent not in table.consumer_agents:
            issues.append(Issue(WARN, "建议消费", f"能力清单外智能体: {agent!r}"))


def _parse_business(rec: EntryRecord, val, table: CodeTable, issues: list) -> None:
    """业务视图 columns (present when the CSV went through derive_business_view.py)."""
    rec.business_code = val("业务码")
    if not rec.business_code:
        return
    rec.business_scene = val("业务场景") or table.business_scenes.get(rec.business_code, "")
    cls, cls_name, priority = table.business_class_of(rec.business_code)
    rec.business_class = val("业务需求类") or cls_name
    rec.business_priority = val("业务优先级") or priority
    rec.business_pending = val("业务待定") == "是" or rec.business_code == "待定"
    if rec.business_code != "待定" and rec.business_code not in table.business_scenes:
        issues.append(Issue(WARN, "业务码", f"27场景外取值: {rec.business_code!r}"))


def parse_row(
    row: dict,
    table: CodeTable,
    today: date | None = None,
) -> EntryRecord:
    """Parse one annotation CSV row into a validated EntryRecord.

    Tolerant by design: primary/secondary markers ("主(副)"), decorated values,
    and both column dialects are accepted; every fallback is recorded as an
    issue so the import report and review queue stay honest.
    """
    row = normalize_headers(row)
    rec = EntryRecord(spec_version=table.version)
    issues = rec.issues
    today = today or date.today()

    def val(key: str) -> str:
        return str(row.get(key, "") or "").strip()

    _parse_identity(rec, val, issues)
    _parse_axes(rec, val, table, issues)
    _parse_genre_rules(rec, val, table, issues)
    _parse_source_facet(rec, val, table, issues)
    _parse_geo(rec, val, table, issues)
    _parse_industry_mode(rec, val, table, issues)
    _parse_time_state(rec, val, table, issues, today)
    _parse_quality(rec, val, table, issues)
    _parse_business(rec, val, table, issues)

    # entry_id — validate the pipeline's, else regenerate deterministically
    provided = val("entry_id")
    if provided and _ENTRY_ID_RE.match(provided):
        rec.entry_id = provided
    else:
        rec.entry_id = make_entry_id(rec, rec.source_relpath, table)
        if provided:
            issues.append(Issue(WARN, "entry_id", f"格式不合规: {provided!r} → 重派 {rec.entry_id}"))

    # 公理一: after fallbacks every dimension holds a value; flagged rows queue
    # for manual review unless the sheet explicitly set a later lifecycle state.
    if rec.needs_review and rec.lifecycle_state == "已入库":
        rec.lifecycle_state = "待复核"

    return rec
