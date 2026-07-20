"""受控码表 (controlled vocabularies) for the 八维 classification, v-managed.

Code tables live in corpus/codetables/<version>.json and mirror the spec
workbook《出海智能体LLM-WIKI语料分类·标注规范表.xlsx》. Per the spec's 码表
版本管理 rule (§5.2), a new vocabulary version is a new JSON file; existing
versions are immutable so historical annotations stay interpretable.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_VERSION = "v2026.06"

_TABLES_DIR = Path(__file__).resolve().parent / "codetables"

_STAGE_RE = re.compile(r"S[0-4]")
_DOMAIN_RE = re.compile(r"[GCOZ]\d|X9")
_RULE_RE = re.compile(r"R[0-6]")
_EVIDENCE_RE = re.compile(r"E[0-4]")
_TIMELINESS_RE = re.compile(r"M[1-3]")


class CodeTable:
    """One immutable version of the controlled vocabularies."""

    def __init__(self, data: dict):
        self._data = data
        self.version: str = data["version"]
        self.stages: dict = data["stages"]
        self.domains: dict = data["domains"]
        self.genres: dict = data["genres"]
        self.rules: dict = data["rules"]
        self.evidence: dict = data["evidence"]
        self.gov_depts: dict = data["gov_depts"]
        self.origins: dict = data["origins"]
        self.regions: dict = data["regions"]
        self.countries: dict = data["countries"]
        self.industries: list = data["industries"]
        self.modes: list = data["modes"]
        self.entry_modes: list = data["entry_modes"]
        self.timeliness: dict = data["timeliness"]
        self.lifecycle_states: list = data["lifecycle_states"]
        self.business_classes: dict = data["business_classes"]
        self.business_scenes: dict = data["business_scenes"]
        self.consumer_agents: list = data["consumer_agents"]

        self._genre_names = {g["name"] for g in self.genres.values()}
        self._genre_short = {g["name"]: g["short"] for g in self.genres.values()}
        self._dept_names = set(self.gov_depts.values())
        self._origin_names = set(self.origins.values())

    # ---- generic accessors -------------------------------------------------

    def raw(self, key: str):
        return self._data.get(key)

    @property
    def stage_codes(self) -> set:
        return set(self.stages)

    @property
    def domain_codes(self) -> set:
        return set(self.domains)

    # ---- normalization -----------------------------------------------------

    def normalize_genre(self, value: str) -> tuple[str | None, bool]:
        """Return (canonical genre name, alias_applied). None if unknown."""
        v = str(value or "").strip()
        m = re.fullmatch(r"T(\d+)", v)
        if m and v in self.genres:
            return self.genres[v]["name"], False
        if v in self._genre_names:
            return v, False
        alias = self._data.get("genre_aliases", {}).get(v)
        if alias:
            return alias, True
        return None, False

    def genre_short(self, name: str) -> str:
        return self._genre_short.get(name, "其他")

    def normalize_dept(self, value: str) -> str | None:
        """Canonical department name from a name, alias, or U-code."""
        v = str(value or "").strip()
        if v in self.gov_depts:  # U-code
            return self.gov_depts[v]
        if v in self._dept_names:
            return v
        return self._data.get("gov_dept_aliases", {}).get(v)

    def normalize_origin(self, value: str) -> str | None:
        v = str(value or "").strip()
        if v in self.origins:  # O-code
            return self.origins[v]
        if v in self._origin_names:
            return v
        return None

    def country_code(self, name: str) -> str | None:
        """ISO 3166-1 alpha-3 for a Chinese country name (or pass-through code)."""
        v = str(name or "").strip()
        if v in self.countries:
            return self.countries[v]
        v_up = v.upper()
        alias = self._data.get("country_aliases", {}).get(v_up)
        if alias:
            return alias
        if re.fullmatch(r"[A-Z]{2,4}", v_up) and v_up in set(self.countries.values()):
            return v_up
        return None

    def region_code(self, name: str) -> str | None:
        return self.regions.get(str(name or "").strip())

    def normalize_industry(self, value: str) -> str | None:
        v = str(value or "").strip()
        if v in self.industries:
            return v
        if v in self._data.get("gbt4754", {}):  # GB/T 4754 门类码 A–T
            return self._data["gbt4754"][v]["simplified"]
        return self._data.get("industry_aliases", {}).get(v)

    def normalize_mode(self, value: str) -> str | None:
        v = str(value or "").strip()
        if v in self.modes or v in self.entry_modes:
            return v
        return self._data.get("mode_aliases", {}).get(v)

    def confidence_value(self, value) -> float | None:
        """Normalize 置信度 — accepts 高/中/低 or a numeric string — to 0..1."""
        v = str(value or "").strip()
        if v in self._data.get("confidence_words", {}):
            return self._data["confidence_words"][v]
        try:
            f = float(v)
        except ValueError:
            return None
        return f if 0.0 <= f <= 1.0 else None

    def review_days(self, timeliness_code: str) -> int:
        entry = self.timeliness.get(timeliness_code)
        default = self.timeliness[self._data["timeliness_default"]]
        return (entry or default)["review_days"]

    def business_class_of(self, scene_code: str) -> tuple[str, str, str]:
        """(class code, class name, priority) for a scene code like 'B4.14'."""
        cls = scene_code.split(".")[0] if scene_code and scene_code != "待定" else "待定"
        info = self.business_classes.get(cls)
        if info is None:
            return "待定", "待定", "-"
        return cls, info["name"], info["priority"]

    # ---- code extraction from free-form annotation strings -----------------

    @staticmethod
    def extract_stages(raw: str) -> list:
        return list(dict.fromkeys(_STAGE_RE.findall(str(raw or ""))))

    @staticmethod
    def extract_domains(raw: str) -> list:
        return list(dict.fromkeys(_DOMAIN_RE.findall(str(raw or ""))))

    @staticmethod
    def extract_rules(raw: str) -> list:
        return list(dict.fromkeys(_RULE_RE.findall(str(raw or ""))))

    @staticmethod
    def extract_evidence(raw: str) -> str | None:
        m = _EVIDENCE_RE.search(str(raw or ""))
        return m.group(0) if m else None

    @staticmethod
    def extract_timeliness(raw: str) -> str | None:
        m = _TIMELINESS_RE.search(str(raw or ""))
        return m.group(0) if m else None


@lru_cache(maxsize=4)
def load(version: str = DEFAULT_VERSION) -> CodeTable:
    """Load a code-table version (cached). Raises FileNotFoundError if absent."""
    filename = version.replace(".", "_") + ".json"
    path = _TABLES_DIR / filename
    if not path.exists():
        available = sorted(p.stem.replace("_", ".") for p in _TABLES_DIR.glob("v*.json"))
        raise FileNotFoundError(
            f"码表版本 {version} 不存在 ({path}); 可用版本: {available}"
        )
    return CodeTable(json.loads(path.read_text(encoding="utf-8")))
