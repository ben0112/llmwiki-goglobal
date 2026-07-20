"""出海智能体语料分类体系 (Go-Global corpus classification) integration.

Implements the v2026.06 spec's eight-dimension ("八维身份证") metadata layer
for LLM Wiki: versioned controlled vocabularies (code tables), entry metadata
parsing/validation, and an importer that loads `标注明细.csv` annotation
output into an LLM Wiki workspace.
"""

SPEC_VERSION = "v2026.06"
