"""Make the repo-root `corpus` package importable for these tests.

Appended (not prepended) so the pip-installed `mcp` package and the api/
sys.path entry from the root conftest keep resolution priority.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
