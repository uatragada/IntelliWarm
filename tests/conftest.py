"""
Shared pytest scaffolding for IntelliWarm.
"""

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
