"""Makes the repository importable as `core.*` / `scorer.*` / `app.*` under pytest,
whether the suite is run from the repo root or from anywhere else.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
