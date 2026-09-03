"""Vercel's Python entrypoint for the console.

Vercel looks for a top-level `app` in this file. There is no `data/processed`
here - only the committed `demo/` bundle - so demo mode is forced on before
the console module is imported, the same override `orbweaver/console/demo.py`
already defines for exactly this case. `sys.path` needs the repo root added
explicitly: Vercel's Python runtime does not put the project root on the path
the way running `uvicorn` from a checkout does.
"""
import os
import sys
from pathlib import Path

os.environ["ORBWEAVER_DEMO"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbweaver.console.app import app  # noqa: E402
