from __future__ import annotations

import sys
from pathlib import Path

# Allow `python run_medai_cli.py ...` to work even before `pip install -e agent-harness`.
ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "agent-harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cli_anything.medai.medai_cli import main

if __name__ == "__main__":
    main()
