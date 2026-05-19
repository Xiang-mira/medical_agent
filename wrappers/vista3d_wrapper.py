#!/usr/bin/env python3
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/vista3d_predict_and_split.py"), run_name="__main__")
