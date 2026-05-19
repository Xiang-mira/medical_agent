from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from .json_utils import to_jsonable


def append_review_item(queue_path: str | Path, item: dict) -> str:
    p = Path(queue_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    item = dict(item)
    item.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")
    return str(p.resolve())


def read_review_queue(queue_path: str | Path) -> list[dict]:
    p = Path(queue_path)
    if not p.exists(): return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: out.append(json.loads(line))
            except Exception as exc:
                print(f"[human_review] warning: skipped malformed review queue line: {exc}", file=sys.stderr)
    return out
