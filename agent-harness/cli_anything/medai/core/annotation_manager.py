from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class AnnotationManager:
    """
    Versioned annotation storage for one case.

    Layout:
        <base_dir>/<case_id>/
            raw/                        original reference labels (read-only copy)
            predictions/
                round_000/              model predictions per round
                round_001/
            updated/                    currently accepted / updated annotation
            decisions.jsonl             append-only log of all VLM / verifier decisions
    """

    def __init__(self, base_dir: str | Path, case_id: str):
        self.case_id = case_id
        self.base = Path(base_dir).resolve() / case_id
        self.raw_dir = self.base / "raw"
        self.predictions_dir = self.base / "predictions"
        self.updated_dir = self.base / "updated"
        self.decisions_path = self.base / "decisions.jsonl"
        for d in (self.raw_dir, self.predictions_dir, self.updated_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── raw ──────────────────────────────────────────────────────────────────

    def save_raw(self, organ: str, source_mask: str | Path) -> Path:
        """Copy reference annotation into raw/. Never overwrite existing."""
        dst = self.raw_dir / f"{organ}.nii.gz"
        if not dst.exists() and Path(source_mask).exists():
            shutil.copy2(source_mask, dst)
        return dst

    def get_raw(self, organ: str) -> Path | None:
        p = self.raw_dir / f"{organ}.nii.gz"
        return p if p.exists() else None

    # ── predictions ──────────────────────────────────────────────────────────

    def save_prediction(self, organ: str, source_mask: str | Path, round_idx: int) -> Path:
        round_dir = self.predictions_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        dst = round_dir / f"{organ}.nii.gz"
        if Path(source_mask).exists():
            shutil.copy2(source_mask, dst)
        return dst

    def get_prediction(self, organ: str, round_idx: int) -> Path | None:
        p = self.predictions_dir / f"round_{round_idx:03d}" / f"{organ}.nii.gz"
        return p if p.exists() else None

    # ── updated ───────────────────────────────────────────────────────────────

    def apply_update(
        self,
        organ: str,
        winner: str,
        round_idx: int,
        candidate_a: "Path | None" = None,
        candidate_b: "Path | None" = None,
    ) -> "Path | None":
        """
        Copy the winning annotation to updated/.

        candidate_a / candidate_b are the actual files that were shown to the VLM.
        They must be passed explicitly so that:
          - winner='A' copies candidate_a (which may already be updated/, not raw/)
          - winner='B' copies candidate_b (this round's prediction)
        Falls back to get_raw / get_prediction only when the caller omits them.
        """
        if winner == "A":
            src = candidate_a if candidate_a is not None else self.get_raw(organ)
        elif winner == "B":
            src = candidate_b if candidate_b is not None else self.get_prediction(organ, round_idx)
        else:
            return None
        if src is None or not src.exists():
            return None
        dst = self.updated_dir / f"{organ}.nii.gz"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst

    def get_current(self, organ: str) -> Path | None:
        """Return updated annotation if exists, else raw."""
        updated = self.updated_dir / f"{organ}.nii.gz"
        if updated.exists():
            return updated
        return self.get_raw(organ)

    # ── decisions log ─────────────────────────────────────────────────────────

    def log_decision(self, organ: str, round_idx: int, decision: dict) -> None:
        entry = {
            "case_id": self.case_id,
            "organ": organ,
            "round": round_idx,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **decision,
        }
        with self.decisions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_history(self) -> list[dict]:
        if not self.decisions_path.exists():
            return []
        out = []
        for line in self.decisions_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        raw_organs = [p.stem.replace(".nii", "") for p in self.raw_dir.glob("*.nii.gz")]
        updated_organs = [p.stem.replace(".nii", "") for p in self.updated_dir.glob("*.nii.gz")]
        history = self.get_history()
        rounds_done = sorted({h["round"] for h in history})
        return {
            "case_id": self.case_id,
            "base_dir": str(self.base),
            "raw_organs": raw_organs,
            "updated_organs": updated_organs,
            "num_decisions": len(history),
            "rounds_completed": rounds_done,
            "decisions": history,
        }
