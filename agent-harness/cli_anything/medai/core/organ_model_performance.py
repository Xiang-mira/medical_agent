"""Organ-model performance tracker using Welford online statistics.

Tracks per-(organ, model) DSC statistics across EM-loop rounds so that
multimodel_loop.py can progressively narrow down which specialist model
performs best for each organ, rather than running all models every round.

Exploration phase: any model with < MIN_CASES_FOR_EXPLOITATION cases for a
given organ → run all models (explore).
Exploitation phase: all models have >= MIN_CASES_FOR_EXPLOITATION cases →
run only the top-K models by mean DSC.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIN_CASES_FOR_EXPLOITATION = 3  # minimum cases before switching from explore to exploit


class OrganModelPerformance:
    """Persistent per-(organ, model) DSC statistics backed by a JSON file.

    JSON schema:
    {
      "<organ>": {
        "<model_key>": {
          "n_cases": int,
          "mean_dice": float,
          "m2": float,          # Welford M2 accumulator for variance
          "std_dice": float,    # sqrt(M2 / (n-1)), 0.0 when n < 2
          "last_updated": str   # ISO timestamp
        }
      }
    }
    """

    def __init__(self, json_path: str | Path):
        self._path = Path(json_path)
        self._data: dict[str, dict[str, dict[str, Any]]] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── update ───────────────────────────────────────────────────────────────

    def update(self, organ: str, model: str, dice_score: float) -> None:
        """Welford online update for (organ, model) statistics."""
        if organ not in self._data:
            self._data[organ] = {}
        if model not in self._data[organ]:
            self._data[organ][model] = {"n_cases": 0, "mean_dice": 0.0, "m2": 0.0, "std_dice": 0.0, "last_updated": ""}

        entry = self._data[organ][model]
        n = entry["n_cases"] + 1
        mean = entry["mean_dice"]
        m2 = entry["m2"]

        # Welford's online algorithm
        delta = dice_score - mean
        mean += delta / n
        delta2 = dice_score - mean
        m2 += delta * delta2

        entry["n_cases"] = n
        entry["mean_dice"] = round(mean, 6)
        entry["m2"] = round(m2, 8)
        entry["std_dice"] = round(math.sqrt(m2 / (n - 1)), 6) if n >= 2 else 0.0
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()

        self._save()

    # ── query ────────────────────────────────────────────────────────────────

    def should_run_all(self, organ: str) -> bool:
        """True if any model has fewer than MIN_CASES_FOR_EXPLOITATION cases.

        During the exploration phase we run all available models so every model
        gets a fair chance to demonstrate its performance on this organ.
        """
        organ_data = self._data.get(organ, {})
        if not organ_data:
            return True  # no data at all → explore
        return any(
            entry.get("n_cases", 0) < MIN_CASES_FOR_EXPLOITATION
            for entry in organ_data.values()
        )

    def get_top_k_models(self, organ: str, k: int = 2) -> list[str]:
        """Return the top-K model keys by mean DSC for this organ.

        Only considers models with >= MIN_CASES_FOR_EXPLOITATION cases.
        Falls back to all tracked models if fewer than k qualify.
        """
        organ_data = self._data.get(organ, {})
        qualified = [
            (model, entry["mean_dice"])
            for model, entry in organ_data.items()
            if entry.get("n_cases", 0) >= MIN_CASES_FOR_EXPLOITATION
        ]
        if not qualified:
            # Not enough data — return all tracked models
            return list(organ_data.keys())
        qualified.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in qualified[:k]]

    def get_stats(self, organ: str, model: str) -> dict[str, Any] | None:
        """Return the statistics dict for (organ, model), or None if not tracked."""
        return self._data.get(organ, {}).get(model)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all tracked (organ, model) pairs."""
        total_pairs = sum(len(v) for v in self._data.values())
        total_cases = sum(
            entry.get("n_cases", 0)
            for organ_data in self._data.values()
            for entry in organ_data.values()
        )
        return {
            "num_organs_tracked": len(self._data),
            "num_organ_model_pairs": total_pairs,
            "total_case_evaluations": total_cases,
            "min_cases_for_exploitation": MIN_CASES_FOR_EXPLOITATION,
            "organs": {
                organ: {
                    model: {
                        "n": entry["n_cases"],
                        "mean_dice": entry["mean_dice"],
                        "std_dice": entry["std_dice"],
                    }
                    for model, entry in sorted(
                        organ_data.items(),
                        key=lambda x: x[1].get("mean_dice", 0),
                        reverse=True,
                    )
                }
                for organ, organ_data in sorted(self._data.items())
            },
        }
