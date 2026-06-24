"""Leave-one-evidence-family-out estimated reliability tracker.

Legacy ``mean_dice`` files compared predictions with a current pseudo reference
and could create a self-reinforcing ranking loop.  This v2 tracker stores only
cross-family observations.  Its values are estimated pseudo-label reliability,
not expert accuracy.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "estimated_model_reliability_v2"
MIN_CASES_FOR_EXPLOITATION = 3


class OrganModelPerformance:
    def __init__(self, json_path: str | Path):
        requested = Path(json_path)
        self.legacy_path: Path | None = None
        if requested.exists():
            try:
                existing = json.loads(requested.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            if existing and existing.get("schema_version") != SCHEMA_VERSION:
                self.legacy_path = requested
                requested = requested.with_name(requested.stem + ".estimated_v2.json")
        self._path = requested
        self._doc = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metric_family": "leave_one_evidence_family_out_estimated_reliability",
            "accuracy_warning": "Values are pseudo-label reliability estimates, not expert accuracy.",
            "global_models": {}, "organ_families": {}, "organs": {}, "insufficient_observations": [],
        }

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._empty()
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
            return doc if doc.get("schema_version") == SCHEMA_VERSION else self._empty()
        except Exception:
            return self._empty()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._doc, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _update_stat(bucket: dict[str, Any], key: str, score: float) -> dict[str, Any]:
        entry = bucket.setdefault(key, {"n_cases": 0, "mean_estimated_reliability": 0.0, "m2": 0.0, "std_estimated_reliability": 0.0})
        n = int(entry["n_cases"]) + 1
        mean = float(entry["mean_estimated_reliability"])
        m2 = float(entry.get("m2", 0.0))
        delta = score - mean
        mean += delta / n
        m2 += delta * (score - mean)
        entry.update({
            "n_cases": n, "mean_estimated_reliability": round(mean, 6), "m2": round(m2, 8),
            "std_estimated_reliability": round(math.sqrt(m2 / (n - 1)), 6) if n >= 2 else 0.0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })
        return entry

    def record_insufficient(self, *, organ: str, model: str, evidence_family: str, other_family_count: int, case_id: str | None = None, reason: str = "requires_two_other_evidence_families") -> None:
        rows = self._doc.setdefault("insufficient_observations", [])
        rows.append({"organ": organ, "model": model, "evidence_family": evidence_family, "other_family_count": int(other_family_count), "case_id": case_id, "reason": reason})
        del rows[:-10000]
        self._save()

    def update_leave_one_family_out(
        self, *, organ: str, organ_family: str, model: str, evidence_family: str,
        dice: float, nsd: float, ct_support: float, anatomy_plausibility: float,
        other_family_count: int, case_id: str | None = None,
    ) -> bool:
        if int(other_family_count) < 2:
            self.record_insufficient(organ=organ, model=model, evidence_family=evidence_family, other_family_count=other_family_count, case_id=case_id)
            return False
        values = [dice, nsd, ct_support, anatomy_plausibility]
        if any(v is None or not math.isfinite(float(v)) for v in values):
            self.record_insufficient(organ=organ, model=model, evidence_family=evidence_family, other_family_count=other_family_count, case_id=case_id, reason="missing_observation_component")
            return False
        score = max(0.0, min(1.0, 0.6 * float(dice) + 0.2 * float(nsd) + 0.1 * float(ct_support) + 0.1 * float(anatomy_plausibility)))
        self._update_stat(self._doc["global_models"], model, score)
        family_bucket = self._doc["organ_families"].setdefault(organ_family or "unknown", {})
        self._update_stat(family_bucket, model, score)
        organ_bucket = self._doc["organs"].setdefault(organ, {})
        entry = self._update_stat(organ_bucket, model, score)
        entry.update({"evidence_family": evidence_family, "last_case_id": case_id, "observation_metric": "loo_family_estimated_reliability"})
        self._save()
        return True

    def update(self, organ: str, model: str, dice_score: float) -> None:
        """Reject the legacy self-referential update API.

        Kept only to make accidental callers fail loudly instead of silently
        rebuilding the old feedback loop.
        """
        raise RuntimeError("Legacy pseudo-reference Dice updates are disabled; use update_leave_one_family_out")

    @staticmethod
    def _shrunk_mean(entry: dict[str, Any] | None, prior: float, strength: float) -> float:
        if not entry:
            return prior
        n = float(entry.get("n_cases", 0))
        mean = float(entry.get("mean_estimated_reliability", prior))
        return (n * mean + strength * prior) / (n + strength) if n + strength > 0 else prior

    def get_estimated_reliability(self, organ: str, model: str, organ_family: str = "unknown") -> float | None:
        global_entry = self._doc["global_models"].get(model)
        if not global_entry:
            return None
        global_mean = float(global_entry.get("mean_estimated_reliability", 0.5))
        family_mean = self._shrunk_mean(self._doc["organ_families"].get(organ_family, {}).get(model), global_mean, 20.0)
        return round(self._shrunk_mean(self._doc["organs"].get(organ, {}).get(model), family_mean, 10.0), 6)

    def get_stats(self, organ: str, model: str) -> dict[str, Any] | None:
        entry = self._doc["organs"].get(organ, {}).get(model)
        if not entry:
            return None
        return {**entry, "estimated_reliability": self.get_estimated_reliability(organ, model)}

    def should_run_all(self, organ: str) -> bool:
        data = self._doc["organs"].get(organ, {})
        return not data or any(int(v.get("n_cases", 0)) < MIN_CASES_FOR_EXPLOITATION for v in data.values())

    def get_top_k_models(self, organ: str, k: int = 2) -> list[str]:
        data = self._doc["organs"].get(organ, {})
        qualified = [(model, float(row.get("mean_estimated_reliability", 0.0))) for model, row in data.items() if int(row.get("n_cases", 0)) >= MIN_CASES_FOR_EXPLOITATION]
        qualified.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _ in qualified[:k]] if qualified else list(data)

    def summary(self) -> dict[str, Any]:
        return {**self._doc, "storage_path": str(self._path), "legacy_path": str(self.legacy_path) if self.legacy_path else None}
