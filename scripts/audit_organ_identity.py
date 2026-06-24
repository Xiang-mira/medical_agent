#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.organ_taxonomy import load_taxonomy, normalize_canonical_id, taxonomy_entry


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _selection_rows(estep: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((estep / "annotation_versions").glob("*/selection_metadata.json")):
        doc = _read_json(path, {})
        items = doc.get("selected_organs") or doc.get("selections") or doc.get("rows") or []
        if isinstance(items, list):
            rows.extend({"metadata_path": str(path), **item} for item in items if isinstance(item, dict))
    return rows


def _source_label(row: dict[str, Any]) -> str:
    explicit = row.get("source_local_label")
    if explicit:
        return normalize_canonical_id(explicit)
    source = str(row.get("selected_prediction") or row.get("selected_pre_shapekit_prediction") or row.get("mask_path") or "")
    name = Path(source).name
    return normalize_canonical_id(name[:-7] if name.endswith(".nii.gz") else name)


def audit(estep: Path, taxonomy_path: Path, training_manifest: Path | None) -> dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    rows = _selection_rows(estep)
    findings: list[dict[str, Any]] = []
    candidate_path_findings: list[dict[str, Any]] = []
    dice_csv = estep / "dice_metrics.csv"
    if dice_csv.exists():
        with dice_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                prediction = Path(str(row.get("prediction") or ""))
                name = prediction.name[:-7] if prediction.name.endswith(".nii.gz") else prediction.stem
                organ = normalize_canonical_id(row.get("organ"))
                if name and normalize_canonical_id(name) != organ and prediction.exists():
                    candidate_path_findings.append({
                        "case_id": row.get("case_id"), "organ": organ, "model": row.get("model"),
                        "prediction": str(prediction), "source_local_label": normalize_canonical_id(name),
                        "reason": "candidate_path_canonical_id_mismatch",
                    })
    valid = 0
    selected_keys: set[tuple[str, str]] = set()
    for row in rows:
        organ = normalize_canonical_id(row.get("organ"))
        case_id = str(row.get("case_id") or Path(row["metadata_path"]).parent.name)
        source = _source_label(row)
        resolved = normalize_canonical_id(row.get("resolved_canonical_id") or source)
        entry = taxonomy_entry(taxonomy, organ)
        reasons: list[str] = []
        if entry is None:
            reasons.append("target_missing_from_taxonomy")
        if not source:
            reasons.append("source_local_label_unknown")
        if resolved != organ:
            reasons.append("resolved_canonical_id_mismatch")
        if row.get("identity_status") not in {None, "", "valid"}:
            reasons.extend(str(x) for x in row.get("identity_mismatch_reasons", [row.get("identity_status")]))
        if reasons:
            findings.append({
                "case_id": case_id, "organ": organ, "source_local_label": source,
                "resolved_canonical_id": resolved, "comparison_family": entry.get("comparison_family") if entry else None,
                "reasons": sorted(set(reasons)), "metadata_path": row["metadata_path"],
                "selected_prediction": row.get("selected_prediction"),
            })
        else:
            valid += 1
            selected_keys.add((case_id, organ))

    manifest_rows = _read_json(training_manifest, []) if training_manifest and training_manifest.exists() else []
    contaminated_manifest = []
    sanitized_manifest = []
    unverified_manifest = []
    finding_keys = {(x["case_id"], x["organ"]) for x in findings}
    if isinstance(manifest_rows, list):
        for row in manifest_rows:
            key = (str(row.get("case_id")), normalize_canonical_id(row.get("organ")))
            if key in finding_keys or key not in selected_keys:
                status = "identity_mismatch" if key in finding_keys else "legacy_unverified"
                excluded = {**row, "identity_status": status, "excluded_reason": "selection_identity_not_verified"}
                contaminated_manifest.append(excluded)
                if status == "legacy_unverified":
                    unverified_manifest.append(excluded)
            else:
                sanitized_manifest.append({**row, "identity_status": "valid"})
    audit_status = "identity_mismatch_found" if findings or candidate_path_findings else ("legacy_provenance_unverified" if contaminated_manifest else "success")
    return {
        "stage": "organ_identity_audit",
        "status": audit_status,
        "estep": str(estep),
        "taxonomy": str(taxonomy_path),
        "num_selection_rows": len(rows),
        "num_valid": valid,
        "num_identity_mismatches": len(findings),
        "num_candidate_path_mismatches": len(candidate_path_findings),
        "candidate_path_findings": candidate_path_findings,
        "findings": findings,
        "training_manifest": str(training_manifest) if training_manifest else None,
        "num_manifest_rows": len(manifest_rows) if isinstance(manifest_rows, list) else 0,
        "num_manifest_excluded": len(contaminated_manifest),
        "num_manifest_legacy_unverified": len(unverified_manifest),
        "contaminated_manifest_rows": contaminated_manifest,
        "sanitized_manifest_rows": sanitized_manifest,
        "policy": "Artifacts are never deleted or overwritten; mismatches are quarantined from M-step.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only exact organ identity audit for an existing E-step.")
    parser.add_argument("--estep", default=str(ROOT / "outputs/round1/estep"))
    parser.add_argument("--taxonomy", default=str(ROOT / "configs/organ_taxonomy.json"))
    parser.add_argument("--training-manifest", default=str(ROOT / "outputs/round1/estep/training_manifest.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs/round1/estep/organ_identity_contamination_report.json"))
    args = parser.parse_args()
    report = audit(Path(args.estep).resolve(), Path(args.taxonomy).resolve(), Path(args.training_manifest).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sanitized = output.with_name(output.stem + ".sanitized_training_manifest.json")
    sanitized.write_text(json.dumps(report["sanitized_manifest_rows"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "num_selection_rows", "num_valid", "num_identity_mismatches", "num_manifest_excluded", "num_manifest_legacy_unverified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
