from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ALLOWED_RENAME_STATUS = {"confirmed", "pending_review", "rejected"}
ALLOWED_GAP_RESOLUTION = {"rename", "generate", "manual_review"}
ALLOWED_GAP_STATUS = {"confirmed", "pending_review", "rejected"}
NIFTI_SUFFIX = ".nii.gz"


class DeliveryError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def strip_nii_suffix(name: str) -> str:
    if name.endswith(NIFTI_SUFFIX):
        return name[: -len(NIFTI_SUFFIX)]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def safe_label_name(name: str) -> str:
    value = normalize_name(strip_nii_suffix(name))
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise DeliveryError(f"Invalid label name: {name!r}")
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(k).strip(): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
            if any(str(v or "").strip() for v in row.values())
        ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_taxonomy_names(path: Path, *, expected_count: int = 373) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("target_organs"), list):
        names = [safe_label_name(str(x)) for x in doc["target_organs"]]
    elif isinstance(doc, dict) and isinstance(doc.get("organs"), dict):
        names = [safe_label_name(str(x)) for x in doc["organs"].keys()]
    elif isinstance(doc, dict) and isinstance(doc.get("organ_to_id"), dict):
        names = [safe_label_name(str(x)) for x in doc["organ_to_id"].keys()]
    elif isinstance(doc, list):
        names = [safe_label_name(str(x)) for x in doc]
    else:
        raise DeliveryError(f"Unsupported taxonomy format: {path}")
    unique = list(dict.fromkeys(names))
    if len(unique) != expected_count:
        raise DeliveryError(f"Taxonomy must contain {expected_count} unique names, got {len(unique)}: {path}")
    return unique


def discover_mask_dirs(data_root: Path) -> list[tuple[str, Path, Path]]:
    """Return (case_id, case_root, mask_dir) for directories containing NIfTI masks."""
    root = data_root.resolve()
    out: list[tuple[str, Path, Path]] = []
    if not root.exists():
        raise DeliveryError(f"data-root does not exist: {root}")
    for seg in sorted(root.rglob("*")):
        if not seg.is_dir():
            continue
        if not any(p.is_file() and p.name.endswith(NIFTI_SUFFIX) for p in seg.iterdir()):
            continue
        case_root = seg.parent if seg.name == "segmentations" else seg
        case_id = case_root.name
        out.append((case_id, case_root, seg))
    if root.is_dir() and any(p.is_file() and p.name.endswith(NIFTI_SUFFIX) for p in root.iterdir()):
        out.insert(0, (root.name, root, root))
    seen: set[Path] = set()
    unique = []
    for item in out:
        if item[2] not in seen:
            unique.append(item)
            seen.add(item[2])
    return unique


def candidate_score_name(source: str, target: str) -> tuple[int, str]:
    s = normalize_name(source)
    t = normalize_name(target)
    if s == t:
        return 100, "exact_after_normalization"
    variants = {s, s.replace("vena", "vein"), s.replace("vein", "vena")}
    stokens = s.split("_")
    ttokens = t.split("_")
    if t in variants:
        return 95, "vena_vein"
    if sorted(stokens) == sorted(ttokens):
        return 90, "token_order"
    if {x for x in stokens if x not in {"left", "right"}} == {x for x in ttokens if x not in {"left", "right"}}:
        return 85, "left_right_position"
    ratio = difflib.SequenceMatcher(None, s, t).ratio()
    if ratio >= 0.82:
        return int(ratio * 80), "spelling_similarity"
    return 0, ""


def audit_label_names(data_root: Path, taxonomy: Path, output_dir: Path, *, max_candidates: int = 5) -> dict[str, Any]:
    targets = load_taxonomy_names(taxonomy)
    target_set = set(targets)
    inventory: dict[str, set[str]] = {}
    for case_id, _case_root, mask_dir in discover_mask_dirs(data_root):
        for mask in sorted(mask_dir.glob(f"*{NIFTI_SUFFIX}")):
            inventory.setdefault(safe_label_name(mask.name), set()).add(case_id)
    label_rows = [
        {"label_name": name, "case_count": len(cases), "example_case_id": sorted(cases)[0]}
        for name, cases in sorted(inventory.items())
    ]
    exact = sorted(set(inventory) & target_set)
    unmatched_targets = sorted(target_set - set(inventory))
    extra = sorted(set(inventory) - target_set)
    candidate_rows: list[dict[str, Any]] = []
    for source in extra:
        scored = []
        for target in unmatched_targets:
            score, reason = candidate_score_name(source, target)
            if score:
                scored.append((score, reason, target))
        for score, reason, target in sorted(scored, reverse=True)[:max_candidates]:
            candidate_rows.append({"source_name": source, "candidate_target_name": target, "score": score, "reason": reason})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "label_inventory.csv", label_rows, ["label_name", "case_count", "example_case_id"])
    write_csv(output_dir / "exact_matches.csv", [{"label_name": x} for x in exact], ["label_name"])
    write_csv(output_dir / "unmatched_target_labels.csv", [{"target_name": x} for x in unmatched_targets], ["target_name"])
    write_csv(output_dir / "unmatched_dataset_labels.csv", [{"source_name": x, "case_count": len(inventory[x])} for x in extra], ["source_name", "case_count"])
    write_csv(output_dir / "candidate_matches.csv", candidate_rows, ["source_name", "candidate_target_name", "score", "reason"])
    summary = {
        "status": "success",
        "data_root": str(data_root),
        "taxonomy": str(taxonomy),
        "taxonomy_count": len(targets),
        "cases_scanned": len({c for cases in inventory.values() for c in cases}),
        "dataset_label_count": len(inventory),
        "exact_match_count": len(exact),
        "unmatched_target_count": len(unmatched_targets),
        "unmatched_dataset_label_count": len(extra),
        "candidate_count": len(candidate_rows),
    }
    write_json(output_dir / "audit_summary.json", summary)
    return summary


@dataclass(frozen=True)
class RenameMapping:
    source_name: str
    target_name: str
    status: str
    reason: str
    notes: str


def read_rename_mapping(path: Path) -> list[RenameMapping]:
    rows = []
    for row in read_csv_rows(path):
        status = row.get("status", "")
        if status not in ALLOWED_RENAME_STATUS:
            raise DeliveryError(f"Invalid mapping status {status!r}; allowed={sorted(ALLOWED_RENAME_STATUS)}")
        rows.append(
            RenameMapping(
                safe_label_name(row.get("source_name", "")),
                safe_label_name(row.get("target_name", "")),
                status,
                row.get("reason", ""),
                row.get("notes", ""),
            )
        )
    return rows


def _side(name: str) -> str | None:
    parts = set(normalize_name(name).split("_"))
    left = "left" in parts
    right = "right" in parts
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def validate_rename_mapping(path: Path, taxonomy: Path, report: Path | None = None) -> dict[str, Any]:
    targets = set(load_taxonomy_names(taxonomy))
    rows = read_rename_mapping(path)
    errors: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str]] = set()
    source_to_targets: dict[str, set[str]] = {}
    for idx, row in enumerate(rows, start=2):
        key = (row.source_name, row.target_name, row.status)
        if key in seen_rows:
            errors.append({"row": idx, "type": "duplicate_row", "source_name": row.source_name, "target_name": row.target_name})
        seen_rows.add(key)
        if row.target_name not in targets:
            errors.append({"row": idx, "type": "target_not_in_taxonomy", "target_name": row.target_name})
        if row.source_name == row.target_name:
            errors.append({"row": idx, "type": "source_equals_target", "source_name": row.source_name})
        if _side(row.source_name) and _side(row.target_name) and _side(row.source_name) != _side(row.target_name):
            errors.append({"row": idx, "type": "left_right_conflict", "source_name": row.source_name, "target_name": row.target_name})
        if row.status == "confirmed":
            source_to_targets.setdefault(row.source_name, set()).add(row.target_name)
    for source, mapped_targets in sorted(source_to_targets.items()):
        if len(mapped_targets) > 1:
            errors.append({"type": "one_to_many_source_mapping", "source_name": source, "target_names": sorted(mapped_targets)})
    summary = {"status": "failed" if errors else "success", "mapping_file": str(path), "taxonomy": str(taxonomy), "rows": len(rows), "errors": errors}
    if report:
        write_json(report, summary)
    if errors:
        raise DeliveryError(f"Rename mapping validation failed with {len(errors)} error(s)")
    return summary


def apply_rename(data_root: Path, mapping_file: Path, taxonomy: Path, report: Path, *, apply: bool = False) -> dict[str, Any]:
    validate_rename_mapping(mapping_file, taxonomy)
    mappings = [m for m in read_rename_mapping(mapping_file) if m.status == "confirmed"]
    mode = "apply" if apply else "dry_run"
    report_rows: list[dict[str, Any]] = []
    case_count = 0
    for case_id, _case_root, mask_dir in discover_mask_dirs(data_root):
        case_count += 1
        present = {safe_label_name(p.name): p for p in mask_dir.glob(f"*{NIFTI_SUFFIX}")}
        target_hits: dict[str, list[str]] = {}
        for m in mappings:
            if m.source_name in present:
                target_hits.setdefault(m.target_name, []).append(m.source_name)
        blocked_targets = {target for target, sources in target_hits.items() if len(sources) > 1}
        for m in mappings:
            src = mask_dir / f"{m.source_name}{NIFTI_SUFFIX}"
            dst = mask_dir / f"{m.target_name}{NIFTI_SUFFIX}"
            row = {
                "case_id": case_id,
                "source_name": m.source_name,
                "target_name": m.target_name,
                "source_path": str(src),
                "target_path": str(dst),
                "mode": mode,
                "status": "",
                "reason": "",
            }
            if m.target_name in blocked_targets:
                row.update({"status": "conflict", "reason": "multiple_sources_to_same_target_in_case"})
            elif src.exists() and dst.exists():
                row.update({"status": "conflict", "reason": "source_and_target_exist"})
            elif not src.exists() and dst.exists():
                row.update({"status": "already_applied", "reason": "source_missing_target_exists"})
            elif not src.exists() and not dst.exists():
                row.update({"status": "source_missing", "reason": "source_and_target_missing"})
            else:
                row.update({"status": "renamed" if apply else "would_rename", "reason": "confirmed_mapping"})
                if apply:
                    src.rename(dst)
            report_rows.append(row)
    fields = ["case_id", "source_name", "target_name", "source_path", "target_path", "mode", "status", "reason"]
    write_csv(report, report_rows, fields)
    status_counts: dict[str, int] = {}
    for row in report_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {"status": "success", "mode": mode, "cases_scanned": case_count, "operations": len(report_rows), "status_counts": status_counts, "report": str(report)}


def read_gap_resolution(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    for idx, row in enumerate(rows, start=2):
        if row.get("resolution") not in ALLOWED_GAP_RESOLUTION:
            raise DeliveryError(f"Invalid resolution at row {idx}: {row.get('resolution')!r}")
        if row.get("status") not in ALLOWED_GAP_STATUS:
            raise DeliveryError(f"Invalid status at row {idx}: {row.get('status')!r}")
        row["target_name"] = safe_label_name(row.get("target_name", ""))
        if row.get("dataset_name"):
            row["dataset_name"] = safe_label_name(row["dataset_name"])
    return rows


def confirmed_generate_organs(gap_file: Path, taxonomy: Path, rename_mapping: Path | None = None) -> list[str]:
    targets = set(load_taxonomy_names(taxonomy))
    rows = read_gap_resolution(gap_file)
    generate = [r["target_name"] for r in rows if r.get("resolution") == "generate" and r.get("status") == "confirmed"]
    bad = sorted(set(generate) - targets)
    if bad:
        raise DeliveryError(f"Generate organs not in taxonomy: {bad[:20]}")
    rename_targets = set()
    if rename_mapping and rename_mapping.exists():
        rename_targets = {m.target_name for m in read_rename_mapping(rename_mapping) if m.status == "confirmed"}
    overlap = sorted(set(generate) & rename_targets)
    if overlap:
        raise DeliveryError(f"Generate targets overlap confirmed rename targets: {overlap[:20]}")
    return list(dict.fromkeys(generate))


def validate_gap_resolution(gap_file: Path, taxonomy: Path, rename_mapping: Path | None, report: Path | None = None) -> dict[str, Any]:
    rows = read_gap_resolution(gap_file)
    generate = confirmed_generate_organs(gap_file, taxonomy, rename_mapping)
    summary = {"status": "success", "gap_file": str(gap_file), "rows": len(rows), "confirmed_generate_count": len(generate), "confirmed_generate_organs": generate}
    if report:
        write_json(report, summary)
    return summary


def validate_100case_manifest(path: Path, *, check_exists: bool = True, report: Path | None = None) -> dict[str, Any]:
    rows = read_csv_rows(path)
    errors = []
    required = {"index", "case_id", "reference_mask_dir"}
    columns = set(rows[0]) if rows else set()
    missing_cols = sorted(required - columns)
    if missing_cols:
        errors.append({"type": "missing_columns", "columns": missing_cols})
    if rows and not ({"image_path", "ct_path"} & columns):
        errors.append({"type": "missing_columns", "columns": ["image_path_or_ct_path"]})
    if len(rows) != 100:
        errors.append({"type": "row_count", "actual": len(rows), "expected": 100})
    indices = []
    case_ids = []
    for i, row in enumerate(rows):
        try:
            indices.append(int(row.get("index", "")))
        except ValueError:
            errors.append({"row": i + 2, "type": "invalid_index", "index": row.get("index")})
        case_ids.append(row.get("case_id", ""))
        if check_exists:
            image_path = row.get("image_path") or row.get("ct_path")
            if image_path and not Path(image_path).exists():
                errors.append({"row": i + 2, "type": "image_path_missing", "path": image_path})
            if row.get("reference_mask_dir") and not Path(row["reference_mask_dir"]).exists():
                errors.append({"row": i + 2, "type": "reference_mask_dir_missing", "path": row["reference_mask_dir"]})
    if indices and indices != list(range(100)):
        errors.append({"type": "index_sequence", "actual": indices[:105], "expected": "0-99"})
    duplicates = sorted({x for x in case_ids if case_ids.count(x) > 1})
    if duplicates:
        errors.append({"type": "duplicate_case_id", "case_ids": duplicates})
    summary = {"status": "failed" if errors else "success", "manifest": str(path), "rows": len(rows), "errors": errors}
    if report:
        write_json(report, summary)
    if errors:
        raise DeliveryError(f"100-case manifest validation failed with {len(errors)} error(s)")
    return summary


def run_preflight(
    manifest: Path,
    gap_file: Path,
    taxonomy: Path,
    output_dir: Path,
    *,
    rename_mapping: Path | None = None,
    teacher_entry: Path | None = None,
    registry: Path | None = None,
    output_root: Path | None = None,
    image_root: Path | None = None,
    mask_root: Path | None = None,
    code_root: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    organs = confirmed_generate_organs(gap_file, taxonomy, rename_mapping)
    cases = read_csv_rows(manifest)
    case_rows = []
    conflict_rows = []
    for row in cases:
        mask_dir = Path(row.get("reference_mask_dir", ""))
        existing = [organ for organ in organs if (mask_dir / f"{organ}{NIFTI_SUFFIX}").exists()]
        if existing:
            conflict_rows.extend({"case_id": row.get("case_id"), "target_name": organ, "path": str(mask_dir / f"{organ}{NIFTI_SUFFIX}"), "reason": "official_mask_exists"} for organ in existing)
        case_rows.append({
            "index": row.get("index"),
            "case_id": row.get("case_id"),
            "image_path": row.get("image_path") or row.get("ct_path"),
            "reference_mask_dir": row.get("reference_mask_dir"),
            "image_exists": Path(row.get("image_path") or row.get("ct_path") or "").is_file(),
            "reference_mask_dir_exists": mask_dir.is_dir(),
            "existing_target_mask_count": len(existing),
        })
    organ_rows = [{"target_name": organ, "in_taxonomy": True, "source": "confirmed_generate"} for organ in organs]
    write_csv(output_dir / "preflight_cases.csv", case_rows, ["index", "case_id", "image_path", "reference_mask_dir", "image_exists", "reference_mask_dir_exists", "existing_target_mask_count"])
    write_csv(output_dir / "preflight_organs.csv", organ_rows, ["target_name", "in_taxonomy", "source"])
    write_csv(output_dir / "preflight_conflicts.csv", conflict_rows, ["case_id", "target_name", "path", "reason"])
    errors = []
    try:
        validate_100case_manifest(manifest, check_exists=True)
    except DeliveryError as exc:
        errors.append(str(exc))
    for label, path in (("teacher_entry", teacher_entry), ("registry", registry)):
        if path and not path.exists():
            errors.append(f"{label} does not exist: {path}")
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)
        if not os.access(output_root, os.W_OK):
            errors.append(f"output_root not writable: {output_root}")
        forbidden = [p for p in (image_root, mask_root, code_root) if p]
        resolved_out = output_root.resolve()
        for root in forbidden:
            try:
                resolved_out.relative_to(root.resolve())
                errors.append(f"output_root is inside forbidden root: {root}")
            except ValueError:
                pass
    if conflict_rows:
        errors.append("official target masks already exist; default policy blocks inference")
    summary = {
        "status": "failed" if errors else "success",
        "manifest": str(manifest),
        "case_count": len(cases),
        "organ_count": len(organs),
        "conflict_count": len(conflict_rows),
        "errors": errors,
    }
    write_json(output_dir / "preflight_summary.json", summary)
    if errors:
        raise DeliveryError("; ".join(errors))
    return summary


def load_manifest_case(manifest: Path, case_index: int) -> dict[str, str]:
    rows = read_csv_rows(manifest)
    if case_index < 0 or case_index >= len(rows):
        raise DeliveryError(f"case index out of range: {case_index}")
    return rows[case_index]


def case_image_path(case: dict[str, str]) -> str:
    value = case.get("image_path") or case.get("ct_path")
    if not value:
        raise DeliveryError(f"manifest case {case.get('case_id', '<unknown>')} is missing image_path/ct_path")
    return value


def locate_generated_mask_dir(case_run_dir: Path, case_id: str) -> Path | None:
    candidates = [
        case_run_dir / "raw" / "annotation_versions" / case_id / "updated",
        case_run_dir / "raw" / "standard_dataset" / case_id / "segmentations",
        case_run_dir / "raw" / "cases" / case_id / "selected_after_candidate_shapekit" / case_id / "segmentations",
        case_run_dir / "raw" / "cases" / case_id / "selected_after_candidate_shapekit" / "segmentations",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob(f"*{NIFTI_SUFFIX}")):
            return path
    for path in (case_run_dir / "raw").rglob("segmentations"):
        if path.is_dir() and any(path.glob(f"*{NIFTI_SUFFIX}")):
            return path
    return None


def run_teacher_case(
    manifest: Path,
    gap_file: Path,
    taxonomy: Path,
    run_dir: Path,
    *,
    case_index: int,
    code_root: Path,
    python: str = sys.executable,
    teacher_entry: Path | None = None,
    registry: Path | None = None,
    models: str = "epai_20250421,vsmtrans",
    target_config: Path | None = None,
    timeout_sec: int = 1800,
    device: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    teacher_entry = teacher_entry or code_root / "run_medai_cli.py"
    target_config = target_config or taxonomy
    registry = registry or code_root / "configs" / "model_registry.yaml"
    organs = confirmed_generate_organs(gap_file, taxonomy)
    case = load_manifest_case(manifest, case_index)
    case_id = case["case_id"]
    case_dir = run_dir / "cases" / case_id
    raw_dir = case_dir / "raw"
    log_dir = case_dir / "logs"
    status_path = case_dir / "status.json"
    existing_validated = status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "validated"
    if existing_validated and locate_generated_mask_dir(case_dir, case_id):
        status = {"status": "skipped_validated", "case_id": case_id, "generated_file_count": len(list(locate_generated_mask_dir(case_dir, case_id).glob(f'*{NIFTI_SUFFIX}')))}
        write_json(status_path, status)
        return status
    case_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    one_case_manifest = case_dir / "case_manifest.csv"
    write_csv(one_case_manifest, [{
        "case_id": case_id,
        "ct_path": case_image_path(case),
        "annotation_folder": case["reference_mask_dir"],
    }], ["case_id", "ct_path", "annotation_folder"])
    command = [
        python, str(teacher_entry), "--json", "run-loop",
        "--case-list", str(one_case_manifest),
        "--models", models,
        "--organs", ",".join(organs),
        "--target-config", str(target_config),
        "--registry", str(registry),
        "--output", str(raw_dir),
        "--timeout-sec", str(timeout_sec),
        "--strict-delivery-targets",
        "--log-file", str(log_dir / "run_loop.log"),
    ]
    if device:
        command.extend(["--device", device])
    if dry_run:
        command.append("--dry-run")
    (case_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = utc_now()
    write_json(status_path, {"status": "running", "case_id": case_id, "command": command, "image_path": case_image_path(case), "organ_list": organs, "output_directory": str(raw_dir), "python_executable": python, "started_at": started, "log_file": str(log_dir / "run_loop.log")})
    proc = subprocess.run(command, cwd=str(code_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (log_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (log_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    generated_dir = locate_generated_mask_dir(case_dir, case_id)
    generated_files = sorted(generated_dir.glob(f"*{NIFTI_SUFFIX}")) if generated_dir else []
    expected_files = [p for p in generated_files if safe_label_name(p.name) in set(organs)]
    missing_organs = sorted(set(organs) - {safe_label_name(p.name) for p in expected_files})
    run_summary_path = raw_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    strict_failures = run_summary.get("strict_delivery_failures", []) if isinstance(run_summary, dict) else []
    status_name = "inference_succeeded" if proc.returncode == 0 and not missing_organs and not strict_failures else "failed"
    if dry_run and proc.returncode == 0:
        status_name = "pending"
    status = {
        "status": status_name,
        "case_id": case_id,
        "command": command,
        "image_path": case_image_path(case),
        "organ_list": organs,
        "output_directory": str(raw_dir),
        "python_executable": python,
        "started_at": started,
        "ended_at": utc_now(),
        "return_code": proc.returncode,
        "generated_file_count": len(expected_files),
        "expected_file_count": len(organs),
        "missing_expected_organs": missing_organs,
        "run_summary": str(run_summary_path) if run_summary_path.exists() else "",
        "strict_delivery_failures": strict_failures,
        "log_file": str(log_dir / "run_loop.log"),
        "validation_status": "not_run",
        "failure_reason": "" if status_name != "failed" else (
            "strict_delivery_failure" if strict_failures
            else "expected_mask_missing" if missing_organs
            else "no_real_expected_nifti_output" if proc.returncode == 0
            else "teacher_command_failed"
        ),
    }
    write_json(status_path, status)
    if proc.returncode != 0:
        raise DeliveryError(f"Teacher command failed for {case_id}: {(proc.stderr or proc.stdout)[-2000:]}")
    if status_name == "failed":
        raise DeliveryError(f"Teacher command produced no expected NIfTI output for {case_id}")
    return status


def _load_nifti(path: Path):
    try:
        import nibabel as nib  # type: ignore
        return "nibabel", nib.load(str(path))
    except ImportError as exc:
        raise DeliveryError("nibabel is required for NIfTI validation") from exc


def validate_generated_masks(run_dir: Path, manifest: Path, gap_file: Path, taxonomy: Path, output_dir: Path) -> dict[str, Any]:
    organs = set(confirmed_generate_organs(gap_file, taxonomy))
    rows = []
    conflicts = []
    for case in read_csv_rows(manifest):
        case_id = case["case_id"]
        ct_path = Path(case_image_path(case))
        ref_dir = Path(case["reference_mask_dir"])
        mask_dir = locate_generated_mask_dir(run_dir / "cases" / case_id, case_id)
        try:
            _tool, ct_img = _load_nifti(ct_path)
            ct_shape = tuple(ct_img.shape)
            ct_affine = ct_img.affine
            ct_spacing = tuple(float(x) for x in ct_img.header.get_zooms()[: len(ct_shape)])
        except Exception as exc:
            for organ in organs:
                rows.append({"case_id": case_id, "target_name": organ, "status": "failed", "reason": f"ct_unreadable:{exc}"})
            continue
        for organ in sorted(organs):
            path = mask_dir / f"{organ}{NIFTI_SUFFIX}" if mask_dir else Path("")
            row: dict[str, Any] = {"case_id": case_id, "target_name": organ, "mask_path": str(path), "status": "failed", "reason": ""}
            if (ref_dir / f"{organ}{NIFTI_SUFFIX}").exists():
                conflicts.append({"case_id": case_id, "target_name": organ, "path": str(ref_dir / f"{organ}{NIFTI_SUFFIX}"), "reason": "official_mask_exists"})
                row["reason"] = "official_mask_exists"
            elif not path.is_file():
                row["reason"] = "mask_missing"
            else:
                try:
                    _tool, img = _load_nifti(path)
                    data = img.get_fdata(dtype="float32")
                    positives = int((data > 0).sum())
                    total = int(data.size)
                    spacing = tuple(float(x) for x in img.header.get_zooms()[: len(img.shape)])
                    shape_ok = tuple(img.shape) == ct_shape
                    spacing_ok = spacing == ct_spacing
                    try:
                        import numpy as np  # type: ignore
                        affine_ok = bool(np.allclose(img.affine, ct_affine, atol=1e-3))
                    except Exception:
                        affine_ok = img.affine.tolist() == ct_affine.tolist()
                    row.update({
                        "shape": "x".join(map(str, img.shape)),
                        "ct_shape": "x".join(map(str, ct_shape)),
                        "shape_match": shape_ok,
                        "spacing": "|".join(map(str, spacing)),
                        "ct_spacing": "|".join(map(str, ct_spacing)),
                        "spacing_match": spacing_ok,
                        "affine_match": affine_ok,
                        "dtype": str(img.get_data_dtype()),
                        "min": float(data.min()) if total else "",
                        "max": float(data.max()) if total else "",
                        "positive_voxels": positives,
                        "total_voxels": total,
                        "occupancy_ratio": positives / total if total else 0,
                        "is_all_zero": positives == 0,
                        "near_full_image": (positives / total) > 0.85 if total else False,
                        "left_right_warning": _side(organ) or "",
                    })
                    if shape_ok and spacing_ok and affine_ok:
                        row.update({"status": "passed", "reason": "validated"})
                    else:
                        row["reason"] = "geometry_mismatch"
                except Exception as exc:
                    row["reason"] = f"mask_unreadable:{exc}"
            rows.append(row)
    fields = ["case_id", "target_name", "mask_path", "status", "reason", "shape", "ct_shape", "shape_match", "spacing", "ct_spacing", "spacing_match", "affine_match", "dtype", "min", "max", "positive_voxels", "total_voxels", "occupancy_ratio", "is_all_zero", "near_full_image", "left_right_warning"]
    write_csv(output_dir / "validation_report.csv", rows, fields)
    write_csv(output_dir / "copy_conflicts.csv", conflicts, ["case_id", "target_name", "path", "reason"])
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    for case_id, case_rows in by_case.items():
        status_path = run_dir / "cases" / case_id / "status.json"
        if status_path.exists() and all(r["status"] == "passed" for r in case_rows):
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "validated"
            status["validation_status"] = "passed"
            status["generated_file_count"] = len(case_rows)
            write_json(status_path, status)
    summary = {"status": "success" if rows and all(r["status"] == "passed" for r in rows) and not conflicts else "failed", "rows": len(rows), "passed": sum(1 for r in rows if r["status"] == "passed"), "failed": sum(1 for r in rows if r["status"] != "passed"), "conflicts": len(conflicts)}
    write_json(output_dir / "validation_summary.json", summary)
    return summary


def summarize_run(run_dir: Path, manifest: Path, output_dir: Path) -> dict[str, Any]:
    cases = read_csv_rows(manifest)
    rows = []
    for row in cases:
        case_id = row["case_id"]
        index = row["index"]
        status_path = run_dir / "cases" / case_id / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = status.get("status", "unknown")
            reason = status.get("failure_reason", "")
        else:
            state = "incomplete"
            reason = "missing_status_json"
        rows.append({"index": index, "case_id": case_id, "status": state, "reason": reason, "status_path": str(status_path)})
    failed = [r for r in rows if r["status"] in {"failed", "validation_failed"}]
    incomplete = [r for r in rows if r["status"] in {"pending", "running", "incomplete", "inference_succeeded"}]
    validated = [r for r in rows if r["status"] in {"validated", "skipped_validated"}]
    write_csv(output_dir / "run_summary.csv", rows, ["index", "case_id", "status", "reason", "status_path"])
    write_csv(output_dir / "validated_cases.csv", validated, ["index", "case_id", "status", "reason", "status_path"])
    write_csv(output_dir / "failed_cases.csv", failed, ["index", "case_id", "status", "reason", "status_path"])
    write_csv(output_dir / "incomplete_cases.csv", incomplete, ["index", "case_id", "status", "reason", "status_path"])
    failed_indices = sorted(int(r["index"]) for r in [*failed, *incomplete] if str(r["index"]).isdigit())
    spec = compact_indices(failed_indices)
    (output_dir / "failed_array_indices.txt").write_text("\n".join(map(str, failed_indices)) + ("\n" if failed_indices else ""), encoding="utf-8")
    (output_dir / "failed_array_spec.txt").write_text(spec + ("\n" if spec else ""), encoding="utf-8")
    summary = {"status": "success", "total_cases": len(rows), "validated": len(validated), "failed": len(failed), "incomplete": len(incomplete), "failed_array_spec": spec}
    write_json(output_dir / "summary.json", summary)
    return summary


def compact_indices(indices: Iterable[int]) -> str:
    values = sorted(set(indices))
    ranges = []
    i = 0
    while i < len(values):
        start = end = values[i]
        while i + 1 < len(values) and values[i + 1] == end + 1:
            i += 1
            end = values[i]
        ranges.append(str(start) if start == end else f"{start}-{end}")
        i += 1
    return ",".join(ranges)


def package_dataset_overlay(
    run_dir: Path,
    manifest: Path,
    validation_report: Path,
    delivery_dir: Path,
    *,
    fail_on_conflict: bool = True,
) -> dict[str, Any]:
    rows = [r for r in read_csv_rows(validation_report) if r.get("status") == "passed"]
    manifests = {r["case_id"]: r for r in read_csv_rows(manifest)}
    staging = delivery_dir.with_name(delivery_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    copy_rows = []
    conflicts = []
    for row in rows:
        case_id = row["case_id"]
        case = manifests[case_id]
        ref_dir = Path(case["reference_mask_dir"]).resolve()
        case_root = ref_dir.parent if ref_dir.name == "segmentations" else ref_dir
        rel_case = Path(case_id)
        rel_mask_subdir = Path("segmentations")
        try:
            rel_case = case_root.relative_to(ref_dir.parents[1])
            rel_mask_subdir = ref_dir.relative_to(case_root)
        except Exception:
            pass
        src = Path(row["mask_path"])
        dst = staging / rel_case / rel_mask_subdir / src.name
        if dst.exists():
            conflicts.append({"case_id": case_id, "target_name": row["target_name"], "path": str(dst), "reason": "delivery_target_exists"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.is_symlink():
            raise DeliveryError(f"Packaging produced a symlink unexpectedly: {dst}")
        copy_rows.append({"case_id": case_id, "target_name": row["target_name"], "source_path": str(src), "delivery_path": str(dst), "sha256": sha256_file(dst)})
    if conflicts and fail_on_conflict:
        write_csv(staging / "copy_conflicts.csv", conflicts, ["case_id", "target_name", "path", "reason"])
        raise DeliveryError(f"Packaging blocked by {len(conflicts)} conflict(s)")
    write_csv(staging / "manifest.csv", copy_rows, ["case_id", "target_name", "source_path", "delivery_path", "sha256"])
    shutil.copy2(validation_report, staging / "validation_report.csv")
    write_csv(staging / "copy_conflicts.csv", conflicts, ["case_id", "target_name", "path", "reason"])
    failed = []
    run_summary = run_dir / "summaries" / "failed_cases.csv"
    if run_summary.exists():
        shutil.copy2(run_summary, staging / "failed_cases.csv")
    else:
        write_csv(staging / "failed_cases.csv", failed, ["index", "case_id", "status", "reason", "status_path"])
    if delivery_dir.exists():
        raise DeliveryError(f"Refusing to overwrite existing delivery dir: {delivery_dir}")
    staging.rename(delivery_dir)
    return {"status": "success", "delivery_dir": str(delivery_dir), "copied_masks": len(copy_rows), "conflicts": len(conflicts)}
