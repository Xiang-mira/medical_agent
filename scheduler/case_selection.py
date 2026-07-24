from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml
from nibabel.orientations import aff2axcodes

from .config import resolve_path
from .utils import ROOT, SchedulerError, ensure_not_raw_data_write_path, normalize_mask_stem, sha256_file, utc_now, write_json_atomic


STRICT_INPUT_COLUMNS = ("case_id", "ct_path")
EVAL_COLUMNS = ("case_id", "ct_path", "mask_dir")
CASE_SELECTION_STAGES = [
    ("inventory_cpu", []),
    ("header_scan_array", ["inventory_cpu"]),
    ("merge_header_metrics", ["header_scan_array"]),
    ("candidate_prefilter", ["merge_header_metrics"]),
    ("deep_mask_audit_array", ["candidate_prefilter"]),
    ("merge_deep_metrics", ["deep_mask_audit_array"]),
    ("select_and_split", ["merge_deep_metrics"]),
    ("validate_manifests", ["select_and_split"]),
    ("final_selection_report", ["validate_manifests"]),
]
CASE_SELECTION_STAGE_COMMANDS = {
    "inventory_cpu": "inventory",
    "header_scan_array": "scan-headers",
    "merge_header_metrics": "prefilter",
    "candidate_prefilter": "prefilter",
    "deep_mask_audit_array": "deep-audit",
    "merge_deep_metrics": "select",
    "select_and_split": "select",
    "validate_manifests": "validate",
    "final_selection_report": "validate",
}


@dataclass(frozen=True)
class CaseSelectionConfig:
    path: Path
    data: dict[str, Any]

    @property
    def paths(self) -> dict[str, Any]:
        return self.data.get("paths", {})

    @property
    def candidate_pool(self) -> dict[str, Any]:
        return self.data.get("candidate_pool", {})

    @property
    def scoring(self) -> dict[str, float]:
        return self.data.get("scoring", {})


def load_case_selection_config(path: str | Path = "configs/abdomenatlaspro_case_selection.yaml") -> CaseSelectionConfig:
    resolved = resolve_path(path)
    assert resolved is not None
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SchedulerError(f"Case selection config must be a YAML mapping: {resolved}")
    return CaseSelectionConfig(path=resolved, data=data)


def _csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if any((v or "").strip() for v in row.values())]


def _percentiles(rows: list[dict[str, Any]], key: str, out_key: str) -> None:
    values = sorted(float(r.get(key) or 0.0) for r in rows)
    if not values:
        return
    denom = max(len(values) - 1, 1)
    for row in rows:
        value = float(row.get(key) or 0.0)
        idx = values.index(value)
        row[out_key] = idx / denom


def _short_affine(affine: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 5) for x in row] for row in affine.tolist()]


def scan_ct_header(ct_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ct_path": str(ct_path),
        "header_status": "failed",
        "header_error": "",
    }
    try:
        img = nib.load(str(ct_path))
        shape = tuple(int(x) for x in img.shape[:3])
        zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
        orientation = "".join(str(x) for x in aff2axcodes(img.affine))
        si_axis = next((i for i, code in enumerate(orientation) if code in {"S", "I"}), None)
        if si_axis is None:
            raise SchedulerError(f"No superior-inferior axis in orientation {orientation!r}")
        row.update(
            {
                "array_shape_x": shape[0],
                "array_shape_y": shape[1],
                "array_shape_z": shape[2],
                "spacing_x": zooms[0],
                "spacing_y": zooms[1],
                "spacing_z": zooms[2],
                "orientation": orientation,
                "si_axis_index": si_axis,
                "si_spacing_mm": zooms[si_axis],
                "physical_si_extent_mm": shape[si_axis] * zooms[si_axis],
                "affine_summary": _short_affine(np.asarray(img.affine)),
                "header_status": "success",
                "header_error": "",
            }
        )
    except Exception as exc:
        row["header_error"] = str(exc)
    return row


def discover_inventory(image_root: Path, mask_root: Path) -> list[dict[str, Any]]:
    if not image_root.exists() or not mask_root.exists():
        raise SchedulerError(f"Raw roots must exist: image_root={image_root} mask_root={mask_root}")
    rows = []
    for case_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        case_id = case_dir.name
        ct = case_dir / "ct.nii.gz"
        seg = mask_root / case_id / "segmentations"
        if not ct.is_file() or not seg.is_dir():
            continue
        masks = sorted(p for p in seg.iterdir() if p.is_file() and p.name.endswith(".nii.gz"))
        rows.append(
            {
                "case_id": case_id,
                "ct_path": str(ct),
                "mask_dir": str(seg),
                "ct_file_size": ct.stat().st_size,
                "mask_file_count": len(masks),
            }
        )
    return rows


def _mask_lookup(mask_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*.nii.gz")):
        out.setdefault(normalize_mask_stem(path.name), path)
    return out


def _mapping_sources(item: dict[str, Any]) -> list[str]:
    return [str(x) for x in (item.get("source_masks") or item.get("aliases") or [])]


def deep_mask_audit(row: dict[str, Any], mapping_items: list[dict[str, Any]], *, affine_atol: float = 1e-3) -> dict[str, Any]:
    ct_img = nib.load(str(row["ct_path"]))
    ct_shape = tuple(int(x) for x in ct_img.shape[:3])
    ct_affine = np.asarray(ct_img.affine)
    masks = _mask_lookup(Path(str(row["mask_dir"])))
    result = dict(row)
    counters = {
        "readable_mask_count": 0,
        "positive_source_mask_count": 0,
        "all_zero_mask_count": 0,
        "corrupted_mask_count": 0,
        "shape_mismatch_count": 0,
        "affine_mismatch_count": 0,
        "mapped_373_available_count": 0,
        "mapped_373_positive_count": 0,
        "unsupported_mapping_count": 0,
    }
    positive_sources: set[str] = set()
    bad_sources: set[str] = set()
    all_zero_sources: set[str] = set()

    def source_positive(source: str) -> bool:
        if source in positive_sources:
            return True
        if source in all_zero_sources or source in bad_sources or source not in masks:
            return False
        try:
            img = nib.load(str(masks[source]))
            counters["readable_mask_count"] += 1
            if tuple(int(x) for x in img.shape[:3]) != ct_shape:
                counters["shape_mismatch_count"] += 1
                bad_sources.add(source)
                return False
            if not np.allclose(np.asarray(img.affine), ct_affine, atol=affine_atol):
                counters["affine_mismatch_count"] += 1
                bad_sources.add(source)
                return False
            arr = np.asanyarray(img.dataobj)
            if bool(np.any(arr > 0)):
                positive_sources.add(source)
                counters["positive_source_mask_count"] += 1
                return True
            all_zero_sources.add(source)
            counters["all_zero_mask_count"] += 1
            return False
        except Exception:
            counters["corrupted_mask_count"] += 1
            bad_sources.add(source)
            return False

    for item in mapping_items:
        status = str(item.get("mapping_status") or "")
        sources = _mapping_sources(item)
        if status not in {"direct", "alias", "union", "composite"}:
            counters["unsupported_mapping_count"] += 1
            continue
        existing = [s for s in sources if s in masks]
        if existing:
            counters["mapped_373_available_count"] += 1
        if any(source_positive(s) for s in existing):
            counters["mapped_373_positive_count"] += 1
    counters["mapped_373_missing_count"] = max(0, len(mapping_items) - counters["mapped_373_available_count"])
    quality_penalty = counters["corrupted_mask_count"] + counters["shape_mismatch_count"] + counters["affine_mismatch_count"]
    counters["quality_score"] = max(0.0, 1.0 - quality_penalty / max(int(result.get("mask_file_count") or 1), 1))
    counters["quality_pass"] = quality_penalty == 0 and counters["mapped_373_positive_count"] > 0
    counters["exclusion_reason"] = "" if counters["quality_pass"] else "no_positive_compatible_mapped_masks" if counters["mapped_373_positive_count"] == 0 else "mask_quality_failure"
    result.update(counters)
    return result


def build_duplicate_key(row: dict[str, Any]) -> str:
    payload = {
        "ct_file_size": row.get("ct_file_size"),
        "shape": [row.get("array_shape_x"), row.get("array_shape_y"), row.get("array_shape_z")],
        "spacing": [row.get("spacing_x"), row.get("spacing_y"), row.get("spacing_z")],
        "affine": row.get("affine_summary"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def candidate_pool_size(requested_total: int, cfg: CaseSelectionConfig) -> int:
    spec = cfg.candidate_pool
    return min(int(spec.get("maximum", 5000)), max(int(spec.get("minimum", 1000)), requested_total * int(spec.get("multiplier", 20))))


def prefilter_candidates(rows: list[dict[str, Any]], requested_total: int, cfg: CaseSelectionConfig) -> list[dict[str, Any]]:
    size = candidate_pool_size(requested_total, cfg)
    by_extent = sorted(rows, key=lambda r: float(r.get("physical_si_extent_mm") or 0), reverse=True)[:size]
    by_masks = sorted(rows, key=lambda r: int(r.get("mask_file_count") or 0), reverse=True)[:size]
    by_combo = sorted(rows, key=lambda r: (float(r.get("physical_si_extent_mm") or 0), int(r.get("mask_file_count") or 0)), reverse=True)[:size]
    by_case = {r["case_id"]: r for r in [*by_extent, *by_masks, *by_combo] if r.get("header_status") == "success"}
    return list(by_case.values())[: max(size, requested_total)]


def select_and_split(rows: list[dict[str, Any]], train_cases: int, test_cases: int, seed: int, cfg: CaseSelectionConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [dict(r) for r in rows if r.get("quality_pass")]
    if len(valid) < train_cases + test_cases:
        raise SchedulerError(f"Not enough valid candidates: requested={train_cases + test_cases} valid={len(valid)}")
    _percentiles(valid, "mapped_373_positive_count", "mapped_positive_target_percentile")
    _percentiles(valid, "physical_si_extent_mm", "physical_si_extent_percentile")
    _percentiles(valid, "quality_score", "quality_percentile")
    weights = cfg.scoring
    for row in valid:
        row["selection_score"] = (
            float(weights.get("mapped_positive_target_weight", 0.55)) * float(row.get("mapped_positive_target_percentile") or 0)
            + float(weights.get("physical_si_extent_weight", 0.35)) * float(row.get("physical_si_extent_percentile") or 0)
            + float(weights.get("quality_weight", 0.10)) * float(row.get("quality_percentile") or 0)
        )
    valid.sort(key=lambda r: (-float(r["selection_score"]), str(r["case_id"])))
    unique: list[dict[str, Any]] = []
    seen_duplicate_groups: set[str] = set()
    for row in valid:
        key = build_duplicate_key(row)
        row["duplicate_group"] = key
        if key in seen_duplicate_groups:
            continue
        seen_duplicate_groups.add(key)
        unique.append(row)
    backup_count = math.ceil((train_cases + test_cases) * float(cfg.candidate_pool.get("backup_ratio", 0.20)))
    selected = unique[: train_cases + test_cases]
    backups = unique[train_cases + test_cases : train_cases + test_cases + backup_count]
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    rng = random.Random(seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in shuffled:
        if len(train) >= train_cases:
            test.append(row)
        elif len(test) >= test_cases:
            train.append(row)
        else:
            target_train_fraction = train_cases / (train_cases + test_cases)
            if (len(train) + 1) / (len(train) + len(test) + 1) <= target_train_fraction:
                train.append(row)
            else:
                test.append(row)
    return sorted(train, key=lambda r: str(r["case_id"])), sorted(test, key=lambda r: str(r["case_id"])), backups


def validate_selection(train: list[dict[str, Any]], test: list[dict[str, Any]], *, train_cases: int, test_cases: int) -> dict[str, Any]:
    train_ids = {str(r["case_id"]) for r in train}
    test_ids = {str(r["case_id"]) for r in test}
    cross_dups = sorted({str(r.get("duplicate_group")) for r in train} & {str(r.get("duplicate_group")) for r in test})
    status = "success"
    errors = []
    if len(train) != train_cases:
        errors.append(f"TRAIN_COUNT expected {train_cases}, got {len(train)}")
    if len(test) != test_cases:
        errors.append(f"TEST_COUNT expected {test_cases}, got {len(test)}")
    if train_ids & test_ids:
        errors.append(f"TRAIN_TEST_OVERLAP {sorted(train_ids & test_ids)}")
    if cross_dups:
        errors.append(f"DUPLICATE_GROUP_CROSS_SPLIT {cross_dups}")
    if errors:
        status = "failed"
    return {
        "status": status,
        "TRAIN_COUNT": len(train),
        "TEST_COUNT": len(test),
        "TOTAL_COUNT": len(train) + len(test),
        "TRAIN_TEST_OVERLAP": len(train_ids & test_ids),
        "DUPLICATE_GROUP_CROSS_SPLIT": len(cross_dups),
        "errors": errors,
    }


def write_selection_outputs(out_dir: Path, cfg: CaseSelectionConfig, train: list[dict[str, Any]], test: list[dict[str, Any]], backups: list[dict[str, Any]], excluded: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    image_root = resolve_path(cfg.paths.get("image_root"))
    mask_root = resolve_path(cfg.paths.get("mask_root"))
    ensure_not_raw_data_write_path(out_dir, image_root, mask_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_selected = [*train, *test]
    _csv_write(out_dir / "selected_case_metrics.csv", all_selected)
    _csv_write(out_dir / "backup_candidates.csv", backups)
    _csv_write(out_dir / "excluded_cases.csv", excluded)
    _csv_write(out_dir / "duplicate_groups.csv", [{"case_id": r["case_id"], "duplicate_group": r.get("duplicate_group", "")} for r in all_selected])
    _csv_write(out_dir / f"train{len(train)}_input_strict_no_gt.csv", train, STRICT_INPUT_COLUMNS)
    _csv_write(out_dir / f"test{len(test)}_input_strict_no_gt.csv", test, STRICT_INPUT_COLUMNS)
    _csv_write(out_dir / f"train{len(train)}_eval_reference.csv", train, EVAL_COLUMNS)
    _csv_write(out_dir / f"test{len(test)}_eval_reference.csv", test, EVAL_COLUMNS)
    validation = validate_selection(train, test, train_cases=len(train), test_cases=len(test))
    write_json_atomic(out_dir / "validation_report.json", validation)
    split = {
        "seed": seed,
        "train": [r["case_id"] for r in train],
        "test": [r["case_id"] for r in test],
        "strategy": "coverage-enriched selection strategy",
    }
    write_json_atomic(out_dir / "split.json", split)
    write_json_atomic(out_dir / "selection_config.json", cfg.data)
    fingerprint = {
        "created_at": utc_now(),
        "config_source": str(cfg.path),
        "target_count": 373,
        "train_cases": len(train),
        "test_cases": len(test),
        "selected_case_count": len(all_selected),
        "strategy": "coverage-enriched selection strategy",
    }
    write_json_atomic(out_dir / "dataset_fingerprint.json", fingerprint)
    _csv_write(out_dir / "train_test_distribution.csv", [{"split": "train", "cases": len(train)}, {"split": "test", "cases": len(test)}])
    (out_dir / "selection_report.md").write_text(
        "# AbdomenAtlasPro Case Selection\n\n"
        "Strategy: coverage-enriched selection strategy.\n\n"
        "The split is deterministic for the seed and uses only CT/header and ground-truth mask availability/quality metrics for selection; strict input manifests do not contain GT paths.\n",
        encoding="utf-8",
    )
    sha_rows = []
    for path in sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "sha256sums.txt"):
        sha_rows.append(f"{sha256_file(path)}  {path.name}")
    (out_dir / "sha256sums.txt").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    if validation["status"] == "success":
        (out_dir / "SUCCESS").write_text(utc_now() + "\n", encoding="utf-8")
    return {"status": validation["status"], "selection_dir": str(out_dir), "validation": validation}


def _fingerprint_payload(cfg: CaseSelectionConfig, train_cases: int, test_cases: int, seed: int) -> dict[str, Any]:
    return {
        "config": cfg.data,
        "train_cases": train_cases,
        "test_cases": test_cases,
        "seed": seed,
        "version": 1,
    }


def _state_path(output_dir: Path) -> Path:
    return output_dir / "state.json"


def read_selection_state(output_dir: Path) -> dict[str, Any]:
    path = _state_path(output_dir)
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_selection_state(output_dir: Path, state: dict[str, Any]) -> None:
    write_json_atomic(_state_path(output_dir), state)


def run_case_selection(
    train_cases: int,
    test_cases: int,
    seed: int,
    output_dir: Path,
    *,
    config_path: str | Path = "configs/abdomenatlaspro_case_selection.yaml",
    dry_run: bool = False,
    cache_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_case_selection_config(config_path)
    image_root = resolve_path(cfg.paths.get("image_root"))
    mask_root = resolve_path(cfg.paths.get("mask_root"))
    mapping_path = resolve_path(cfg.paths.get("target_mapping"))
    if image_root is None or mask_root is None or mapping_path is None:
        raise SchedulerError("Case selection config must define image_root, mask_root and target_mapping")
    ensure_not_raw_data_write_path(output_dir, image_root, mask_root)
    cache_options = cache_options or {}
    state = read_selection_state(output_dir)
    fingerprint = hashlib.sha256(json.dumps(_fingerprint_payload(cfg, train_cases, test_cases, seed), sort_keys=True).encode("utf-8")).hexdigest()
    if state.get("fingerprint") not in {None, fingerprint}:
        state = {"stages": {}, "fingerprint": fingerprint}
    if dry_run:
        return {"status": "dry_run", "train_cases": train_cases, "test_cases": test_cases, "seed": seed, "output_dir": str(output_dir), "cache_options": cache_options}
    output_dir.mkdir(parents=True, exist_ok=True)
    state.update({"fingerprint": fingerprint, "updated_at": utc_now()})
    stages = state.setdefault("stages", {})
    inv_cache = output_dir / "cache" / "inventory.json"
    header_cache = output_dir / "cache" / "headers.json"
    audit_cache = output_dir / "cache" / "deep_audit.json"
    reuse = bool(cache_options.get("reuse_cache", True))
    if reuse and not cache_options.get("force_inventory") and inv_cache.exists():
        inventory = json.loads(inv_cache.read_text(encoding="utf-8"))
    else:
        inventory = discover_inventory(image_root, mask_root)
        write_json_atomic(inv_cache, inventory)
    stages["inventory"] = {"status": "success", "rows": len(inventory)}
    if reuse and not cache_options.get("force_header_scan") and header_cache.exists():
        header_rows = json.loads(header_cache.read_text(encoding="utf-8"))
    else:
        header_rows = [{**r, **scan_ct_header(Path(str(r["ct_path"])))} for r in inventory]
        write_json_atomic(header_cache, header_rows)
    stages["header_scan"] = {"status": "success", "rows": len(header_rows)}
    candidates = prefilter_candidates(header_rows, train_cases + test_cases, cfg)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if reuse and not cache_options.get("force_deep_audit") and audit_cache.exists():
        cached = json.loads(audit_cache.read_text(encoding="utf-8"))
        audited = cached.get("audited", [])
        excluded = cached.get("excluded", [])
    else:
        audited = []
        excluded = []
        for row in candidates:
            try:
                audited.append(deep_mask_audit(row, list(mapping.get("targets") or [])))
            except Exception as exc:
                bad = dict(row)
                bad.update({"quality_pass": False, "exclusion_reason": str(exc)})
                excluded.append(bad)
        write_json_atomic(audit_cache, {"audited": audited, "excluded": excluded})
    stages["deep_audit"] = {"status": "success", "rows": len(audited), "excluded": len(excluded)}
    write_selection_state(output_dir, state)
    train, test, backups = select_and_split(audited, train_cases, test_cases, seed, cfg)
    result = write_selection_outputs(output_dir, cfg, train, test, backups, excluded, seed=seed)
    stages["select_and_split"] = {"status": result["status"]}
    write_selection_state(output_dir, state)
    return result


def resume_case_selection(output_dir: Path, *, config_path: str | Path = "configs/abdomenatlaspro_case_selection.yaml") -> dict[str, Any]:
    state = read_selection_state(output_dir)
    return {"status": "resumable", "selection_dir": str(output_dir), "state": state, "strategy": "skip completed successful cached stages and rerun missing or failed stages"}


def selection_status(selection_dir: Path) -> dict[str, Any]:
    report = selection_dir / "validation_report.json"
    return json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"status": "missing", "selection_dir": str(selection_dir)}


def clean_cache(selection_dir: Path) -> dict[str, Any]:
    cache = selection_dir / "cache"
    if cache.exists():
        shutil.rmtree(cache)
    return {"status": "success", "removed": str(cache)}


def _case_selector_command(config_path: str | Path, stage: str, train_cases: int, test_cases: int, seed: int, output_dir: Path) -> list[str]:
    return [
        "python",
        str(ROOT / "scripts" / "abdomenatlaspro_case_selector.py"),
        "--config",
        str(config_path),
        stage,
        "--train-cases",
        str(train_cases),
        "--test-cases",
        str(test_cases),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]


def _shell_join(args: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(x)) for x in args)


def _render_case_selection_sbatch(stage: str, deps: list[str], command: list[str], output_dir: Path) -> str:
    log_dir = output_dir / "logs"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={stage}",
            "#SBATCH --partition=cpu",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=32G",
            "#SBATCH --time=12:00:00",
            f"#SBATCH --output={log_dir}/{stage}_%j.out",
            f"#SBATCH --error={log_dir}/{stage}_%j.err",
            "",
            "set -euo pipefail",
            f"mkdir -p {log_dir}",
            f"echo '[case-selection] stage={stage} job=${{SLURM_JOB_ID:-local}}'",
            _shell_join(command),
            "",
        ]
    )


def build_case_selection_slurm_plan(
    train_cases: int,
    test_cases: int,
    seed: int,
    output_dir: Path,
    *,
    config_path: str | Path = "configs/abdomenatlaspro_case_selection.yaml",
    dry_run: bool,
) -> dict[str, Any]:
    cfg = load_case_selection_config(config_path)
    image_root = resolve_path(cfg.paths.get("image_root"))
    mask_root = resolve_path(cfg.paths.get("mask_root"))
    ensure_not_raw_data_write_path(output_dir, image_root, mask_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated_slurm").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    tasks = []
    for stage, deps in CASE_SELECTION_STAGES:
        command = _case_selector_command(config_path, CASE_SELECTION_STAGE_COMMANDS[stage], train_cases, test_cases, seed, output_dir)
        script_path = output_dir / "generated_slurm" / f"{stage}.sbatch"
        script_path.write_text(_render_case_selection_sbatch(stage, deps, command, output_dir), encoding="utf-8")
        tasks.append(
            {
                "stage": stage,
                "dependencies": deps,
                "partition": "cpu",
                "script": str(script_path),
                "command": command,
                "submitted": False,
            }
        )
    plan = {
        "status": "dry_run" if dry_run else "planned",
        "backend": "slurm",
        "submitted": False,
        "partition": "cpu",
        "train_cases": train_cases,
        "test_cases": test_cases,
        "seed": seed,
        "output_dir": str(output_dir),
        "created_at": utc_now(),
        "dag": [{"stage": stage, "dependencies": deps} for stage, deps in CASE_SELECTION_STAGES],
        "tasks": tasks,
        "policy": {
            "dry_run_reads_raw_data": False,
            "dry_run_calls_sbatch": False,
            "dry_run_calls_srun": False,
            "formal_scan_runs_under_slurm": True,
        },
    }
    write_json_atomic(output_dir / "dry_run_plan.json", plan)
    write_selection_state(
        output_dir,
        {
            "status": plan["status"],
            "backend": "slurm",
            "submitted": False,
            "partition": "cpu",
            "updated_at": utc_now(),
            "stages": {task["stage"]: {"status": "planned", "partition": "cpu", "script": task["script"]} for task in tasks},
        },
    )
    return plan


def submit_case_selection_slurm_plan(plan: dict[str, Any]) -> dict[str, Any]:
    submitted: dict[str, Any] = {}
    for task in plan.get("tasks", []):
        cmd = ["sbatch"]
        dep_ids = [submitted[d]["job_id"] for d in task.get("dependencies", []) if submitted.get(d, {}).get("job_id")]
        if dep_ids:
            cmd.append("--dependency=afterok:" + ":".join(dep_ids))
        cmd.append(str(task["script"]))
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise SchedulerError(f"sbatch failed for {task['stage']}: {proc.stderr}")
        job_id = proc.stdout.strip().split()[-1]
        submitted[task["stage"]] = {"status": "submitted", "job_id": job_id, "command": cmd, "stdout": proc.stdout.strip()}
    receipt = {
        "status": "submitted",
        "backend": "slurm",
        "submitted": True,
        "partition": "cpu",
        "jobs": submitted,
        "submitted_at": utc_now(),
    }
    output_dir = Path(str(plan["output_dir"]))
    write_json_atomic(output_dir / "submission_receipt.json", receipt)
    state = read_selection_state(output_dir)
    state.update({"status": "submitted", "submitted": True, "updated_at": utc_now()})
    write_selection_state(output_dir, state)
    return receipt
