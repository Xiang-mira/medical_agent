from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from tools.dataset_delivery.propose_task1_finalization import (
    build_taxonomy_semantic_audit,
    compare_binary_masks,
    fingerprint_source_inventory,
    run_proposal_audit,
)


SEMANTIC_FIELDS = [
    "concept_group",
    "label_a",
    "label_b",
    "relationship",
    "same_anatomy",
    "same_laterality",
    "same_granularity",
    "evidence_type",
    "evidence_source",
    "evidence_reference",
    "decision",
    "notes",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def taxonomy(path: Path, labels: list[str]) -> Path:
    names = list(dict.fromkeys(labels + [f"organ_{i:03d}" for i in range(500)]))[:373]
    path.write_text(json.dumps({"target_organs": names}), encoding="utf-8")
    return path


def semantic(path: Path, rows: list[dict[str, str]]) -> Path:
    return write_csv(path, SEMANTIC_FIELDS, rows)


def semantic_row(
    concept_group: str,
    label_a: str,
    label_b: str,
    relationship: str,
    decision: str,
    *,
    same_anatomy: str = "true",
    same_laterality: str = "true",
    same_granularity: str = "true",
) -> dict[str, str]:
    return {
        "concept_group": concept_group,
        "label_a": label_a,
        "label_b": label_b,
        "relationship": relationship,
        "same_anatomy": same_anatomy,
        "same_laterality": same_laterality,
        "same_granularity": same_granularity,
        "evidence_type": "manual_data_dictionary_review",
        "evidence_source": "test",
        "evidence_reference": "test evidence",
        "decision": decision,
        "notes": "test",
    }


def mapping(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    return write_csv(
        path,
        ["source_name", "target_name", "status", "reason", "notes"],
        [{"source_name": src, "target_name": dst, "status": status, "reason": "test", "notes": ""} for src, dst, status in rows],
    )


def boundary(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    out = []
    for src, dst, status in rows:
        if status == "confirmed":
            out.append({
                "source_name": src,
                "target_name": dst,
                "same_anatomy": "true",
                "same_laterality": "true",
                "same_granularity": "true",
                "requires_voxel_change": "false",
                "classification": "task1_rename",
                "status": "confirmed",
                "reason": "test",
                "evidence": "exact test evidence",
                "alias_group": "",
            })
        else:
            out.append({
                "source_name": src,
                "target_name": dst,
                "same_anatomy": "unknown",
                "same_laterality": "unknown",
                "same_granularity": "unknown",
                "requires_voxel_change": "false",
                "classification": "boundary_review",
                "status": "pending_review",
                "reason": "test_pending",
                "evidence": "pending test evidence",
                "alias_group": "",
            })
    return write_csv(
        path,
        [
            "source_name",
            "target_name",
            "same_anatomy",
            "same_laterality",
            "same_granularity",
            "requires_voxel_change",
            "classification",
            "status",
            "reason",
            "evidence",
            "alias_group",
        ],
        out,
    )


def aliases(path: Path, groups: dict[str, tuple[str, list[str]]]) -> Path:
    rows = []
    for alias_group, (target, sources) in groups.items():
        for source in sources:
            rows.append({
                "alias_group": alias_group,
                "target_name": target,
                "source_name": source,
                "status": "confirmed",
                "evidence": "test alias",
            })
    return write_csv(path, ["alias_group", "target_name", "source_name", "status", "evidence"], rows)


def task2(path: Path, targets: list[str] | None = None) -> Path:
    rows = [
        {"target_name": target, "classification": "task2_generate", "status": "confirmed", "reason": "fixed", "evidence": "test"}
        for target in (targets or [])
    ]
    return write_csv(path, ["target_name", "classification", "status", "reason", "evidence"], rows)


def manifest(path: Path, data_root: Path, case_ids: list[str]) -> Path:
    rows = [
        {"index": str(i), "case_id": case_id, "reference_mask_dir": str(data_root / case_id / "segmentations")}
        for i, case_id in enumerate(case_ids)
    ]
    return write_csv(path, ["index", "case_id", "reference_mask_dir"], rows)


def save_mask(path: Path, arr: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4) if affine is None else affine), str(path))


def make_case(data_root: Path, case_id: str, masks: dict[str, np.ndarray], *, affine: np.ndarray | None = None) -> None:
    for name, arr in masks.items():
        save_mask(data_root / case_id / "segmentations" / f"{name}.nii.gz", arr, affine)


def base_files(tmp_path: Path, *, sources: list[str], target: str, semantic_rows: list[dict[str, str]] | None = None):
    tax = taxonomy(tmp_path / "taxonomy.json", [target, *sources, "coronary_artery"])
    map_path = mapping(tmp_path / "mapping.csv", [(source, target, "confirmed") for source in sources])
    boundary_path = boundary(tmp_path / "task_boundary_classification.csv", [(source, target, "confirmed") for source in sources])
    alias_path = aliases(tmp_path / "aliases.csv", {"alias_target": (target, sources)})
    task2_path = task2(tmp_path / "task2.csv")
    sem_path = semantic(
        tmp_path / "taxonomy_semantic_evidence.csv",
        semantic_rows or [semantic_row("semantic_test", target, sources[0], "exact_synonym", "confirmed_equivalent")],
    )
    return tax, map_path, boundary_path, alias_path, task2_path, sem_path


def test_taxonomy_semantic_rules_cover_confirmed_and_non_merge_relationships(tmp_path: Path) -> None:
    tax = taxonomy(
        tmp_path / "taxonomy.json",
        [
            "artery_common_carotid_left",
            "common_carotid_artery_left",
            "artery_brachiocephalic",
            "brachiocephalic_trunk",
            "carotid_artery_left",
            "common_carotid_artery_left_child",
            "parotid_glands",
            "parotid_gland_left",
            "kidney",
            "kidney_cortex",
            "costal_cartilages",
            "rib_cartilage",
        ],
    )
    sem = semantic(
        tmp_path / "semantic.csv",
        [
            semantic_row("token", "artery_common_carotid_left", "common_carotid_artery_left", "token_order_equivalent", "confirmed_equivalent"),
            semantic_row("synonym", "artery_brachiocephalic", "brachiocephalic_trunk", "exact_synonym", "confirmed_equivalent"),
            semantic_row("parent", "carotid_artery_left", "common_carotid_artery_left_child", "parent_child", "confirmed_not_equivalent", same_granularity="false"),
            semantic_row("aggregate", "parotid_glands", "parotid_gland_left", "aggregate_vs_sided", "confirmed_not_equivalent", same_laterality="false"),
            semantic_row("substructure", "kidney", "kidney_cortex", "whole_vs_substructure", "confirmed_not_equivalent", same_granularity="false"),
            semantic_row("ambiguous", "costal_cartilages", "rib_cartilage", "ambiguous", "pending_review", same_anatomy="unknown", same_laterality="unknown", same_granularity="unknown"),
        ],
    )
    result = build_taxonomy_semantic_audit(
        taxonomy=tax,
        mapping=mapping(tmp_path / "mapping.csv", []),
        boundary_classification=boundary(tmp_path / "boundary.csv", []),
        alias_groups=aliases(tmp_path / "aliases.csv", {}),
        task2_targets=task2(tmp_path / "task2.csv"),
        semantic_evidence=sem,
        output_root=tmp_path / "out",
    )["summary"]
    assert result["token_order_duplicate_count"] == 1
    assert result["confirmed_exact_synonym_count"] == 1
    assert result["parent_child_count"] == 1
    assert result["aggregate_vs_sided_count"] == 1
    assert result["whole_vs_substructure_count"] == 1
    assert result["pending_count"] >= 1
    assert result["medical_concept_count"] == "not_determined"


def test_binary_mask_comparison_statuses(tmp_path: Path) -> None:
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    shifted = arr.copy()
    shifted[1, 1, 2] = 1
    shifted[1, 1, 1] = 0
    empty = np.zeros((3, 3, 3), dtype=np.uint8)
    affine = np.eye(4)
    affine_shifted = np.eye(4)
    affine_shifted[0, 3] = 4
    save_mask(tmp_path / "a.nii.gz", arr)
    save_mask(tmp_path / "b.nii.gz", arr.copy())
    save_mask(tmp_path / "affine.nii.gz", arr.copy(), affine_shifted)
    save_mask(tmp_path / "shape.nii.gz", np.zeros((4, 3, 3), dtype=np.uint8))
    save_mask(tmp_path / "voxel.nii.gz", shifted, affine)
    save_mask(tmp_path / "empty.nii.gz", empty)

    assert compare_binary_masks(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz")["status"] == "exact_equal"
    affine_result = compare_binary_masks(tmp_path / "a.nii.gz", tmp_path / "affine.nii.gz")
    assert affine_result["status"] == "geometry_mismatch"
    assert affine_result["reason"] == "affine_mismatch"
    shape_result = compare_binary_masks(tmp_path / "a.nii.gz", tmp_path / "shape.nii.gz")
    assert shape_result["status"] == "geometry_mismatch"
    assert shape_result["reason"] == "shape_mismatch"
    voxel_result = compare_binary_masks(tmp_path / "a.nii.gz", tmp_path / "voxel.nii.gz")
    assert voxel_result["status"] == "voxel_mismatch"
    assert voxel_result["xor_voxel_count"] == 2
    assert compare_binary_masks(tmp_path / "empty.nii.gz", tmp_path / "empty.nii.gz")["status"] == "empty_mask"


def test_three_source_pairwise_audit_generates_global_priority_and_preserves_source_fingerprint(tmp_path: Path) -> None:
    sources = ["source_a", "source_b", "source_c"]
    tax, map_path, boundary_path, alias_path, task2_path, sem_path = base_files(tmp_path, sources=sources, target="target_a")
    data = tmp_path / "data"
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[0, 0, 0] = 1
    make_case(data, "CASE_001", {source: arr for source in sources})
    make_case(data, "CASE_002", {source: arr for source in sources})
    case_manifest = manifest(tmp_path / "cases.csv", data, ["CASE_001", "CASE_002"])
    before = fingerprint_source_inventory(data, case_manifest)
    summary = run_proposal_audit(
        data_root=data,
        case_manifest=case_manifest,
        output_root=tmp_path / "out",
        mapping=map_path,
        taxonomy=tax,
        boundary_classification=boundary_path,
        alias_groups=alias_path,
        task2_targets=task2_path,
        semantic_evidence=sem_path,
    )
    after = fingerprint_source_inventory(data, case_manifest)
    rows = read_csv(tmp_path / "out" / "alias_voxel_equivalence_rows.csv")
    priorities = read_csv(tmp_path / "out" / "proposed" / "proposed_source_priority.csv")
    assert sum(1 for row in rows if row["status"] == "exact_equal") == 6
    assert priorities[0]["case_id"] == "*"
    assert priorities[0]["decision"] == "use_global_source_priority"
    assert before["hash"] == after["hash"]
    assert summary["source_data_mutation"] is False
    assert summary["canonical_config_mutation"] is False
    assert summary["focus_alias_summary"]["celiac"]["coexisting_case_count"] == 0


def test_partial_alias_equivalence_generates_case_priority_and_blocks_mismatch(tmp_path: Path) -> None:
    sources = ["source_a", "source_b"]
    tax, map_path, boundary_path, alias_path, task2_path, sem_path = base_files(tmp_path, sources=sources, target="target_a")
    data = tmp_path / "data"
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[0, 0, 0] = 1
    mismatch = arr.copy()
    mismatch[2, 2, 2] = 1
    make_case(data, "CASE_EXACT", {"source_a": arr, "source_b": arr.copy()})
    make_case(data, "CASE_BAD", {"source_a": arr, "source_b": mismatch})
    summary = run_proposal_audit(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data, ["CASE_EXACT", "CASE_BAD"]),
        output_root=tmp_path / "out",
        mapping=map_path,
        taxonomy=tax,
        boundary_classification=boundary_path,
        alias_groups=alias_path,
        task2_targets=task2_path,
        semantic_evidence=sem_path,
    )
    priorities = read_csv(tmp_path / "out" / "proposed" / "proposed_source_priority.csv")
    blocked_cases = read_csv(tmp_path / "out" / "blocked_cases.csv")
    blocked_groups = read_csv(tmp_path / "out" / "blocked_alias_groups.csv")
    assert priorities[0]["case_id"] == "CASE_EXACT"
    assert priorities[0]["decision"] == "use_case_source_priority"
    assert blocked_cases[0]["case_id"] == "CASE_BAD"
    assert blocked_cases[0]["decision"] == "blocked_alias_mismatch"
    assert blocked_groups[0]["decision"] == "blocked_alias_mismatch"
    assert "alias_target" in summary["alias_voxel_equivalence_summary"]["blocked_alias_groups"]


def test_coronary_singular_plural_stays_pending_and_canonical_files_are_not_overwritten(tmp_path: Path) -> None:
    tax = taxonomy(tmp_path / "taxonomy.json", ["coronary_artery", "coronary_arteries"])
    map_path = mapping(tmp_path / "mapping.csv", [("coronary_arteries", "coronary_artery", "confirmed")])
    boundary_path = boundary(tmp_path / "task_boundary_classification.csv", [("coronary_arteries", "coronary_artery", "confirmed")])
    alias_path = aliases(tmp_path / "aliases.csv", {})
    task2_path = task2(tmp_path / "task2.csv")
    sem_path = semantic(
        tmp_path / "semantic.csv",
        [
            semantic_row(
                "semantic_coronary_singular_plural",
                "coronary_artery",
                "coronary_arteries",
                "ambiguous",
                "pending_review",
                same_anatomy="unknown",
                same_laterality="unknown",
                same_granularity="unknown",
            )
        ],
    )
    data = tmp_path / "data"
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[0, 0, 0] = 1
    make_case(data, "CASE_001", {"coronary_arteries": arr})
    original_mapping_text = map_path.read_text(encoding="utf-8")
    summary = run_proposal_audit(
        data_root=data,
        case_manifest=manifest(tmp_path / "cases.csv", data, ["CASE_001"]),
        output_root=tmp_path / "out",
        mapping=map_path,
        taxonomy=tax,
        boundary_classification=boundary_path,
        alias_groups=alias_path,
        task2_targets=task2_path,
        semantic_evidence=sem_path,
    )
    proposed_mapping = read_csv(tmp_path / "out" / "proposed" / "proposed_organ_rename_mapping_373.csv")
    proposed_boundary = read_csv(tmp_path / "out" / "proposed" / "proposed_task_boundary_classification.csv")
    pending_rows = read_csv(tmp_path / "out" / "pending_semantic_review.csv")
    assert summary["coronary"]["status"] == "pending_review"
    assert proposed_mapping[0]["status"] == "pending_review"
    assert proposed_mapping[0]["reason"] == "insufficient_evidence_singular_plural_scope"
    assert proposed_boundary[0]["classification"] == "boundary_review"
    assert any(row["concept_group"] == "semantic_coronary_singular_plural" for row in pending_rows)
    assert map_path.read_text(encoding="utf-8") == original_mapping_text
    assert (data / "CASE_001" / "segmentations" / "coronary_arteries.nii.gz").exists()
    assert not (data / "CASE_001" / "segmentations" / "coronary_artery.nii.gz").exists()


def test_task1_proposal_audit_python_310_compile() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "tools/dataset_delivery/propose_task1_finalization.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
