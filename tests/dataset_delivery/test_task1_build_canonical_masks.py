from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.dataset_delivery import task1_build_canonical_masks as task1


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _mask(root: Path, case_id: str, name: str, data: bytes = b"mask") -> Path:
    path = root / "inputs" / "masks_original" / case_id / "segmentations" / f"{name}.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _config(tmp_path: Path) -> dict[str, Path]:
    mapping = _write(
        tmp_path / "organ_rename_mapping_373.csv",
        "\n".join(
            [
                "source_name,target_name,status,reason,notes",
                "left_gluteus_medius,gluteus_medius_left,confirmed,token_order,same",
                "right_lung_upper_lobe,lung_upper_right_lobe,confirmed,token_order,same",
                "lung_lobe_upper_right,lung_upper_right_lobe,confirmed,token_order,same",
                "vertebra_t12,vertebrae_t12,confirmed,spelling,same",
                "celiac_artery,celiac_aa,confirmed,token_order,same",
                "costa_1_left,rib_left_1,confirmed,token_order,same",
                "coronary_arteries,coronary_artery,pending_review,scope_uncertain,review",
            ]
        )
        + "\n",
    )
    non_rename = _write(
        tmp_path / "non_rename_decisions.csv",
        "\n".join(
            [
                "source_name,target_name,classification,status,reason,evidence",
                "iliac_vena_left,common_iliac_vein_left,task2_generate,confirmed_non_rename,generic iliac vein to common iliac vein segment requires new mask,evidence",
                "kidney,kidney_cortex,task2_generate,confirmed_non_rename,whole organ to cortex is coarse-to-fine,evidence",
            ]
        )
        + "\n",
    )
    aliases = _write(
        tmp_path / "task1_alias_groups.csv",
        "\n".join(
            [
                "alias_group,target_name,source_name,status,evidence",
                "alias_lung_upper_right_lobe,lung_upper_right_lobe,right_lung_upper_lobe,confirmed,evidence",
                "alias_lung_upper_right_lobe,lung_upper_right_lobe,lung_lobe_upper_right,confirmed,evidence",
            ]
        )
        + "\n",
    )
    target_config = _write(
        tmp_path / "targets.json",
        json.dumps(
            {
                "target_organs": [
                    "gluteus_medius_left",
                    "lung_upper_right_lobe",
                    "vertebrae_T12",
                    "iliac_vena_left",
                    "common_iliac_vein_left",
                    "kidney",
                    "kidney_cortex",
                    "inferior_vena_cava",
                    "celiac_aa",
                    "rib_left_1",
                    "coronary_artery",
                    "artery_common_carotid_left",
                    "artery_common_carotid_right",
                    "artery_subclavian_left",
                    "artery_subclavian_right",
                ]
            }
        ),
    )
    return {"mapping": mapping, "non_rename": non_rename, "aliases": aliases, "target_config": target_config}


def _run(workspace: Path, cfg: dict[str, Path], *, mode: str = "dry-run", allow_unmapped: bool = False):
    return task1.build_task1_canonical_masks(
        workspace_root=workspace,
        mapping=cfg["mapping"],
        non_rename_decisions=cfg["non_rename"],
        alias_groups=cfg["aliases"],
        target_config=cfg["target_config"],
        mode=mode,
        expected_case_count=1,
        allow_unmapped=allow_unmapped,
    )


def _audit(workspace: Path) -> list[dict[str, str]]:
    with (workspace / "manifests" / "task1_373_rename_audit.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _outputs(workspace: Path) -> list[dict[str, str]]:
    with (workspace / "manifests" / "task1_373_canonical_outputs.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coverage(workspace: Path) -> list[dict[str, str]]:
    with (workspace / "manifests" / "task1_373_source_coverage.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_task1_canonical_dry_run_classifies_source_masks_without_writing(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    original = _mask(workspace, "CASE001", "left_gluteus_medius")
    _mask(workspace, "CASE001", "vertebrae_T12")
    _mask(workspace, "CASE001", "iliac_vena_left")
    _mask(workspace, "CASE001", "kidney")
    _mask(workspace, "CASE001", "lesion")
    _mask(workspace, "CASE001", "mystery_label")

    summary = _run(workspace, cfg, allow_unmapped=True)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert summary["status"] == "READY"
    assert rows["left_gluteus_medius"]["action"] == "CONFIRMED_RENAME"
    assert rows["left_gluteus_medius"]["canonical_name"] == "gluteus_medius_left"
    assert rows["vertebrae_t12"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["vertebrae_t12"]["canonical_name"] == "vertebrae_t12"
    assert rows["iliac_vena_left"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["kidney"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["lesion"]["action"] == "EXCLUDED_NON_ANATOMY"
    assert rows["mystery_label"]["action"] == "UNMAPPED"
    assert not (workspace / "inputs" / "masks_373_canonical").exists()
    assert original.read_bytes() == b"mask"


def test_task1_apply_is_checksum_identity_idempotent_and_restart_safe(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    source = _mask(workspace, "CASE001", "left_gluteus_medius", b"same-bytes")
    _mask(workspace, "CASE001", "vertebrae_T12", b"vertebra")

    first = _run(workspace, cfg, mode="apply", allow_unmapped=True)
    second = _run(workspace, cfg, mode="apply", allow_unmapped=True)
    validate = _run(workspace, cfg, mode="validate", allow_unmapped=True)
    dest = workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations" / "gluteus_medius_left.nii.gz"
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert first["status"] == second["status"] == validate["status"] == "READY"
    assert dest.read_bytes() == source.read_bytes()
    assert rows["left_gluteus_medius"]["source_checksum"] == rows["left_gluteus_medius"]["destination_checksum"]
    assert source.read_bytes() == b"same-bytes"


def test_confirmed_alias_checksum_mismatch_resolves_without_blocking_or_union(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "right_lung_upper_lobe", b"a")
    _mask(workspace, "CASE001", "lung_lobe_upper_right", b"b")

    summary = _run(workspace, cfg, allow_unmapped=True)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert summary["status"] == "READY"
    assert summary["unapproved_collision_count"] == 0
    assert rows["right_lung_upper_lobe"]["action"] == "SELECTED_CONFIRMED_ALIAS"
    assert rows["lung_lobe_upper_right"]["action"] == "REDUNDANT_CONFIRMED_ALIAS"
    assert rows["right_lung_upper_lobe"]["selection_policy"] == "confirmed_alias_priority_alias_groups_then_mapping_order"

    applied = _run(workspace, cfg, mode="apply", allow_unmapped=True)
    dest = workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations" / "lung_upper_right_lobe.nii.gz"
    assert applied["status"] == "READY"
    assert dest.read_bytes() == b"a"
    assert dest.read_bytes() != b"ab"


def test_canonical_source_exists_over_confirmed_alias_without_checksum_comparison(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "lung_upper_right_lobe", b"canonical")
    _mask(workspace, "CASE001", "right_lung_upper_lobe", b"alias")

    _run(workspace, cfg, mode="apply", allow_unmapped=True)
    rows = {row["source_name"]: row for row in _audit(workspace)}
    outputs = _outputs(workspace)
    dest = workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations" / "lung_upper_right_lobe.nii.gz"

    assert rows["lung_upper_right_lobe"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["right_lung_upper_lobe"]["action"] == "REDUNDANT_CONFIRMED_ALIAS"
    assert rows["lung_upper_right_lobe"]["selection_policy"] == "canonical_source_over_confirmed_aliases"
    assert outputs[0]["resolution_type"] == "CANONICAL_SOURCE_OVER_ALIAS"
    assert dest.read_bytes() == b"canonical"


def test_confirmed_alias_selection_is_independent_of_filesystem_order(tmp_path: Path):
    cfg = _config(tmp_path)
    selections = []
    for index, names in enumerate((["lung_lobe_upper_right", "right_lung_upper_lobe"], ["right_lung_upper_lobe", "lung_lobe_upper_right"])):
        workspace = tmp_path / f"workspace_{index}"
        for name in names:
            _mask(workspace, "CASE001", name, name.encode())
        _run(workspace, cfg, mode="apply", allow_unmapped=True)
        outputs = _outputs(workspace)
        selections.append(outputs[0]["selected_source_name"])

    assert selections == ["right_lung_upper_lobe", "right_lung_upper_lobe"]


def test_non_confirmed_many_to_one_still_blocks_as_unapproved_collision(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "right_lung_upper_lobe", b"a")
    _mask(workspace, "CASE001", "unapproved_alias", b"b")
    source_dir = workspace / "inputs" / "masks_original" / "CASE001" / "segmentations"
    dest_dir = workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations"

    rows, collisions, _unmapped = task1._plan_case(
        case_id="CASE001",
        source_dir=source_dir,
        dest_dir=dest_dir,
        target_set={"lung_upper_right_lobe"},
        confirmed={
            "right_lung_upper_lobe": SimpleNamespace(target_name="lung_upper_right_lobe", status="confirmed", reason="confirmed"),
            "unapproved_alias": SimpleNamespace(target_name="lung_upper_right_lobe", status="pending_review", reason="not_confirmed"),
        },
        pending={},
        rejected={},
        mapping_order={("right_lung_upper_lobe", "lung_upper_right_lobe"): 0, ("unapproved_alias", "lung_upper_right_lobe"): 1},
        non_rename={},
        alias_groups={},
    )
    by_source = {row["source_name"]: row for row in rows}

    assert len(collisions) == 1
    assert by_source["right_lung_upper_lobe"]["action"] == "UNAPPROVED_COLLISION"
    assert by_source["unapproved_alias"]["action"] == "UNAPPROVED_COLLISION"
    assert collisions[0]["collision_status"] == "UNAPPROVED_COLLISION"


def test_task1_unmapped_blocks_apply_unless_explicitly_allowed(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "unknown_source")

    blocked = _run(workspace, cfg)

    assert blocked["status"] == "BLOCKED"
    with pytest.raises(Exception, match="unmapped"):
        _run(workspace, cfg, mode="apply")


def test_task1_non_rename_coarse_to_fine_is_forbidden_not_renamed(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "iliac_vena_left")
    _mask(workspace, "CASE001", "kidney")

    _run(workspace, cfg, allow_unmapped=True)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert rows["iliac_vena_left"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["kidney"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert not any(row["canonical_name"] == "common_iliac_vein_left" for row in rows.values())
    assert not any(row["canonical_name"] == "kidney_cortex" for row in rows.values())


def test_pathology_labels_are_excluded_from_373_canonical_tree(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    for name in ("colon_cancer_primaries", "pancreatic_pdac", "pancreatic_pnet"):
        _mask(workspace, "CASE001", name)

    summary = _run(workspace, cfg, allow_unmapped=True)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert summary["excluded_non_anatomy_count"] == 3
    assert all(rows[name]["action"] == "EXCLUDED_NON_ANATOMY" for name in rows)
    assert _outputs(workspace) == []


def test_canonical_source_without_self_mapping_is_authoritative_for_celiac_and_rib_aliases(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "celiac_aa", b"canonical-celiac")
    _mask(workspace, "CASE001", "celiac_artery", b"alias-celiac")
    _mask(workspace, "CASE001", "rib_left_1", b"canonical-rib")
    _mask(workspace, "CASE001", "costa_1_left", b"alias-rib")

    summary = _run(workspace, cfg, mode="apply", allow_unmapped=True)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert summary["status"] == "READY"
    assert summary["unapproved_collision_count"] == 0
    assert rows["celiac_aa"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["celiac_artery"]["action"] == "REDUNDANT_CONFIRMED_ALIAS"
    assert rows["celiac_aa"]["canonical_name"] == "celiac_aa"
    assert rows["rib_left_1"]["action"] == "CANONICAL_SOURCE_SELECTED"
    assert rows["costa_1_left"]["action"] == "REDUNDANT_CONFIRMED_ALIAS"
    assert rows["rib_left_1"]["canonical_name"] == "rib_left_1"
    assert (workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations" / "celiac_aa.nii.gz").read_bytes() == b"canonical-celiac"
    assert not (workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations" / "celiac_aa_celiac_artery.nii.gz").exists()


def test_pending_mapping_is_audit_only_nonblocking_and_not_written(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "coronary_arteries", b"pending")

    summary = _run(workspace, cfg)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert summary["status"] == "READY"
    assert summary["pending_mapping_count"] == 1
    assert summary["unmapped_count"] == 0
    assert rows["coronary_arteries"]["action"] == "PENDING_MAPPING"
    assert rows["coronary_arteries"]["destination_path"] == ""
    assert _outputs(workspace) == []
    assert not (workspace / "inputs" / "masks_373_canonical" / "CASE001" / "segmentations" / "coronary_artery.nii.gz").exists()


def test_outside_373_and_granularity_sources_are_nonblocking_exclusions(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    for name in ("lymph_nodes", "visceral_adipose_tissue", "liver_vessels", "skeletal_muscle"):
        _mask(workspace, "CASE001", name)

    summary = _run(workspace, cfg)
    rows = {row["source_name"]: row for row in _audit(workspace)}

    assert summary["status"] == "READY"
    assert rows["lymph_nodes"]["action"] == "EXCLUDED_OUTSIDE_373"
    assert rows["visceral_adipose_tissue"]["action"] == "EXCLUDED_OUTSIDE_373"
    assert rows["liver_vessels"]["action"] == "OUTSIDE_373_OR_GRANULARITY_MISMATCH"
    assert rows["skeletal_muscle"]["action"] == "OUTSIDE_373_OR_GRANULARITY_MISMATCH"
    assert summary["unmapped_count"] == 0
    assert _outputs(workspace) == []


def test_summary_status_counts_match_audit_rows(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "left_gluteus_medius")
    _mask(workspace, "CASE001", "lung_upper_right_lobe")
    _mask(workspace, "CASE001", "right_lung_upper_lobe")
    _mask(workspace, "CASE001", "lymph_nodes")
    _mask(workspace, "CASE001", "coronary_arteries")

    summary = _run(workspace, cfg)
    rows = _audit(workspace)
    counted: dict[str, int] = {}
    for row in rows:
        counted[row["action"]] = counted.get(row["action"], 0) + 1

    assert summary["status_counts"] == counted
    assert summary["audit_row_count"] == len(rows)
    assert summary["confirmed_rename_count"] == counted["CONFIRMED_RENAME"]
    assert summary["canonical_source_selected_count"] == counted["CANONICAL_SOURCE_SELECTED"]
    assert summary["redundant_confirmed_alias_count"] == counted["REDUNDANT_CONFIRMED_ALIAS"]


def test_coverage_audit_distinguishes_available_granularity_gap_and_missing_without_absence_inference(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "left_gluteus_medius", b"source")
    _mask(workspace, "CASE001", "iliac_vena_left", b"coarse")

    apply_summary = _run(workspace, cfg, mode="apply", allow_unmapped=True)
    coverage_summary = task1.audit_task1_source_coverage(
        workspace_root=workspace,
        non_rename_decisions=cfg["non_rename"],
        target_config=cfg["target_config"],
        expected_case_count=1,
    )
    rows = {(row["case_id"], row["canonical_target"]): row for row in _coverage(workspace)}

    assert apply_summary["status"] == "READY"
    assert coverage_summary["status"] == "READY"
    assert coverage_summary["coverage_row_count"] == 15
    assert rows[("CASE001", "gluteus_medius_left")]["coverage_status"] == "SOURCE_AVAILABLE"
    assert rows[("CASE001", "common_iliac_vein_left")]["coverage_status"] == "GRANULARITY_GAP"
    assert rows[("CASE001", "common_iliac_vein_left")]["source_label_names"] == "iliac_vena_left"
    assert rows[("CASE001", "kidney_cortex")]["coverage_status"] == "TARGET_MISSING"
    assert rows[("CASE001", "kidney_cortex")]["needs_task2_generation"] == "true"
    assert coverage_summary["absence_inference_policy"] == "source_mask_absence_is_not_ABSENT_or_OUT_OF_FOV"
    assert {row["coverage_status"] for row in rows.values()}.isdisjoint({"ABSENT", "OUT_OF_FOV"})


def test_four_teacher_confirmed_artery_legacy_mappings_are_in_default_artifacts():
    mapping_rows = {
        row["source_name"]: row
        for row in task1.read_csv_rows(task1.DEFAULT_MAPPING)
        if row["source_name"] in {
            "left_common_carotid_artery",
            "right_common_carotid_artery",
            "left_subclavian_artery",
            "right_subclavian_artery",
        }
    }
    boundary_rows = {
        row["source_name"]: row
        for row in task1.read_csv_rows(task1.DEFAULT_BOUNDARY)
        if row["source_name"] in mapping_rows
    }
    target_set, _raw_count = task1._canonical_target_contract(task1.DEFAULT_TARGET_CONFIG)

    assert mapping_rows == {
        "left_common_carotid_artery": {
            "source_name": "left_common_carotid_artery",
            "target_name": "artery_common_carotid_left",
            "status": "confirmed",
            "reason": "token_order",
            "notes": "same_anatomy_same_side_same_granularity",
        },
        "right_common_carotid_artery": {
            "source_name": "right_common_carotid_artery",
            "target_name": "artery_common_carotid_right",
            "status": "confirmed",
            "reason": "token_order",
            "notes": "same_anatomy_same_side_same_granularity",
        },
        "left_subclavian_artery": {
            "source_name": "left_subclavian_artery",
            "target_name": "artery_subclavian_left",
            "status": "confirmed",
            "reason": "token_order",
            "notes": "same_anatomy_same_side_same_granularity",
        },
        "right_subclavian_artery": {
            "source_name": "right_subclavian_artery",
            "target_name": "artery_subclavian_right",
            "status": "confirmed",
            "reason": "token_order",
            "notes": "same_anatomy_same_side_same_granularity",
        },
    }
    assert set(boundary_rows) == set(mapping_rows)
    assert all(row["classification"] == "task1_rename" for row in boundary_rows.values())
    assert all(row["status"] == "confirmed" for row in boundary_rows.values())
    assert {row["target_name"] for row in mapping_rows.values()} <= target_set
