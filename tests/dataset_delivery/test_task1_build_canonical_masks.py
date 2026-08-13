from __future__ import annotations

import csv
import json
from pathlib import Path

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
        json.dumps({"target_organs": ["gluteus_medius_left", "lung_upper_right_lobe", "vertebrae_T12", "iliac_vena_left", "kidney", "inferior_vena_cava"]}),
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
    assert rows["left_gluteus_medius"]["action"] == "RENAMED"
    assert rows["left_gluteus_medius"]["canonical_name"] == "gluteus_medius_left"
    assert rows["vertebrae_t12"]["action"] == "KEEP_CANONICAL"
    assert rows["vertebrae_t12"]["canonical_name"] == "vertebrae_t12"
    assert rows["iliac_vena_left"]["action"] == "KEEP_CANONICAL"
    assert rows["kidney"]["action"] == "KEEP_CANONICAL"
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


def test_task1_collision_fails_closed_even_for_approved_alias_group(tmp_path: Path):
    cfg = _config(tmp_path)
    workspace = tmp_path / "workspace"
    _mask(workspace, "CASE001", "right_lung_upper_lobe", b"a")
    _mask(workspace, "CASE001", "lung_lobe_upper_right", b"b")

    summary = _run(workspace, cfg, allow_unmapped=True)
    rows = _audit(workspace)
    collisions = list(csv.DictReader((workspace / "manifests" / "task1_373_collision_report.csv").open("r", encoding="utf-8")))

    assert summary["status"] == "BLOCKED"
    assert summary["collision_count"] == 1
    assert {row["action"] for row in rows} == {"COLLISION"}
    assert collisions[0]["collision_status"] == "APPROVED_ALIAS_COLLISION"
    with pytest.raises(Exception, match="collision"):
        _run(workspace, cfg, mode="apply", allow_unmapped=True)


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

    assert rows["iliac_vena_left"]["action"] == "KEEP_CANONICAL"
    assert rows["kidney"]["action"] == "KEEP_CANONICAL"
    assert not any(row["canonical_name"] == "common_iliac_vein_left" for row in rows.values())
    assert not any(row["canonical_name"] == "kidney_cortex" for row in rows.values())
