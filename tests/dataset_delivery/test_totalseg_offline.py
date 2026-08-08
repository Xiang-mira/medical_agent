from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _write_totalseg_home(root: Path, *, license_present: bool = True, datasets: list[str] | None = None) -> Path:
    home = root / "totalseg_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(
        json.dumps({"license_number": "configured"} if license_present else {}),
        encoding="utf-8",
    )
    for dataset in datasets or []:
        (home / "nnunet" / "results" / dataset).mkdir(parents=True, exist_ok=True)
    return home


def test_totalseg_brain_ventricle_mapping_is_model_task_scoped():
    aliases = json.loads((REPO_ROOT / "configs" / "model_label_aliases.json").read_text(encoding="utf-8"))
    ts = aliases["models"]["totalsegmentator"]
    scoped = ts["task_scoped_aliases"]["brain_structures"]["ventricle"]

    assert ts["local_to_global"]["ventricle"] == "brain_ventricle"
    assert ts["mapping_types"]["ventricle"] == "exact_model_task_scoped_alias"
    assert scoped["canonical_target"] == "brain_ventricle"
    assert scoped["source_task"] == "brain_structures"
    assert scoped["source_label_id"] == 10
    assert "heart_ventricle_left" not in scoped.values()
    assert "heart_ventricle_right" not in scoped.values()


def test_totalseg_brain_structures_contract_records_task409_label10():
    config = json.loads((REPO_ROOT / "configs" / "totalseg_subtask_organs.json").read_text(encoding="utf-8"))
    brain = config["subtasks"]["brain_structures"]

    assert brain["task_id"] == 409
    assert brain["output_aliases"]["ventricle"]["canonical_target"] == "brain_ventricle"
    assert brain["output_aliases"]["ventricle"]["source_label_id"] == 10
    assert "Dataset409_neuro_550subj" in brain["offline_required_datasets"]
    assert "Dataset298_TotalSegmentator_total_6mm_1559subj" in brain["offline_required_datasets"]


def test_totalseg_offline_preflight_reports_missing_license_and_assets(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core.totalseg_runner import preflight_totalseg_offline_assets

    home = _write_totalseg_home(tmp_path, license_present=False, datasets=[])
    monkeypatch.setenv("MEDAI_TOTALSEG_HOME", str(home))
    monkeypatch.setenv("MEDAI_TOTALSEG_OFFLINE", "1")

    preflight = preflight_totalseg_offline_assets(["brain_structures"])

    assert preflight["status"] == "failed"
    assert "BLOCKED_LICENSE_REQUIRED" in preflight["failures"]
    assert "TOTALSEG_OFFLINE_ASSET_MISSING" in preflight["failures"]
    assert "Dataset409_neuro_550subj" in preflight["missing_datasets"]
    assert "Dataset298_TotalSegmentator_total_6mm_1559subj" in preflight["missing_datasets"]


def test_totalseg_offline_mode_fails_fast_without_subprocess(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import totalseg_runner

    home = _write_totalseg_home(
        tmp_path,
        license_present=True,
        datasets=["Dataset298_TotalSegmentator_total_6mm_1559subj"],
    )
    monkeypatch.setenv("MEDAI_TOTALSEG_HOME", str(home))
    monkeypatch.setenv("MEDAI_TOTALSEG_OFFLINE", "1")
    monkeypatch.setattr(
        totalseg_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess should not run when offline assets are incomplete")),
    )

    result = totalseg_runner.run_totalseg_with_contract(
        tmp_path / "ct.nii.gz",
        tmp_path / "case" / "per_model" / "totalsegmentator",
        tmp_path / "case",
        subtasks=["brain_structures"],
        case_id="case_001",
    )

    assert result["status"] == "failed"
    assert result["reason"] == "TOTALSEG_OFFLINE_ASSET_MISSING"
    assert result["offline_preflight"]["missing_datasets"] == ["Dataset409_neuro_550subj"]


def test_totalseg_offline_mode_fails_fast_when_crop_dependency_missing(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import totalseg_runner

    home = _write_totalseg_home(
        tmp_path,
        license_present=True,
        datasets=["Dataset409_neuro_550subj"],
    )
    monkeypatch.setenv("MEDAI_TOTALSEG_HOME", str(home))
    monkeypatch.setenv("MEDAI_TOTALSEG_OFFLINE", "1")
    monkeypatch.setattr(
        totalseg_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess should not run when crop assets are incomplete")),
    )

    result = totalseg_runner.run_totalseg_with_contract(
        tmp_path / "ct.nii.gz",
        tmp_path / "case" / "per_model" / "totalsegmentator",
        tmp_path / "case",
        subtasks=["brain_structures"],
        case_id="case_001",
    )

    assert result["status"] == "failed"
    assert result["reason"] == "TOTALSEG_OFFLINE_ASSET_MISSING"
    assert result["offline_preflight"]["missing_datasets"] == ["Dataset298_TotalSegmentator_total_6mm_1559subj"]


def test_totalseg_valid_offline_cached_mask_writes_identity_provenance(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import totalseg_runner

    home = _write_totalseg_home(
        tmp_path,
        license_present=True,
        datasets=[
            "Dataset298_TotalSegmentator_total_6mm_1559subj",
            "Dataset409_neuro_550subj",
        ],
    )
    monkeypatch.setenv("MEDAI_TOTALSEG_HOME", str(home))
    monkeypatch.setenv("MEDAI_TOTALSEG_OFFLINE", "1")
    monkeypatch.setattr(
        totalseg_runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""),
    )
    ct = _save(np.zeros((3, 3, 3), dtype=np.int16), tmp_path / "ct.nii.gz")
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    case_out = tmp_path / "case"
    _save(mask, case_out / "segmentations" / "totalseg_brain_structures" / "ventricle.nii.gz")
    subtask_config = json.loads((REPO_ROOT / "configs" / "totalseg_subtask_organs.json").read_text(encoding="utf-8"))

    result = totalseg_runner.run_totalseg_with_contract(
        ct,
        case_out / "per_model" / "totalsegmentator",
        case_out,
        subtask_config=subtask_config,
        subtasks=["brain_structures"],
        case_id="case_001",
    )

    assert result["status"] == "success"
    assert (case_out / "segmentations" / "ventricle.nii.gz").exists()
    assert (case_out / "segmentations" / "brain_ventricle.nii.gz").exists()
    provenance = json.loads((case_out / "segmentations" / "identity_provenance.json").read_text(encoding="utf-8"))
    item = provenance["organs"]["brain_ventricle"]
    assert item["source_task"] == "brain_structures"
    assert item["source_local_label"] == "ventricle"
    assert item["source_label_id"] == 10
    assert item["resolved_canonical_id"] == "brain_ventricle"


def test_totalseg_offline_manifest_records_checksums_without_license_value(tmp_path: Path, monkeypatch):
    from tools.dataset_delivery.totalseg_brain_ventricle_offline import (
        build_brain_ventricle_manifest,
        verify_brain_ventricle_manifest,
    )

    home = _write_totalseg_home(
        tmp_path,
        license_present=True,
        datasets=[
            "Dataset298_TotalSegmentator_total_6mm_1559subj",
            "Dataset409_neuro_550subj",
        ],
    )
    weight_file = home / "nnunet" / "results" / "Dataset409_neuro_550subj" / "fold_0" / "checkpoint_final.pth"
    weight_file.parent.mkdir(parents=True, exist_ok=True)
    weight_file.write_bytes(b"official-weight-placeholder")
    monkeypatch.setenv("MEDAI_TOTALSEG_HOME", str(home))

    manifest = build_brain_ventricle_manifest(home=home, output_manifest=tmp_path / "manifest.json")
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    verification = verify_brain_ventricle_manifest(home=home, manifest_path=tmp_path / "manifest.json")

    assert manifest["status"] == "READY"
    assert manifest["canonical_target"] == "brain_ventricle"
    assert manifest["task_id"] == 409
    assert manifest["source_class_id"] == 10
    assert manifest["license_present"] is True
    assert "configured" not in text
    assert verification["status"] == "READY"


def test_totalseg_manifest_preflight_reports_version_and_checksum_mismatch(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core.totalseg_runner import preflight_totalseg_offline_assets
    from tools.dataset_delivery.totalseg_brain_ventricle_offline import build_brain_ventricle_manifest

    home = _write_totalseg_home(
        tmp_path,
        license_present=True,
        datasets=[
            "Dataset298_TotalSegmentator_total_6mm_1559subj",
            "Dataset409_neuro_550subj",
        ],
    )
    weight_file = home / "nnunet" / "results" / "Dataset409_neuro_550subj" / "fold_0" / "checkpoint_final.pth"
    weight_file.parent.mkdir(parents=True, exist_ok=True)
    weight_file.write_bytes(b"official-weight-placeholder")
    manifest_path = tmp_path / "manifest.json"
    build_brain_ventricle_manifest(home=home, output_manifest=manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["totalsegmentator_version"] = "0.0.0"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    weight_file.write_bytes(b"changed")
    monkeypatch.setenv("MEDAI_TOTALSEG_HOME", str(home))
    monkeypatch.setenv("MEDAI_TOTALSEG_OFFLINE", "1")
    monkeypatch.setenv("MEDAI_TOTALSEG_MANIFEST", str(manifest_path))

    preflight = preflight_totalseg_offline_assets(["brain_structures"])

    assert preflight["status"] == "failed"
    assert "TOTALSEG_VERSION_MISMATCH" in preflight["failures"]
    assert "TOTALSEG_OFFLINE_CHECKSUM_MISMATCH" in preflight["failures"]
