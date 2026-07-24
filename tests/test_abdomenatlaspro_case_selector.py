from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from scheduler.case_selection import deep_mask_audit, run_case_selection, scan_ct_header
from scheduler.manifest import assert_strict_no_gt_manifest


def _nii(path: Path, shape=(4, 5, 6), affine=None, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.uint8) + value
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))


def test_scan_ct_header_uses_true_si_axis(tmp_path):
    affine = np.array([[0, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    ct = tmp_path / "ct.nii.gz"
    _nii(ct, shape=(7, 8, 9), affine=affine)
    row = scan_ct_header(ct)
    assert row["header_status"] == "success"
    assert row["si_axis_index"] == 0
    assert row["physical_si_extent_mm"] == 28


def test_deep_mask_audit_counts_positive_zero_and_bad_shape(tmp_path):
    ct = tmp_path / "images" / "BDMAP_0001" / "ct.nii.gz"
    seg = tmp_path / "masks" / "BDMAP_0001" / "segmentations"
    _nii(ct)
    _nii(seg / "liver.nii.gz", value=1)
    _nii(seg / "spleen.nii.gz", value=0)
    _nii(seg / "kidney.nii.gz", shape=(3, 5, 6), value=1)
    row = {"case_id": "BDMAP_0001", "ct_path": str(ct), "mask_dir": str(seg), "mask_file_count": 3}
    mapping = [
        {"target_name": "liver", "mapping_status": "direct", "source_masks": ["liver"]},
        {"target_name": "spleen", "mapping_status": "direct", "source_masks": ["spleen"]},
        {"target_name": "kidney", "mapping_status": "direct", "source_masks": ["kidney"]},
        {"target_name": "absent", "mapping_status": "dataset_absent", "source_masks": []},
    ]
    out = deep_mask_audit(row, mapping)
    assert out["mapped_373_positive_count"] == 1
    assert out["all_zero_mask_count"] == 1
    assert out["shape_mismatch_count"] == 1
    assert out["unsupported_mapping_count"] == 1


def test_run_case_selection_writes_no_gt_manifests(tmp_path):
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    for idx in range(1, 7):
        case = f"BDMAP_{idx:04d}"
        _nii(image_root / case / "ct.nii.gz", shape=(4, 5, 6 + idx))
        _nii(mask_root / case / "segmentations" / "liver.nii.gz", shape=(4, 5, 6 + idx), value=1)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"targets": [{"target_name": "liver", "mapping_status": "direct", "source_masks": ["liver"]}]}), encoding="utf-8")
    cfg = tmp_path / "case_selection.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"image_root": str(image_root), "mask_root": str(mask_root), "target_mapping": str(mapping)},
                "candidate_pool": {"multiplier": 2, "minimum": 4, "maximum": 10, "backup_ratio": 0.2},
                "scoring": {"mapped_positive_target_weight": 0.55, "physical_si_extent_weight": 0.35, "quality_weight": 0.10},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "selection"
    result = run_case_selection(2, 2, 123, out, config_path=cfg)
    assert result["status"] == "success"
    train = out / "train2_input_strict_no_gt.csv"
    test = out / "test2_input_strict_no_gt.csv"
    assert assert_strict_no_gt_manifest(train)["rows"] == 2
    assert assert_strict_no_gt_manifest(test)["rows"] == 2
    assert (out / "SUCCESS").exists()


def test_direct_script_help_runs_without_pythonpath():
    proc = subprocess.run([sys.executable, "scripts/abdomenatlaspro_case_selector.py", "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0
    assert "run" in proc.stdout


def test_slurm_dry_run_writes_cpu_plan_without_sbatch_or_scan(monkeypatch, tmp_path):
    import scripts.abdomenatlaspro_case_selector as selector
    import scheduler.case_selection as case_selection

    out = tmp_path / "slurm_dry_run"
    called = {"sbatch": False, "scan": False}
    monkeypatch.setattr(case_selection, "discover_inventory", lambda *a, **k: called.update(scan=True))

    def fake_run(*args, **kwargs):
        called["sbatch"] = True
        raise AssertionError("dry-run must not call subprocess.run/sbatch")

    monkeypatch.setattr(case_selection.subprocess, "run", fake_run)
    rc = selector.main([
        "run",
        "--train-cases",
        "2",
        "--test-cases",
        "2",
        "--seed",
        "20260724",
        "--backend",
        "slurm",
        "--output-dir",
        str(out),
        "--dry-run",
    ])
    assert rc == 0
    assert out.is_dir()
    assert (out / "dry_run_plan.json").exists()
    assert (out / "state.json").exists()
    scripts = sorted((out / "generated_slurm").glob("*.sbatch"))
    assert scripts
    text = "\n".join(p.read_text(encoding="utf-8") for p in scripts)
    assert "#SBATCH --partition=cpu" in text
    assert "--gres=gpu" not in text
    assert "srun" not in text
    assert called == {"sbatch": False, "scan": False}


def test_slurm_backend_submits_cpu_dag_without_local_full_scan(monkeypatch, tmp_path):
    import scripts.abdomenatlaspro_case_selector as selector
    import scheduler.case_selection as case_selection

    out = tmp_path / "slurm_submit"
    monkeypatch.setattr(case_selection, "discover_inventory", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not scan locally")))
    submitted = []

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "sbatch"
        submitted.append(cmd)
        return type("Proc", (), {"returncode": 0, "stdout": f"Submitted batch job {1000 + len(submitted)}\n", "stderr": ""})()

    monkeypatch.setattr(case_selection.subprocess, "run", fake_run)
    rc = selector.main([
        "run",
        "--train-cases",
        "2",
        "--test-cases",
        "2",
        "--seed",
        "20260724",
        "--backend",
        "slurm",
        "--output-dir",
        str(out),
    ])
    assert rc == 0
    assert submitted
    assert (out / "submission_receipt.json").exists()
