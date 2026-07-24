from __future__ import annotations

import json
import subprocess
import sys


def test_completion_script_promotes_real_masks(tmp_path):
    mask_root = tmp_path / "masks"
    seg = mask_root / "BDMAP_00000001" / "segmentations"
    seg.mkdir(parents=True)
    (seg / "ulna.nii.gz").write_bytes(b"not-a-real-nifti-needed-for-name-audit")

    targets = ["ulna", *[f"organ_{idx:03d}" for idx in range(372)]]
    target_config = tmp_path / "targets.json"
    target_config.write_text(json.dumps({
        "target_organs": targets,
        "organ_to_student_id": {target: idx for idx, target in enumerate(targets)},
        "organ_to_prompt": {target: f"Segment {target}." for target in targets},
    }), encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "targets": [
            {
                "target_name": "ulna",
                "target_index": 0,
                "mapping_status": "unverified_skip",
                "participates_in_pilot338": False,
            },
            *[
                {
                    "target_name": target,
                    "target_index": idx,
                    "mapping_status": "direct",
                    "participates_in_pilot338": True,
                }
                for idx, target in enumerate(targets[1:], start=1)
            ],
        ]
    }), encoding="utf-8")
    output_mapping = tmp_path / "completed.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/complete_abdomenatlaspro_373_mapping.py",
            "--mask-root",
            str(mask_root),
            "--target-config",
            str(target_config),
            "--target-mapping",
            str(mapping),
            "--output-mapping",
            str(output_mapping),
            "--output-target-config",
            str(tmp_path / "full_config.json"),
            "--output-allowlist",
            str(tmp_path / "allowlist.json"),
            "--report",
            str(tmp_path / "report.json"),
            "--write",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    completed = json.loads(output_mapping.read_text(encoding="utf-8"))
    ulna = next(item for item in completed["targets"] if item["target_name"] == "ulna")
    assert ulna["mapping_status"] == "direct"
    assert ulna["participates_in_pilot338"] is True
    assert ulna["participates_in_abdomenatlaspro_373"] is True
    assert completed["pilot_target_count"] == 373
    assert completed["completed_target_count"] == 373
