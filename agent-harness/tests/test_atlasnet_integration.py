from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ATLASNET_373_ORGANS = {
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "cbd_stent",
    "celiac_aa (celiac_artery)", "colon", "common_bile_duct", "duodenum",
    "gall_bladder", "inferior_vena_cava", "intestine", "kidney_left",
    "kidney_right", "liver", "pancreas", "pancreatic_duct",
    "portal_vein_and_splenic_vein", "renal_vein_left", "renal_vein_right",
    "spleen", "stomach", "superior_mesenteric_artery",
}


def test_atlasnet_registry_entry_is_enabled_and_complete():
    sys.path.insert(0, str(ROOT / "agent-harness"))
    from cli_anything.medai.core.model_registry import load_registry

    entry = load_registry(ROOT / "configs/model_registry.yaml")["models"]["atlasnet"]
    assert entry["enabled"] is True
    assert entry["status"] == "ready_if_checkpoint_folder_present"
    assert entry["private_checkpoint"] is False
    assert len(entry["covered_organs"]) == 25
    assert "atlasnet_predict_and_split.py" in entry["command_template"]


def test_atlasnet_cli_dry_run_uses_uniform_per_model_contract(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_medai_cli.py"),
            "--json",
            "infer",
            "--model", "atlasnet",
            "--image", str(tmp_path / "ct.nii.gz"),
            "--output", str(tmp_path / "outputs"),
            "--case-id", "case_001",
            "--device", "cuda:0",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)["infer"]
    assert result["status"] == "dry_run"
    assert result["model_key"] == "atlasnet"
    assert result["per_model_dir"].endswith("case_001/per_model/atlasnet")
    assert "--device \"cuda:0\"" in result["command"]
    assert "--per-model-dir" in result["command"]


def test_atlasnet_wrapper_dry_run_matches_official_modelfolder_command(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/atlasnet_predict_and_split.py"),
            "--image", str(tmp_path / "ct.nii.gz"),
            "--output", str(tmp_path / "case"),
            "--atlas-root", str(tmp_path / "ATLAS-Net"),
            "--label-map", str(ROOT / "configs/atlasnet_label_map.json"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    command = result["command"]
    assert result["status"] == "dry_run"
    assert command[0] == "nnUNetv2_predict_from_modelfolder"
    assert "checkpoint_final.pth" in command
    assert "nnUNetTrainer__nnUNetPlans__3d_fullres" in " ".join(command)


def test_atlasnet_is_formally_routed_for_exactly_its_373_organs():
    sys.path.insert(0, str(ROOT / "agent-harness"))
    from cli_anything.medai.core.organ_router import route_organs

    target = json.loads((ROOT / "configs/student_3d_prompt_target_organs.json").read_text())
    routed = route_organs(target["target_organs"])
    actual = {
        organ
        for organ, candidates in routed["ranked_candidates"].items()
        if any(item.get("model_key") == "atlasnet" for item in candidates)
    }
    assert actual == ATLASNET_373_ORGANS
    assert not ({"pancreatic_pdac", "pancreatic_cyst", "pancreatic_pnet"} & set(target["target_organs"]))
