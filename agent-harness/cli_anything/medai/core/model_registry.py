from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _norm_key(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _split_checkpoint_families(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    # Keep the visible checkpoint tokens while stripping task hints in parentheses.
    raw = re.sub(r"\([^)]*\)", "", raw)
    parts = re.split(r"/|,|;|\+|\band\b", raw)
    families: list[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        # Normalize common labels without losing the human-readable spelling.
        compact = re.sub(r"\s+", " ", p)
        families.append(compact)
    # Preserve order and remove duplicates.
    seen = set()
    out = []
    for f in families:
        k = _norm_key(f)
        if k not in seen:
            out.append(f)
            seen.add(k)
    return out


def _family_key(name: str) -> str:
    n = _norm_key(name)
    aliases = {
        "totalsegmentator": "totalsegmentator",
        "total_segmentator": "totalsegmentator",
        "total_seg": "totalsegmentator",
        "moose3_0": "moose3_0",
        "moose_3_0": "moose3_0",
        "vsmtrans": "vsmtrans",
        "vs_mtrans": "vsmtrans",
        "vista3d": "vista3d",
        "atlas_net": "atlasnet",
        "atlasnet": "atlasnet",
        "nnunet": "nnunet_private",
        "nnunet_private": "nnunet_private",
        "saros_nnunet": "saros_nnunet",
        "epai": "epai_20250421",
    }
    return aliases.get(n, n)


# ---------------------------------------------------------------------------
# Model-specific M-step metadata
# ---------------------------------------------------------------------------
# The teacher's intended loop is selected-model-aware: choose a model for the
# task/organ in the E-step, refine its annotations, then update that model or a
# compatible model family in the M-step.  These fields make that explicit.

def _mstep_metadata_for_model(key: str, checkpoint_root: str = "checkpoints") -> dict[str, Any]:
    key = _family_key(key)
    trainable_nnunet = {
        "epai_20250421": {
            "trainable": "yes_if_nnunet_checkpoint_present",
            "mstep_role": "primary_trainable_for_pancreas_and_pancreatic_tumor_and_candidate_for_25_class_abdominal_organs",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1939,
            "mstep_init_from": f"{checkpoint_root}/qchen76_2025_0421",
            "mstep_reason": "ePAI qchen76_2025_0421 is the teacher-specified 25-class abdominal organ/duct/tumor checkpoint and is nnUNet-style, so it is the preferred M-step target for PanTS pancreas tasks and a candidate for related organ refinement.",
        },
        "cads": {
            "trainable": "yes_if_full_nnunet_training_state_present",
            "mstep_role": "trainable_candidate_for_broad_abdominal_organs",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1551,
            "mstep_init_from": f"{checkpoint_root}/CADS_series/Dataset551_Totalseg251",
            "mstep_reason": "CADS is exposed as an nnUNet-style checkpoint family and covers many abdominal structures; fine-tuning requires its complete training state/checkpoint layout.",
        },
        "moose": {
            "trainable": "yes_if_full_nnunet_training_state_present",
            "mstep_role": "conditional_trainable_candidate_for_moose_tasks",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1888,
            "mstep_init_from": f"{checkpoint_root}/MOOSE_series/Dataset888_Cardiac",
            "mstep_reason": "MOOSE is treated as nnUNet-style only when the corresponding full dataset/checkpoint is present; the current lightweight export exposes limited runnable structure.",
        },
        "moose3_0": {
            "trainable": "yes_if_full_nnunet_training_state_present",
            "mstep_role": "conditional_trainable_candidate_for_moose3_tasks",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1889,
            "mstep_init_from": f"{checkpoint_root}/MOOSE_series",
            "mstep_reason": "MOOSE3.0 can be an M-step target only if the actual nnUNet-compatible training state for the relevant organ task is available.",
        },
        "vsmtrans": {
            "trainable": "yes_if_full_nnunet_training_state_present",
            "mstep_role": "trainable_candidate_for_abdominal_organs",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1001,
            "mstep_init_from": f"{checkpoint_root}/VSmTrans/nnUNet_results/Dataset001_BDMAP",
            "mstep_reason": "VSmTrans is routed to abdominal organs; fine-tuning is possible only if its nnUNet-compatible training state and environment are available.",
        },
        "nnunet_private": {
            "trainable": "yes_if_full_nnunet_training_state_present",
            "mstep_role": "private_abdominal_organ_trainable_backend",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1224,
            "mstep_init_from": f"{checkpoint_root}/nnUNet_private/Dataset224_AbdomenAtlas1.1",
            "mstep_reason": "Private AbdomenAtlas nnUNet is a natural trainable backend for organ-mask refinement if the lab's full checkpoint/training layout is mounted.",
        },
        "saros_nnunet": {
            "trainable": "yes_if_full_nnunet_training_state_present",
            "mstep_role": "conditional_trainable_body_region_backend",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 2345,
            "mstep_init_from": f"{checkpoint_root}/nnUNet_private/Dataset1345_SAROS",
            "mstep_reason": "SAROS is a private nnUNet-style model but not the first-choice PanTS pancreas target.",
        },
        "atlasnet": {
            "trainable": "yes_if_atlasnet_nnunet_training_state_present",
            "mstep_role": "conditional_trainable_abdominal_25_class_backend",
            "mstep_backend": "nnunetv2_finetune",
            "mstep_dataset_id": 1002,
            "mstep_init_from": f"{checkpoint_root}/ATLAS-Net",
            "mstep_reason": "ATLAS-Net is described as an nnUNet v2 abdominal model; fine-tuning requires downloaded weights and nnUNet-compatible training files.",
        },
    }
    non_trainable = {
        "totalsegmentator": {
            "trainable": "conditional_public_nnunet_recipe",
            "mstep_role": "public_baseline_and_optional_totalseg_style_training_backend",
            "mstep_backend": "totalseg_public_nnunet",
            "mstep_dataset_id": 2101,
            "mstep_init_from": "third_party/TotalSegmentator-master",
            "mstep_reason": "TotalSegmentator is a public E-step baseline and has a bundled public nnUNet training recipe under resources/train_nnunet.md and train_nnunet.sh. This supports a TotalSegmentator-style nnUNet M-step plan, but it does not fully reproduce the released TotalSegmentator v2 model because the official v2 training used additional non-public data.",
        },
        "vista3d": {
            "trainable": "conditional_monai_bundle_finetune",
            "mstep_role": "foundation_candidate_and_optional_monai_bundle_finetune_backend",
            "mstep_backend": "monai_bundle_finetune",
            "mstep_dataset_id": 3101,
            "mstep_init_from": "third_party/VISTA3D-Inference-Pipeline-master",
            "mstep_reason": "VISTA3D is integrated as a high-resource foundation segmentation candidate and the bundled pipeline contains MONAI bundle training/fine-tuning/continual-learning configs (configs/train.json, train_continual.json, multi_gpu_train.json, scripts/trainer.py). Fine-tuning requires a VISTA3D checkpoint, datalist, MONAI environment, and sufficient GPU memory; this is not a from-scratch reproduction of the original foundation model.",
        },
        "unest": {
            "trainable": "external_script_required",
            "mstep_role": "kidney_substructure_inference_candidate",
            "mstep_backend": "external_training_required",
            "mstep_reason": "UNEST is task-specific for renal structures. The lightweight export contains a run script, not a complete training wrapper.",
        },
        "mock_seg": {
            "trainable": "no",
            "mstep_role": "dry_run_only",
            "mstep_backend": "none",
            "mstep_reason": "Synthetic mock model for pipeline dry-runs only.",
        },
    }
    template = {
        "trainable": "unknown_template_only",
        "mstep_role": "template_only_until_real_script_arrives",
        "mstep_backend": "none",
        "mstep_reason": "This model family appears in class_checkpoint_map.xlsx, but no complete runnable inference/training script or checkpoint folder was available in the lightweight export.",
    }
    if key in trainable_nnunet:
        return trainable_nnunet[key]
    if key in non_trainable:
        return non_trainable[key]
    return template

def apply_mstep_metadata(entry: dict[str, Any], key: str, checkpoint_root: str = "checkpoints") -> dict[str, Any]:
    entry.update(_mstep_metadata_for_model(key, checkpoint_root))
    return entry


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    """Read the first worksheet of a simple .xlsx using OOXML only.

    This avoids adding a hard dependency on openpyxl while still making the
    registry builder usable on fresh machines.
    """
    shared: list[str] = []
    with zipfile.ZipFile(path) as zf:
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", XLSX_NS):
                texts = [t.text or "" for t in si.findall(".//a:t", XLSX_NS)]
                shared.append("".join(texts))
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            sheet_name = next(n for n in zf.namelist() if n.startswith("xl/worksheets/sheet"))
        root = ET.fromstring(zf.read(sheet_name))

    rows_by_idx: dict[int, dict[int, str]] = defaultdict(dict)
    for row in root.findall(".//a:sheetData/a:row", XLSX_NS):
        r_idx = int(row.attrib.get("r", "1"))
        for c in row.findall("a:c", XLSX_NS):
            ref = c.attrib.get("r", "A1")
            col_letters = re.sub(r"\d", "", ref)
            col_idx = 0
            for ch in col_letters:
                col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
            col_idx -= 1
            cell_type = c.attrib.get("t")
            v = c.find("a:v", XLSX_NS)
            value = "" if v is None or v.text is None else v.text
            if cell_type == "s" and value:
                value = shared[int(value)]
            rows_by_idx[r_idx][col_idx] = str(value).strip()
    if not rows_by_idx:
        return []
    max_col = max((max(cols.keys()) for cols in rows_by_idx.values() if cols), default=-1)
    rows: list[list[str]] = []
    for r in range(1, max(rows_by_idx.keys()) + 1):
        rows.append([rows_by_idx.get(r, {}).get(c, "") for c in range(max_col + 1)])
    return rows


def parse_checkpoint_map(path: str | Path) -> dict[str, Any]:
    xlsx = Path(path).resolve()
    rows = _read_xlsx_rows(xlsx)
    if not rows:
        return {"status": "failed", "reason": "empty workbook", "path": str(xlsx), "organs": [], "models": {}}
    header = [_norm_key(x) for x in rows[0]]
    try:
        organ_idx = header.index("anatomical_structures")
    except ValueError:
        organ_idx = 0
    try:
        ckpt_idx = header.index("checkpoint")
    except ValueError:
        ckpt_idx = 1 if len(header) > 1 else 0

    organ_records: list[dict[str, Any]] = []
    coverage: dict[str, list[str]] = defaultdict(list)
    for row in rows[1:]:
        if organ_idx >= len(row):
            continue
        organ_raw = row[organ_idx].strip()
        checkpoint_raw = row[ckpt_idx].strip() if ckpt_idx < len(row) else ""
        if not organ_raw:
            continue
        organ = _norm_key(organ_raw)
        families = _split_checkpoint_families(checkpoint_raw)
        model_keys = [_family_key(f) for f in families]
        record = {
            "organ": organ,
            "organ_display": organ_raw.strip(),
            "checkpoint": checkpoint_raw,
            "candidate_models": model_keys,
            "candidate_model_names": families,
        }
        organ_records.append(record)
        for mk in model_keys:
            coverage[mk].append(organ)
    return {
        "status": "success",
        "source_xlsx": str(xlsx),
        "num_organs": len(organ_records),
        "num_model_families": len(coverage),
        "organs": organ_records,
        "coverage": {k: sorted(set(v)) for k, v in sorted(coverage.items())},
    }


def _default_model_entry(key: str, organs: list[str], checkpoint_root: str = "checkpoints") -> dict[str, Any]:
    family_name = key
    base: dict[str, Any] = {
        "name": family_name,
        "type": "segmentation",
        "source": "class_checkpoint_map.xlsx",
        "covered_organs": sorted(set(organs)),
        "status": "template",
        "output_layout": "case_id/segmentations/*.nii.gz",
        "notes": "Command template may need editing after the real checkpoint folder and original inference script are available.",
    }
    if key == "totalsegmentator":
        base.update({
            "name": "TotalSegmentator",
            "status": "ready_if_installed",
            "runner": "builtin_totalsegmentator",
            "command_template": "TotalSegmentator -i {image} -o {output}",
            "private_checkpoint": False,
        })
    elif key == "vista3d":
        base.update({
            "name": "VISTA3D Inference Pipeline",
            "runner": "command_template",
            "status": "ready_if_vista3d_env_and_checkpoint_present",
            "checkpoint_path": "third_party/VISTA3D-Inference-Pipeline-master",
            "label_map_path": "third_party/VISTA3D-Inference-Pipeline-master/label_mappings/label_dict_127_abdomenAtlas3-1.json",
            "command_template": "python scripts/vista3d_predict_and_split.py --image {image} --output {case_output} --vista-root {checkpoint_path} --label-map {label_map_path}",
            "private_checkpoint": False,
            "source_code_path": "third_party/VISTA3D-Inference-Pipeline-master",
            "notes": "Full source is bundled under third_party/VISTA3D-Inference-Pipeline-master. Real inference requires MONAI bundle/checkpoint download and a GPU environment; output is standardized to segmentations/*.nii.gz.",
        })
    elif key == "epai_20250421":
        base.update({
            "name": "ePAI qchen76 2025-04-21 / Dataset1017 25-class abdomen-pancreas model",
            "runner": "command_template",
            "status": "ready_if_checkpoint_folder_present",
            "covered_organs": [
                "aorta", "adrenal_gland_left", "adrenal_gland_right", "common_bile_duct",
                "celiac_aa", "colon", "duodenum", "gall_bladder", "postcava",
                "kidney_left", "kidney_right", "liver", "pancreas", "pancreatic_duct",
                "superior_mesenteric_artery", "intestine", "spleen", "stomach",
                "veins", "renal_vein_left", "renal_vein_right", "cbd_stent",
                "pancreatic_pdac", "pancreatic_cyst", "pancreatic_pnet",
            ],
            "supported_organ_aliases": {
                "celiac_aa_celiac_artery": "celiac_aa",
                "inferior_vena_cava": "postcava",
                "small_intestine": "intestine",
                "portal_splenic_veins": "veins",
                "portal_vein_and_splenic_vein": "veins",
            },
            "checkpoint_path": f"{checkpoint_root}/nnUNet_private/nnUNet_private/Dataset1339_ePAI/nnUNetTrainer__nnUNetPlans__3d_fullres",
            "dataset_json_path": f"{checkpoint_root}/nnUNet_private/nnUNet_private/Dataset1339_ePAI/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json",
            "dataset_id": 1339,
            "trainer": "nnUNetTrainer",
            "plans": "nnUNetPlans",
            "folds": "all",
            "command_template": "python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_root} --dataset-json {dataset_json_path} --model-folder {checkpoint_path} --workdir third_party/ePAI-main/train --trainer {trainer} --plans {plans} --configuration 3d_fullres --folds {folds} --checkpoint-name checkpoint_final.pth --output-label-mode all_organs",
            "private_checkpoint": True,
            "source_code_path": "third_party/ePAI-main",
            "all_organ_patch": "scripts/patch_epai_enable_all_organs.py",
            "notes": "Verified against qchen76_2025_0421.tar.gz. Dataset1017_ePAI_3MM has background plus 25 foreground organ/duct/tumor labels. ePAI source is bundled and patched so all-organ output is the default; pancreas/tumor-only filtering can be restored with --output_label_mode pancreas_only.",
        })
    elif key == "atlasnet":
        base.update({
            "name": "ATLAS-Net nnUNet v2 abdominal 25-class model",
            "runner": "command_template",
            "status": "ready_if_atlasnet_checkpoint_present",
            "checkpoint_path": f"{checkpoint_root}/ATLAS-Net",
            "label_map_path": "configs/atlasnet_label_map.json",
            "command_template": "python scripts/atlasnet_predict_and_split.py --image {image} --output {case_output} --atlas-root {checkpoint_path} --label-map {label_map_path}",
            "private_checkpoint": False,
            "source_code_path": "docs/ATLAS-Net_model_card_from_teacher.docx",
            "notes": "ATLAS-Net model card from teacher's 'Another version of ShapeKit.docx' is integrated. It covers 25 abdominal organ/tumor labels and expects Linux/CUDA/nnUNet v2 plus downloaded ATLAS-Net weights.",
        })
    elif key == "cads":
        base.update({
            "name": "CADS abdominal nnUNet Dataset551",
            "runner": "command_template",
            "status": "ready_if_checkpoint_folder_present",
            "checkpoint_path": f"{checkpoint_root}/CADS_series/CADS_series",
            "dataset_json_path": f"{checkpoint_root}/CADS_series/CADS_series/Dataset551_Totalseg251/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/dataset.json",
            "dataset_id": 551,
            "trainer": "nnUNetTrainerNoMirroring",
            "plans": "nnUNetResEncUNetLPlans",
            "folds": "all",
            "command_template": "python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_path} --dataset-json {dataset_json_path} --trainer {trainer} --plans {plans} --configuration 3d_fullres --folds {folds}",
            "private_checkpoint": True,
            "notes": "Uses teacher Drive CADS_series/run_CADS.sh structure; Dataset551 covers abdomen organs relevant to PanTS.",
        })
    elif key in {"moose", "moose3_0"}:
        base.update({
            "name": "MOOSE available cardiac/vascular nnUNet Dataset888",
            "runner": "command_template",
            "status": "partial_ready_if_checkpoint_folder_present",
            "checkpoint_path": f"{checkpoint_root}/MOOSE_series/MOOSE_series",
            "dataset_json_path": f"{checkpoint_root}/MOOSE_series/MOOSE_series/Dataset888_Cardiac/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json",
            "dataset_id": 888,
            "trainer": "nnUNetTrainerNoMirroring",
            "plans": "nnUNetPlans",
            "folds": "all",
            "command_template": "python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_path} --dataset-json {dataset_json_path} --trainer {trainer} --plans {plans} --configuration 3d_fullres --folds {folds}",
            "private_checkpoint": True,
            "notes": "The lightweight export only exposed MOOSE Dataset666/888. Dataset888 is useful for aorta/vascular structures; not a full abdominal organ model.",
        })
    elif key == "vsmtrans":
        base.update({
            "name": "VSmTrans BDMAP nnUNet Dataset001",
            "runner": "command_template",
            "status": "ready_if_checkpoint_folder_present",
            "checkpoint_path": f"{checkpoint_root}/VSmTrans/VSmTrans/nnUNet_results",
            "dataset_json_path": f"{checkpoint_root}/VSmTrans/VSmTrans/nnUNet_results/Dataset001_BDMAP/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json",
            "dataset_id": 1,
            "trainer": "nnUNetTrainer",
            "plans": "nnUNetPlans",
            "folds": "0",
            "command_template": "python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_path} --dataset-json {dataset_json_path} --trainer {trainer} --plans {plans} --configuration 3d_fullres --folds {folds}",
            "private_checkpoint": True,
            "source_code_path": "third_party/VSmTrans_lightweight",
            "notes": "VSmTrans README, predict_script.py, bounding_boxes.py and nnUNet lightweight source are included. Real inference requires the actual VSmTrans/nnUNet checkpoint folder.",
        })
    elif key == "nnunet_private":
        base.update({
            "name": "Private AbdomenAtlas nnUNet Dataset224",
            "runner": "command_template",
            "status": "ready_if_checkpoint_folder_present",
            "checkpoint_path": f"{checkpoint_root}/nnUNet_private/nnUNet_private",
            "dataset_json_path": f"{checkpoint_root}/nnUNet_private/nnUNet_private/Dataset224_AbdomenAtlas1.1/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/dataset.json",
            "dataset_id": 224,
            "trainer": "nnUNetTrainer",
            "plans": "nnUNetResEncUNetLPlans",
            "folds": "all",
            "command_template": "python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_path} --dataset-json {dataset_json_path} --trainer {trainer} --plans {plans} --configuration 3d_fullres --folds {folds}",
            "private_checkpoint": True,
            "notes": "Private AbdomenAtlas organ model from Drive export; useful as an organ-mask candidate.",
        })
    elif key == "saros_nnunet":
        base.update({
            "name": "Private SAROS nnUNet Dataset1345",
            "runner": "command_template",
            "status": "ready_if_checkpoint_folder_present",
            "checkpoint_path": f"{checkpoint_root}/nnUNet_private/nnUNet_private",
            "dataset_json_path": f"{checkpoint_root}/nnUNet_private/nnUNet_private/Dataset1345_SAROS/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/dataset.json",
            "dataset_id": 1345,
            "trainer": "nnUNetTrainer",
            "plans": "nnUNetResEncUNetLPlans",
            "folds": "all",
            "command_template": "python scripts/nnunetv2_predict_and_split.py --image {image} --output {case_output} --dataset-id {dataset_id} --nnunet-results {checkpoint_path} --dataset-json {dataset_json_path} --trainer {trainer} --plans {plans} --configuration 3d_fullres --folds {folds}",
            "private_checkpoint": True,
            "notes": "SAROS body-region model; not the first choice for PanTS organ refinement.",
        })
    elif key == "unest":
        base.update({
            "name": "UNEST renalStructures segmentation",
            "runner": "command_template",
            "status": "ready_if_unest_checkpoint_present",
            "checkpoint_path": "third_party/UNEST_renalStructures_lightweight",
            "command_template": "python scripts/unest_predict_and_split.py --image {image} --output {case_output} --unest-root {checkpoint_path}",
            "private_checkpoint": True,
            "source_code_path": "third_party/UNEST_renalStructures_lightweight",
            "notes": "UNEST run_UNEST.sh and configs from the teacher Drive lightweight export are bundled. Real inference requires the actual UNEST model/checkpoint environment.",
        })
    elif key in {"dap", "goacc"}:
        base.update({
            "name": key,
            "runner": "command_template",
            "checkpoint_path": f"{checkpoint_root}/{key}",
            "command_template": "python wrappers/{model_key}_infer.py --image {image} --output {output} --checkpoint {checkpoint_path}",
            "private_checkpoint": True,
            "notes": "Template retained; the lightweight export did not include a runnable script for this family.",
        })
    else:
        base.update({
            "runner": "command_template",
            "checkpoint_path": f"{checkpoint_root}/{key}",
            "command_template": "python wrappers/{model_key}_infer.py --image {image} --output {output} --checkpoint {checkpoint_path}",
            "private_checkpoint": True,
        })
    return apply_mstep_metadata(base, key, checkpoint_root)


def build_registry_dict(checkpoint_map: str | Path, checkpoint_root: str = "checkpoints") -> dict[str, Any]:
    parsed = parse_checkpoint_map(checkpoint_map)
    if parsed.get("status") != "success":
        return parsed
    models = {k: _default_model_entry(k, v, checkpoint_root) for k, v in parsed["coverage"].items()}
    # Add practical local/demo model so the full workflow can be smoke-tested without GPU checkpoints.
    models["mock_seg"] = apply_mstep_metadata({
        "name": "Local mock segmentation wrapper",
        "type": "segmentation",
        "runner": "command_template",
        "status": "ready",
        "private_checkpoint": False,
        "covered_organs": ["liver", "pancreas", "spleen", "kidney_left", "kidney_right", "aorta", "postcava"],
        "command_template": "python third_party/mock_model/mock_seg_infer.py --image {image} --output {output} --case-id {case_id}",
        "output_layout": "case_id/segmentations/*.nii.gz",
        "notes": "Synthetic smoke-test backend only. Use it to validate the CLI loop before real GPU checkpoints are available.",
    }, "mock_seg", checkpoint_root)
    # Teacher Drive has a top-level nnUNet_private folder. Add it explicitly even
    # when the Excel class map only exposes specific private datasets such as SAROS/ePAI.
    models.setdefault("nnunet_private", _default_model_entry("nnunet_private", [
        "pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"
    ], checkpoint_root))
    # ATLAS-Net came from the teacher's 'Another version of ShapeKit' model card,
    # not always from the checkpoint map rows. Keep it as a first-class candidate.
    models.setdefault("atlasnet", _default_model_entry("atlasnet", [
        "aorta", "adrenal_gland_left", "adrenal_gland_right", "common_bile_duct", "colon", "duodenum", "gall_bladder", "inferior_vena_cava", "kidney_left", "kidney_right", "liver", "pancreas", "pancreatic_duct", "superior_mesenteric_artery", "small_intestine", "spleen", "stomach", "portal_vein_and_splenic_vein", "renal_vein_left", "renal_vein_right", "pancreatic_pdac", "pancreatic_cyst", "pancreatic_pnet"
    ], checkpoint_root))
    organ_to_models: dict[str, list[str]] = {}
    for rec in parsed["organs"]:
        models_for_organ = list(rec["candidate_models"])
        if rec["organ"] in {"liver", "pancreas", "spleen", "kidney_left", "kidney_right", "aorta", "postcava"}:
            if "mock_seg" not in models_for_organ:
                models_for_organ.append("mock_seg")
        organ_to_models[rec["organ"]] = models_for_organ
    registry = {
        "schema_version": "1.0",
        "generated_from": str(Path(checkpoint_map).resolve()),
        "checkpoint_root": checkpoint_root,
        "models": dict(sorted(models.items())),
        "organ_to_models": dict(sorted(organ_to_models.items())),
        "notes": [
            "Do not commit private checkpoints to a public repository.",
            "Edit command_template for private models after inspecting each original inference script.",
            "All wrappers normalize outputs to case_id/segmentations/*.nii.gz.",
        ],
    }
    return {"status": "success", "parsed": parsed, "registry": registry}


def write_registry(checkpoint_map: str | Path, output_yaml: str | Path, output_csv: str | Path | None = None, checkpoint_root: str = "checkpoints") -> dict[str, Any]:
    built = build_registry_dict(checkpoint_map, checkpoint_root)
    if built.get("status") != "success":
        return built
    out = Path(output_yaml).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        out.write_text(yaml.safe_dump(built["registry"], sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        out.write_text(json.dumps(built["registry"], indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = None
    if output_csv:
        csv_path = Path(output_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["organ", "organ_display", "checkpoint", "candidate_models", "candidate_model_names"])
            writer.writeheader()
            for rec in built["parsed"]["organs"]:
                writer.writerow({
                    "organ": rec["organ"],
                    "organ_display": rec["organ_display"],
                    "checkpoint": rec["checkpoint"],
                    "candidate_models": ",".join(rec["candidate_models"]),
                    "candidate_model_names": ",".join(rec["candidate_model_names"]),
                })
    return {
        "stage": "build_registry",
        "status": "success",
        "registry_yaml": str(out),
        "parsed_csv": str(csv_path) if csv_path else None,
        "num_organs": built["parsed"].get("num_organs"),
        "num_model_families": built["parsed"].get("num_model_families"),
        "models": sorted(built["registry"]["models"].keys()),
        "sample_organs": built["parsed"]["organs"][:10],
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"model registry not found: {p}")
    text = p.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text)
    raise ImportError(
        "PyYAML is required to load the model registry. "
        "Install it with: pip install pyyaml>=6.0"
    )


def get_model_entry(registry: dict[str, Any], model_key: str) -> dict[str, Any]:
    models = registry.get("models", {})
    if model_key not in models:
        raise KeyError(f"model '{model_key}' not found in registry. Available: {', '.join(sorted(models)[:50])}")
    entry = dict(models[model_key])
    entry["model_key"] = model_key
    return entry


def candidate_models_for_organs(registry: dict[str, Any], organs: list[str], include_mock: bool = False) -> dict[str, list[str]]:
    """Return candidate model keys for organs.

    Prefer the formal organ router when available because it resolves routing
    tokens such as CADS551..CADS559 to concrete runnable model keys.  The older
    registry ``organ_to_models`` map was generated from the checkpoint workbook
    and can contain coarse family names such as ``cads`` that are no longer
    valid runnable registry keys.
    """
    mapping = registry.get("organ_to_models", {})
    models = registry.get("models", {})
    result: dict[str, list[str]] = {}

    routed_by_organ: dict[str, list[str]] = {}
    try:
        from .organ_router import route_organs

        routed = route_organs(organs)
        for organ, candidates in (routed.get("ranked_candidates", {}) or {}).items():
            keys: list[str] = []
            seen_routed: set[str] = set()
            for item in candidates or []:
                model_key = item.get("model_key")
                if (
                    model_key
                    and model_key in models
                    and model_key not in seen_routed
                    and (include_mock or model_key != "mock_seg")
                ):
                    keys.append(model_key)
                    seen_routed.add(model_key)
            routed_by_organ[_norm_key(organ)] = keys
    except Exception:
        routed_by_organ = {}

    for organ in organs:
        key = _norm_key(organ)
        candidates = list(routed_by_organ.get(key) or [])
        if not candidates:
            candidates = [
                m for m in list(mapping.get(key, []))
                if m in models and (include_mock or m != "mock_seg")
            ]
        seen = set(candidates)
        for model_key, entry in models.items():
            covered = {_norm_key(x) for x in entry.get("covered_organs", [])}
            aliases = {_norm_key(k): _norm_key(v) for k, v in (entry.get("supported_organ_aliases") or {}).items()}
            alias_target = aliases.get(key)
            if (key in covered or alias_target in covered) and model_key not in seen and (include_mock or model_key != "mock_seg"):
                candidates.append(model_key)
                seen.add(model_key)
        result[key] = candidates
    return result


def model_inventory(registry: dict[str, Any], include_mock: bool = False) -> dict[str, Any]:
    """Summarize teacher-provided/referenced model families and trainability."""
    rows = []
    for key, entry in sorted(registry.get("models", {}).items()):
        if key == "mock_seg" and not include_mock:
            continue
        rows.append({
            "model_key": key,
            "name": entry.get("name", key),
            "status": entry.get("status"),
            "runner": entry.get("runner"),
            "trainable": entry.get("trainable", "unknown"),
            "mstep_backend": entry.get("mstep_backend", "unknown"),
            "mstep_role": entry.get("mstep_role", "unknown"),
            "mstep_reason": entry.get("mstep_reason", ""),
            "source": entry.get("source_code_path") or entry.get("source") or "class_checkpoint_map.xlsx",
            "private_checkpoint": entry.get("private_checkpoint", False),
            "covered_organs_count": len(entry.get("covered_organs", []) or []),
        })
    trainable = [r for r in rows if str(r["trainable"]).startswith("yes") or str(r["trainable"]).startswith("conditional")]
    conditional = [r for r in rows if "if_" in str(r["trainable"]) or str(r["trainable"]).startswith("conditional")]
    not_trainable = [r for r in rows if str(r["mstep_backend"]) in {"none", "external_training_required"}]
    return {
        "stage": "model_inventory",
        "status": "success",
        "num_models_excluding_mock": len(rows),
        "num_trainable_or_finetunable_if_checkpoint_present": len(trainable),
        "num_not_trainable_in_current_project": len(not_trainable),
        "models": rows,
    }

def recommend_primary_models_for_organs(registry: dict[str, Any], organs: list[str]) -> dict[str, Any]:
    """Rule-based task/organ routing for selected-model-aware EM."""
    recs = {}
    routed_by_organ: dict[str, list[str]] = {}
    try:
        from .organ_router import route_organs

        routed = route_organs(organs)
        models = registry.get("models", {})
        for organ, candidates in (routed.get("ranked_candidates", {}) or {}).items():
            keys: list[str] = []
            seen: set[str] = set()
            for item in candidates or []:
                model_key = item.get("model_key")
                if model_key and model_key in models and model_key not in seen:
                    keys.append(model_key)
                    seen.add(model_key)
            routed_by_organ[_norm_key(organ)] = keys
    except Exception:
        routed_by_organ = {}

    for organ in organs:
        o = _norm_key(organ)
        routed_candidates = routed_by_organ.get(o) or []
        if routed_candidates:
            primary = routed_candidates[0]
            auxiliaries = routed_candidates[1:]
        elif o in {"pancreas", "pancreatic_duct", "pancreatic_pdac", "pancreatic_cyst", "pancreatic_pnet", "pancreatic_lesion", "common_bile_duct", "cbd_stent", "superior_mesenteric_artery", "celiac_aa", "celiac_aa_celiac_artery", "renal_vein_left", "renal_vein_right", "veins", "portal_vein_and_splenic_vein", "portal_splenic_veins"}:
            primary = "epai_20250421"
            auxiliaries = ["atlasnet", "vsmtrans", "cads551", "moose888", "totalsegmentator", "vista3d"]
        elif o in {"liver", "spleen", "stomach", "duodenum", "colon", "kidney_left", "kidney_right", "gall_bladder", "intestine"}:
            primary = "vsmtrans"
            auxiliaries = ["epai_20250421", "cads551", "moose888", "atlasnet", "totalsegmentator", "vista3d", "nnunet_private"]
        elif o in {"aorta", "postcava", "inferior_vena_cava"}:
            primary = "cads551"
            auxiliaries = ["epai_20250421", "vista3d", "moose888", "atlasnet", "vsmtrans", "totalsegmentator"]
        elif o in {"kidney_cortex", "kidney_medulla"}:
            primary = "unest"
            auxiliaries = ["nnunet_private", "cads551", "vsmtrans"]
        else:
            candidates = candidate_models_for_organs(registry, [o]).get(o, [])
            primary = candidates[0] if candidates else None
            auxiliaries = candidates[1:] if len(candidates) > 1 else []
        entry = registry.get("models", {}).get(primary, {}) if primary else {}
        recs[o] = {
            "primary_model": primary,
            "auxiliary_models": [m for m in auxiliaries if m in registry.get("models", {})],
            "mstep_target_model": primary if entry.get("mstep_backend") not in {None, "none", "external_training_required"} else None,
            "trainable": entry.get("trainable"),
            "mstep_backend": entry.get("mstep_backend"),
            "rationale": entry.get("mstep_reason", "No model-specific rationale available."),
        }
    return {"stage": "task_model_routing", "status": "success", "organs": organs, "routing": recs}
