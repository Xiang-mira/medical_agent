# Hugging Face Model Release Guide

HF repo: https://huggingface.co/Xiang-mira/MedIA-Agentic-AI

This guide is for internal research collaboration. The repository is public only to make lab collaboration easy from new GPU/HPC environments; do not advertise JHU/private assets externally. The GitHub repository keeps code, wrappers, pipelines, README files, and configuration templates. Hugging Face stores model assets and large files.

## Project Purpose

This release lets lab teammates move the MedIA segmentation agents to a new GPU or HPC environment without manually copying checkpoints from local machines. Use this Hugging Face repository as the shared model-asset root and the GitHub repository as the code, pipeline, and configuration-template root.

## Included Models

The current HF release includes 23 teacher models and 1 student model. `mock_seg` and `epai_finetuned` are intentionally excluded from this release.

| Model key | Role | Backend | HF path | Checkpoint | Supported organs | Expected GPU memory | Sensitivity level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cads551` | teacher | `nnunetv2` | `teacher_models/cads551` | `fold_all/checkpoint_final.pth` | 17 organs: spleen, kidney_right, kidney_left, gallbladder, liver, stomach, ... | 8-16 GB | `internal` |
| `cads552` | teacher | `nnunetv2` | `teacher_models/cads552` | `fold_all/checkpoint_final.pth` | 24 organs: vertebrae_L5, vertebrae_L4, vertebrae_L3, vertebrae_L2, vertebrae_L1, vertebrae_T12, ... | 8-16 GB | `internal` |
| `cads553` | teacher | `nnunetv2` | `teacher_models/cads553` | `fold_all/checkpoint_final.pth` | 18 organs: esophagus, trachea, heart_myocardium, heart_atrium_left, heart_ventricle_left, heart_atrium_right, ... | 8-16 GB | `internal` |
| `cads554` | teacher | `nnunetv2` | `teacher_models/cads554` | `fold_all/checkpoint_final.pth` | 21 organs: humerus_left, humerus_right, scapula_left, scapula_right, clavicula_left, clavicula_right, ... | 8-16 GB | `internal` |
| `cads555` | teacher | `nnunetv2` | `teacher_models/cads555` | `fold_all/checkpoint_final.pth` | 24 organs: rib_left_1, rib_left_2, rib_left_3, rib_left_4, rib_left_5, rib_left_6, ... | 8-16 GB | `internal` |
| `cads556` | teacher | `nnunetv2` | `teacher_models/cads556` | `fold_all/checkpoint_final.pth` | 15 organs: spinal_canal, larynx, heart, bowel_bag, sigmoid, rectum, ... | 8-16 GB | `internal` |
| `cads557` | teacher | `nnunetv2` | `teacher_models/cads557` | `fold_all/checkpoint_final.pth` | 9 organs: white matter, gray matter, csf, scalp, eye balls, compact bone, ... | 8-16 GB | `internal` |
| `cads558` | teacher | `nnunetv2` | `teacher_models/cads558` | `fold_all/checkpoint_final.pth` | 29 organs: OAR_A_Carotid_L, OAR_A_Carotid_R, OAR_Arytenoid, OAR_Bone_Mandible, OAR_Brainstem, OAR_BuccalMucosa, ... | 8-16 GB | `internal` |
| `cads559` | teacher | `nnunetv2` | `teacher_models/cads559` | `fold_all/checkpoint_final.pth` | 10 organs: subcutaneous_tissue, muscle, abdominal_cavity, thoracic_cavity, bones, glands, ... | 8-16 GB | `internal` |
| `moose666` | teacher | `nnunetv2` | `teacher_models/moose666` | `fold_all/checkpoint_final.pth` | 31 organs: carpal_left, carpal_right, clavicle_left, clavicle_right, femur_left, femur_right, ... | 6-12 GB | `internal` |
| `moose888` | teacher | `nnunetv2` | `teacher_models/moose888` | `fold_all/checkpoint_final.pth` | 13 organs: heart_myocardium, heart_atrium_left, heart_atrium_right, heart_ventricle_left, heart_ventricle_right, aorta, ... | 6-12 GB | `internal` |
| `nnunet_private` | teacher | `nnunetv2` | `teacher_models/nnunet_private` | `fold_all/checkpoint_final.pth` | 34 organs: aorta, gall_bladder, kidney_left, kidney_right, postcava, spleen, ... | 8-16 GB | `internal` |
| `saros_nnunet` | teacher | `nnunetv2` | `teacher_models/saros_nnunet` | `fold_all/checkpoint_final.pth` | 13 organs: subcutaneous_tissue, muscle, abdominal_cavity, thoracic_cavity, bone, parotid_glands, ... | 10-18 GB | `internal` |
| `atm` | teacher | `nnunetv2` | `teacher_models/atm` | `fold_all/checkpoint_final.pth` | airway_tree | 8-16 GB | `internal` |
| `airrc` | teacher | `nnunetv2` | `teacher_models/airrc` | `fold_all/checkpoint_final.pth` | airway_tree, airway_wall, lung_pulmonary_arteries, lung_pulmonary_veins | 8-16 GB | `internal` |
| `lvp` | teacher | `nnunetv2` | `teacher_models/lvp` | `fold_all/checkpoint_final.pth` | liver_hepatic_vein, liver_portal_vein | 10-18 GB | `internal` |
| `daps` | teacher | `nnunetv2` | `teacher_models/daps` | `fold_all/checkpoint_best.pth` | 30 organs: fat, mediastinal_tissue, gonads, uterocervix, uterus, breast_left, ... | 8-16 GB | `internal` |
| `epai_20250421` | teacher | `nnunetv2` | `teacher_models/epai_20250421` | `fold_all/checkpoint_final.pth` | 25-class abdominal organ/duct/tumor teacher; public metadata intentionally omits class semantics. | 6-12 GB | `sensitive-redacted` |
| `vsmtrans` | teacher | `nnunetv2` | `teacher_models/vsmtrans` | `fold_0/checkpoint_final.pth` | 25 organs: aorta, gall_bladder, kidney_left, kidney_right, liver, pancreas, ... | 8-16 GB | `internal` |
| `vista3d` | teacher | `vista3d` | `teacher_models/vista3d` | `models/model.pt` | 99 organs: airway, aorta, atrial_appendage_left, autochthon_left, autochthon_right, brachiocephalic_trunk, ... | 16-24 GB | `research-runtime` |
| `unest` | teacher | `unest` | `teacher_models/unest` | `models/model.pt` | kidney_cortex, kidney_medulla, kidney_pelvicalyceal_system | 8-16 GB | `research-runtime` |
| `totalsegmentator` | teacher | `external_totalsegmentator_runtime` | `teacher_models/totalsegmentator` | `null` | 121 organs: anterior_scalene_left, anterior_scalene_right, auditory_canal_left, auditory_canal_right, body, body_extremities, ... | 6-12 GB | `external-runtime` |
| `atlasnet` | teacher | `atlasnet_wrapper_over_nnunetv2` | `teacher_models/atlasnet` | `fold_all/checkpoint_final.pth` | 25 organs: adrenal_gland_left, adrenal_gland_right, aorta, cbd_stent, celiac_aa (celiac_artery), colon, ... | 8-16 GB | `public-derived` |
| `voxtell_style_student_round1` | student | `voxtell_style_3d_prompt` | `student_models/voxtell_style_student_round1` | `voxtell_finetuned_model/fold_0/checkpoint_final.pth` | 373 exact prompt-conditioned target structures; see configs/student_3d_prompt_target_organs.json. | 16-24 GB | `project-student` |

## Installation And Download

Install the code repository first, then place the Hugging Face assets under the code checkout so `$HF_ASSET_ROOT` resolves consistently on local GPU boxes and HPC jobs:

```bash
git clone https://github.com/Xiang-mira/medical_agent
cd medical_agent
pip install -e agent-harness

git lfs install
git clone https://huggingface.co/Xiang-mira/MedIA-Agentic-AI checkpoints/MedIA-Agentic-AI
export HF_ASSET_ROOT=$PWD/checkpoints/MedIA-Agentic-AI
```

Alternatively, from the same GitHub code checkout, use the Python downloader:

```bash
python examples/download_from_hf.py \
  --repo-id Xiang-mira/MedIA-Agentic-AI \
  --local-dir checkpoints/MedIA-Agentic-AI
export HF_ASSET_ROOT=$PWD/checkpoints/MedIA-Agentic-AI
```

## Input And Output

Input CT volumes should be 3D NIfTI files (`.nii.gz`). The project wrappers accept direct CT paths such as `/data/case_001/ct.nii.gz`. Raw nnUNet commands require files named like `case_001_0000.nii.gz`, but the MedIA wrappers prepare that temporary layout automatically.

Outputs follow the project contract:

```text
outputs/<case_id>/
  segmentations/*.nii.gz
  combined_labels.nii.gz       # where available
  inference_summary.json       # where available
```

## CLI Examples

ATLAS-Net teacher:

```bash
export HF_ASSET_ROOT=$PWD/checkpoints/MedIA-Agentic-AI
python scripts/atlasnet_predict_and_split.py \
  --image /data/case_001/ct.nii.gz \
  --output outputs/atlasnet_case001 \
  --atlas-root $HF_ASSET_ROOT/teacher_models/atlasnet \
  --label-map configs/atlasnet_label_map.json \
  --device cuda:0 \
  --postprocess-mode auto
```

Generic nnUNet teacher example:

```bash
export HF_ASSET_ROOT=$PWD/checkpoints/MedIA-Agentic-AI
python scripts/nnunetv2_predict_and_split.py \
  --image /data/case_001/ct.nii.gz \
  --output outputs/cads551_case001 \
  --dataset-id 551 \
  --nnunet-results $HF_ASSET_ROOT/teacher_models/cads551 \
  --dataset-json $HF_ASSET_ROOT/teacher_models/cads551/dataset.json \
  --trainer nnUNetTrainerNoMirroring \
  --plans nnUNetResEncUNetLPlans \
  --configuration 3d_fullres \
  --folds all \
  --checkpoint-name checkpoint_final.pth
```

UNEST renal teacher:

```bash
python scripts/unest_predict_and_split.py \
  --image /data/case_001/ct.nii.gz \
  --output outputs/unest_case001 \
  --unest-root $HF_ASSET_ROOT/teacher_models/unest
```

TotalSegmentator is an external runtime. Install it separately, then use the project wrapper:

```bash
python run_medai_cli.py --json infer \
  --image /data/case_001/ct.nii.gz \
  --output-folder outputs/totalseg_case001 \
  --backend totalseg
```

## HPC Submit Example

```bash
sbatch examples/hpc/medai_infer.sbatch /data/case_001/ct.nii.gz outputs/hf_case001 atlasnet
```

## Common Errors

- LFS pointer files instead of real weights: run `git lfs install` and `git lfs pull` inside the HF clone.
- Missing `dataset.json` or `plans.json`: verify `$HF_ASSET_ROOT` points to the HF asset root, not the GitHub code root.
- CUDA out of memory: start with one CT volume, lower concurrent jobs, or use a larger GPU according to `expected_gpu_memory_gb`.
- nnUNet command cannot find model: use the MedIA wrapper examples instead of raw `nnUNetv2_predict` until the local layout is confirmed.
- VISTA3D/UNEST import errors: install their MONAI/runtime dependencies before running those backend-specific wrappers.
- ePAI label names look redacted: this is intentional. Authorized users should use the internal lab label registry and GitHub runner.

## Maintenance

When adding or changing model assets, update `medai_model_manifest.yaml`, `medai_model_manifest.json`, `configs/hf_model_manifest.yaml`, `model_index.yaml`, per-model README files, and `checksums.sha256`. Do not add `mock_seg` or local experimental M-step outputs unless they become reusable HF releases.
