# VoxTell Vendor Audit

## Source

- Official repository: https://github.com/MIC-DKFZ/VoxTell
- Local vendor path: `/home/teacher1/JHU-project1/medical_agent/third_party/VoxTell`
- Intended vendor commit: `a4421c5` (`1st place on official ReXGroundingCT leaderboard`)

## Boundary

The project treats `third_party/VoxTell` as an official vendor checkout. Project-specific behavior must live in the medical-agent adapter layer, mainly `VoxTellStudent`, instead of patching official VoxTell files.

Do not directly modify these vendor paths for project integration:

- `third_party/VoxTell/voxtell/model/*`
- `third_party/VoxTell/voxtell/inference/*`
- `third_party/VoxTell/voxtell/utils/*`

## Project Adapter Responsibilities

The project adapter may handle prompt selection, prompt batching, dry-run planning, output-name standardization, confidence metadata, quality gates, EM loop integration, and manifest generation. These are not upstream VoxTell responsibilities.

## Current Integration Statement

Student inference is backed by the official VoxTell implementation, while this project adds prompt selection, confidence scoring, and quality-gated pseudo-label integration around it. Project-specific distillation is experimental unless the official VoxTell fine-tuning entrypoint is explicitly selected and passes preflight.

## Verification

Run the CPU-only vendor audit script:

```bash
python scripts/audit_voxtell_vendor.py --json-out /tmp/voxtell_vendor_audit.json
```

A clean official vendor state should report `dirty: false` for `third_party/VoxTell`.

## Official Fine-Tune Semantics

Official VoxTell `main` exposes `voxtell-finetune = voxtell.training.run_finetuning:main`. This is **official VoxTell nnU-Net encoder-transfer fine-tuning**, not prompt-conditioned VoxTell Student training. It transfers the pretrained VoxTell image encoder into a standard nnU-Net trainer and trains a multi-class segmentation decoder from scratch.

Project canonical baseline name: `official_voxtell_nnunet_encoder_baseline`. The older `official_voxtell_nnunet_encoder_finetune` name is accepted only as a temporary alias with a warning.

Do not call this simply `official_voxtell_finetune` in reports, because that sounds like full text-prompt VoxTell model fine-tuning. The main project Student trainer is `project_voxtell_prompt_distillation_student`; the older `project_distillation_experimental` name is legacy/audit-only wording.

Required official pipeline:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
voxtell-finetune DATASET_ID 3d_fullres 0 -tr VoxTellTrainer_noMirroring -pretrained_weights /path/to/voxtell_model/fold_0/checkpoint_final.pth
```

The project converter `scripts/convert_voxtell_manifest_to_nnunet.py` prepares `nnUNet_raw/DatasetXXX_Name` from A/B hard pseudo-labels only and writes `converter_audit.json`, `label_mapping.json`, `training_exclusions.json`, and `overlap_conflicts.csv`.
