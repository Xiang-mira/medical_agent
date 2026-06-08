# 373-Organ Auto Fine-Label System and VoxTell Student Implementation

## Formal Target

The formal target is 373 exact organs, with
`configs/student_3d_prompt_target_organs.json` as the source of truth.

Count policy:

- 384 = global label space.
- 8 = SAROS coarse-label organs skipped by policy.
- 3 = organs with no enabled route.
- 373 = current accepted exact 3D prompt targets.
- 377, 358, and VISTA3D 127 are historical or side target spaces and are not the mainline target.

Use:

```bash
python run_medai_cli.py --json validate-373-target
python scripts/audit_373_organ_routing.py
```

## Terminology

The project builds an auto-generated fine-label dataset from scratch. It should
not be described as expert ground truth unless expert verification is added
later.

Allowed mainline statuses:

- `machine_label_candidate`
- `auto_fine_label_candidate`
- `auto_fine_label_accepted`
- `unresolved`
- `expert_verified` only for future expert-confirmed masks

Each final selected mask gets a label passport with maturity, reliability grade,
training weight, QC, ShapeKit, LabelCritic, and source metadata.

## VoxTell Status and I/O

Current status: VoxTell mini train and mini inference plumbing has run
successfully, but this does not prove quality. A prior mini inference produced
an empty mask (`mask_voxels: 0`), so empty-mask QC, prompt wording, orientation,
spacing, and threshold behavior must be tracked.

Official VoxTell input:

- 3D NIfTI CT.
- VoxTell model directory with `plans.json` and `fold_0/checkpoint_final.pth`.
- Free-text prompts such as `liver`, `spleen`, `right kidney`.
- Qwen3-Embedding-4B text encoder.

Official VoxTell output:

- One binary mask per prompt named like `<input_stem>_<prompt>.nii.gz`.
- Python API output shape `(num_prompts, X, Y, Z)`.
- `--save-combined` creates a multi-label file, but overlapping prompt masks are overwritten by later prompts.

Project output:

- One binary mask per organ:
  `student_predictions/<case_id>/<organ>.nii.gz`.
- Combined multi-label VoxTell output is not used formally.
- `voxtell_student_result.json` records official outputs, standardized outputs,
  per-organ status, voxel count, bbox, spacing, batch commands, retry queue, and
  empty-mask flags.

## Main Commands

VoxTell official-style/project wrapper dry run:

```bash
python run_medai_cli.py --json voxtell-student-segment \
  --ct-image data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz \
  --output-folder outputs/voxtell_student_dryrun \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --prompts liver,spleen,kidney_left,kidney_right \
  --prompt-batch-size 2 \
  --dry-run
```

Build VoxTell prompt-student manifest:

```bash
python run_medai_cli.py --json voxtell-student-manifest \
  --cases-root outputs/run_pants50_round1/annotation_versions \
  --output-manifest outputs/run_pants50_round1/voxtell_prompt_student_manifest.json \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --case-list data_manifest/case_list_50_tumor.csv \
  --require-images
```

Train project-specific VoxTell student:

```bash
python scripts/train_voxtell_prompt_student.py \
  --manifest outputs/run_pants50_round1/voxtell_prompt_student_manifest.json \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --text-encoding-model checkpoints/Qwen/Qwen3-Embedding-4B \
  --output-dir outputs/round1/mstep \
  --freeze-encoder
```

Build dashboards:

```bash
python run_medai_cli.py --json auto-fine-label-dashboard \
  --run-output outputs/run_pants50_round1 \
  --student-summary outputs/round1/student_predictions/student_inference_summary.json \
  --failure-json outputs/round1/failure_mining/student_failure_cases.json
```

## Teacher-Facing Answers

- We have run VoxTell mini train/inference plumbing, but have not yet proven
  373-organ quality.
- VoxTell is a 3D prompt-based model: CT + text prompt produces one 3D binary
  mask per prompt.
- It differs from fixed label-id multi-class models; therefore we standardize
  output filenames and keep per-organ binary masks.
- Student predictions are candidates in Round2, never automatic replacements.
- Without expert labels, we report pseudo-consistency, consensus, QC,
  ShapeKit, LabelCritic, and stability signals, not true expert accuracy.

