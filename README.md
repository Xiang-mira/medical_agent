# MedAI Agent Loop

Registry-driven, multi-model pseudo-label refinement for 3D medical image
segmentation. The current mainline builds a strict 373-target organ label space,
runs teacher models through hierarchical ROI inference, scores and fuses
pseudo-label evidence with AutoLabelCore, and trains a VoxTell-style
prompt-conditioned 3D student across EM rounds.

> **Reporting boundary**
>
> Existing masks in the case manifests are treated as prior pseudo references or
> weak labels, not expert ground truth. Unless a separately verified expert-label
> path is configured, Dice/DSC values in this repository are
> **pseudo-consistency metrics**, not anatomical accuracy or expert-label
> performance.

## Current architecture

```text
PanTS / PAINTS CT volumes and prior pseudo references
                         |
                         v
373 exact canonical targets + organ taxonomy + model registry
                         |
                         v
Major organs: full-volume teacher inference
Child targets: parent-mask ROI inference and full-geometry restoration
                         |
                         v
ShapeKit -> structural QC -> exact-identity validation
                         |
                         v
AutoLabelCore evidence scoring and family-balanced candidate fusion
                         |
                         +----> LabelCritic bounded tie-break (supported cases)
                         |
                         v
A/B hard labels + optional C soft labels + auditable rejected/provisional labels
                         |
                         v
VoxTell-style 3D prompt student M-step
                         |
                         v
Student inference, next E-step, convergence check, early stop
```

The formal teacher pool currently contains 21 Drive-aligned registry routes plus
the separate official TotalSegmentator route. Auxiliary, legacy, and
developer-only entries also exist in the registry; `mock_seg` is never a formal
teacher.

## Design rules

- **Exact organ identity:** every comparison, fusion, critic decision, dashboard
  row, and training item is keyed by `(case_id, canonical_id)`.
  `liver`, `liver_segment_1`, `pancreas`, and `pancreas_head` are distinct
  targets. Parent masks define ROIs only and cannot substitute for child masks.
- **Hierarchical inference:** major organs run first. Child structures run inside
  parent-mask ROIs with a configurable physical margin and are restored to the
  original CT geometry. A missing parent blocks its children instead of silently
  falling back to full-volume child inference.
- **Independent evidence:** correlated checkpoints are collapsed into explicit
  evidence families before confidence scoring and fusion.
- **QC is a gate, not proof of accuracy:** geometry, non-empty-mask, containment,
  connected-component, and volume checks can reject a candidate but do not turn
  it into expert-validated truth.
- **Auditable automation:** selection records retain evidence components,
  conflicts, missing evidence, grades, model lineage, critic signals, and
  training weights.
- **Safe shared-GPU behavior:** the resource gate skips work when GPUs are busy.
  The formal runner does not stop a shared vLLM process unless explicitly given
  ownership through `MEDAI_MANAGE_OWN_VLLM=1`.

## Repository layout

```text
agent-harness/
  cli_anything/medai/
    medai_cli.py                 JSON-oriented command-line interface
    core/
      multimodel_loop.py         E-step orchestration and artifact writing
      hierarchical_roi.py       parent-first ROI planning and restoration
      organ_taxonomy.py          canonical identity and hierarchy utilities
      auto_label_core.py         evidence scoring, grading, and fusion
      labelcritic_wrapper.py     bounded pairwise VLM comparison
      voxtell_student.py         prompt-student manifests and inference
configs/
  model_registry.yaml            model routes, checkpoints, and capabilities
  organ_taxonomy.json            strict parent/child taxonomy
  student_3d_prompt_target_organs.json
                                  373 exact formal targets and prompt metadata
  autolabel_core.yaml            evidence weights, thresholds, and grades
  teacher_branch_map.yaml        teacher branch routing
data_manifest/                   case manifests
scripts/
  run_em_training.py             single formal multi-round EM entry point
  train_voxtell_prompt_student.py
  build_organ_taxonomy.py
  audit_organ_identity.py
  audit_organ_mappings.py
  check_gpu_resources.py
docs/                            architecture, policy, audit, and run notes
```

Large datasets, checkpoints, and generated outputs are intentionally not bundled
with the repository.

## Prerequisites

- Linux and Python 3.
- NVIDIA GPU and CUDA-compatible PyTorch for real inference/training.
- Model-specific environments for nnUNet v2, MONAI/VISTA3D, VoxTell, or other
  registered teachers used in a run.
- Local CT volumes and masks in NIfTI format (`.nii.gz`).
- Local checkpoints matching `configs/model_registry.yaml`.
- ShapeKit under `third_party/ShapeKit-main` for formal E-steps.
- A LabelCritic-compatible VLM endpoint, normally at
  `http://localhost:8000`, when critic support is enabled.

`--dry-run`, `mock_seg`, critic `stub`, and explicit debug bypasses are for
smoke testing only. Do not include their results in formal experiment reports.

## Installation and basic checks

From the repository root:

```bash
pip install -e agent-harness
python run_medai_cli.py --json doctor
python run_medai_cli.py --json model-inventory
```

Before any GPU smoke test, inference, or benchmark:

```bash
PYTHONPATH=agent-harness python scripts/check_gpu_resources.py
```

The command returns `skipped_resource_busy` instead of interfering with another
GPU process.

## Data and checkpoints

The default formal case manifest is:

```text
data_manifest/case_list_50_tumor.csv
```

Each CSV row must contain:

```text
case_id,ct_path,annotation_folder
```

Optional report, clinical, and pathology fields are supported by the CLI.
`annotation_folder` should contain PanTS/ShapeKit-style
`segmentations/*.nii.gz` masks. These masks remain prior pseudo references
unless independently verified as expert labels.

Validate the manifest:

```bash
python scripts/validate_case_list_50.py \
  --case-list data_manifest/case_list_50_tumor.csv
```

Or select a 50-case PanTS subset:

```bash
python run_medai_cli.py --json pants-select-50 \
  --pants-root third_party/PanTS-main \
  --output data_manifest/case_list_50_tumor.csv \
  --split train \
  --num-cases 50
```

Typical local checkpoint roots include:

```text
checkpoints/
  qchen76_2025_0421/
  CADS_series/
  MOOSE_series/
  nnUNet_private/
  VSmTrans/
  VISTA3D-Inference-Pipeline-master/
  ATLAS-Net/
  VoxTell/voxtell_v1.1/
  Qwen/Qwen3-Embedding-4B/
```

Always use the paths and dataset metadata recorded in the registry rather than
assuming that two checkpoints with similar names share a label space.

## Rebuild and audit the organ identity layer

`configs/class_checkpoint_map_updates.xlsx` is the source workbook for the
current hierarchy. Rebuild and audit it on CPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=agent-harness \
  python scripts/build_organ_taxonomy.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=agent-harness \
  python scripts/audit_organ_mappings.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=agent-harness \
  python scripts/audit_organ_identity.py
```

Unknown or ambiguous mappings are blocked. Legacy records without exact
source-label provenance are marked `legacy_unverified` and excluded from strict
M-step manifests.

Inspect registry coverage and routing:

```bash
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex

python run_medai_cli.py --json route-models \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex

python run_medai_cli.py --json validate-373-target
```

## Run one E-step

The default `run-loop` target is `student_373`, and the default teacher
inference mode is `hierarchical_roi`.

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models cads551,moose666,epai_20250421,vsmtrans,totalsegmentator \
  --organs student_373 \
  --target-config configs/student_3d_prompt_target_organs.json \
  --output outputs/round1 \
  --teacher-inference-mode hierarchical_roi \
  --roi-margin-mm 20 \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --critic-base-url http://localhost \
  --critic-port 8000
```

For each case, major organs are inferred first. Child tasks use parent ROIs and
are restored to the original CT grid. Compatible crops may be merged to avoid
reloading the same checkpoint, but each child keeps an independent anatomical
support box. At least 95% of a merged-crop prediction must remain inside that
support; otherwise only that child is rerun on its independent ROI.

Primary teachers run first and backups are used when required. Round 2 and later
may reuse a valid Round 1 hierarchical teacher cache; student inference is
rerun after every M-step because the student checkpoint changes.

### E-step selection flow

1. Normalize teacher outputs to exact canonical NIfTI masks.
2. Run ShapeKit when enabled.
3. Apply structural and geometry QC.
4. Collapse correlated models into evidence families.
5. Score available evidence with AutoLabelCore.
6. Fuse eligible candidates with family-balanced weights.
7. Use LabelCritic only as a bounded tie-break signal for supported organs.
8. Assign a grade and training weight.
9. Write selections, review/audit rows, manifests, and run summaries.

The default `configs/autolabel_core.yaml` policy uses A/B labels as hard
training targets. C labels have zero hard-label weight and may be used only
through the explicitly supported VoxTell soft-target path. Provisional and
rejected labels remain auditable with zero training weight.

The optional LongTailCritic is disabled by default. It learns synthetic
structural corruptions from high-quality multi-family seeds; its
`structural_corruption_probability` is not a segmentation-accuracy score.

## Expected E-step artifacts

```text
outputs/round1/
  run_summary.json
  inference_results.json
  dice_metrics.csv                 pseudo-consistency where references exist
  round_metrics.csv
  training_manifest.json
  review_queue.jsonl               audit queue; not a mandatory acceptance gate
  auto_arbitration_log.jsonl
  vlm_decisions.jsonl
  pseudo_label_gap_report.json
  pseudo_label_gap_report.csv
  shapekit_report.json
  annotation_versions/
  cases/
    <case_id>/
      hierarchical_inference_plan.json
      hierarchical_predictions/
      pseudo_label_selection.json
```

The per-case selection file is the main audit record. The root
`training_manifest.json` is the handoff to the M-step.

Audit or rescore AutoLabelCore results:

```bash
python scripts/audit_autolabel_core.py \
  --selection outputs/round1/cases/<case_id>/pseudo_label_selection.json \
  --output outputs/round1/cases/<case_id>/autolabel_core_audit.json

python scripts/rescore_autolabel_v3.py --help
```

## Formal multi-round EM run

`scripts/run_em_training.py` is the single formal end-to-end entry point. It
drives the registry-based teacher pool, hierarchical E-step, pseudo-label
selection, VoxTell M-step, student evaluation, cache reuse, and convergence
stopping.

```bash
export PYTHONPATH="$PWD/agent-harness:${PYTHONPATH:-}"
export MEDAI_CASE_LIST="$PWD/data_manifest/case_list_50_tumor.csv"
export MEDAI_OUTPUT_ROOT="$PWD/outputs/formal_em_$(date +%Y%m%d_%H%M%S)"

export MEDAI_NUM_ROUNDS=3
export MEDAI_STUDENT_BACKEND=voxtell_style_3d_prompt
export MEDAI_TEACHER_INFERENCE_MODE=hierarchical_roi
export MEDAI_ROI_MARGIN_MM=20
export MEDAI_CANDIDATE_MODE=route_pruned_with_competition

export MEDAI_ENABLE_SHAPEKIT=1
export MEDAI_ENABLE_CRITIC=1
export MEDAI_VOXTELL_MODEL_DIR="$PWD/checkpoints/VoxTell/voxtell_v1.1"
export MEDAI_TEXT_ENCODING_MODEL="$PWD/checkpoints/Qwen/Qwen3-Embedding-4B"
export MEDAI_VOXTELL_TRAIN_CMD='python scripts/train_voxtell_prompt_student.py'

python scripts/run_em_training.py
```

Important environment controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDAI_NUM_ROUNDS` | `3` | Maximum EM rounds |
| `MEDAI_STUDENT_BACKEND` | `voxtell_style_3d_prompt` | Current 373-target student |
| `MEDAI_TEACHER_INFERENCE_MODE` | `hierarchical_roi` | Parent-first teacher inference |
| `MEDAI_ROI_MARGIN_MM` | `20` | Physical margin around parent masks |
| `MEDAI_INFER_TIMEOUT_SEC` | `3600` | Per-teacher inference timeout |
| `MEDAI_ENABLE_SHAPEKIT` | enabled | Formal mask post-processing requirement |
| `MEDAI_ENABLE_CRITIC` | enabled | LabelCritic support |
| `MEDAI_CONVERGENCE_AUTOSTOP` | enabled | Stop when student change stabilizes |
| `MEDAI_CONVERGENCE_DSC_DELTA` | `0.01` | Round-over-round stop threshold |
| `MEDAI_CONVERGENCE_MIN_ROUNDS` | `2` | Minimum rounds before early stop |
| `MEDAI_MANAGE_OWN_VLLM` | disabled | Permit control of this run's own VLM process |

Formal runs require ShapeKit and LabelCritic unless a smoke/debug-only bypass is
explicitly enabled. The runner records its environment and preflight state in
the output tree. When convergence criteria are met it writes
`convergence_stop.json`.

`vista3d_legacy` remains available only for reproducing the old 127-label
student path and requires `MEDAI_ALLOW_VISTA3D_LEGACY=1`. It is not the current
373-target mainline.

## VoxTell prompt student

The student consumes 3D CT volumes and exact organ prompts. Qwen is a frozen text
encoder: pooled prompt embeddings condition the trainable VoxTell image/decoder
path through cross-attention and multi-scale mask fusion.

Quick trainer validation:

```bash
python scripts/train_voxtell_prompt_student.py \
  --manifest outputs/round1/mstep/voxtell_prompt_student_manifest.json \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --output-dir outputs/round1/mstep \
  --dry-run
```

The prompt cache records the text-model identity, prompt hash, cache format, and
encoder policy to prevent accidental reuse across incompatible Qwen models or
prompt sets.

### Negative prompts

A missing organ mask is not evidence that the organ is absent. Negative samples
may come only from:

- non-medical absent objects;
- out-of-scan anatomy supported by scan-coverage metadata; or
- explicitly confirmed absent anatomy in case metadata.

Arbitrary missing targets, low-confidence teacher outputs, and prior student
empty masks are not valid negatives. See
[VoxTell negative prompt policy](docs/VOXTELL_NEGATIVE_PROMPT_POLICY.md).

## Selected-model-aware M-step

For model-family experiments outside the default VoxTell EM route:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/round1/training_manifest.json \
  --output-folder outputs/mstep_epai \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --pretrained-weights checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth
```

Trainability is conditional on the registry entry, compatible training state,
checkpoint metadata, model dependencies, and available resources. A dry run
writes a plan; it does not prove that training occurred or that a checkpoint was
created.

## Verification

Run the CPU-focused tests and audits before a formal GPU job:

```bash
PYTHONPATH=agent-harness pytest -q \
  agent-harness/tests/test_autolabel_core.py \
  agent-harness/tests/test_hierarchical_identity.py \
  agent-harness/tests/test_atlasnet_integration.py

PYTHONPATH=agent-harness python scripts/verify_final_v9_integrity.py
PYTHONPATH=agent-harness python scripts/audit_373_organ_routing.py
PYTHONPATH=agent-harness python scripts/audit_teacher_readiness.py
```

Useful CLI checks:

```bash
python run_medai_cli.py --json doctor
python run_medai_cli.py --json validate-373-target
python run_medai_cli.py --json model-inventory
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct
```

## Manual review and derived samples

Human review is optional and remains an audit path rather than a hidden
acceptance requirement:

```bash
python run_medai_cli.py --json itksnap-review \
  --review-queue outputs/round1/review_queue.jsonl \
  --output-script outputs/round1/open_itksnap_review.sh \
  --max-cases 10

python run_medai_cli.py --json build-samples \
  --run-output outputs/round1 \
  --output-jsonl outputs/round1/case_samples.jsonl
```

Rule-based RadThinking-style traces are structured audit artifacts. They must not
be presented as validated clinical reasoning.

## Reporting checklist

When presenting results:

1. State whether reference masks are expert labels, weak labels, or pseudo
   references.
2. Label default Dice/DSC as `pseudo_consistency`.
3. Report exact target coverage and missing/blocked organs.
4. Separate structural QC, LabelCritic decisions, and model agreement from
   anatomical accuracy.
5. Preserve evidence family, canonical organ identity, source-label provenance,
   ShapeKit status, grade, and training weight.
6. Exclude dry-run, mock, stub, legacy-unverified, and resource-skipped records
   from formal metrics.
7. Claim an updated student checkpoint only when real training produced and
   validated an inference-compatible checkpoint.

## Further documentation

- [Hierarchical ROI and strict organ identity](docs/HIERARCHICAL_ROI_AND_ORGAN_IDENTITY.md)
- [Multi-model routing architecture](docs/ARCHITECTURE_multimodel_routing.md)
- [Auto fine-label and VoxTell implementation](docs/AUTO_FINE_LABEL_373_VOXTELL_IMPLEMENTATION.md)
- [VoxTell/Qwen text encoder architecture](docs/VOXTELL_QWEN_TEXT_ENCODER_ARCHITECTURE.md)
- [VoxTell negative prompt policy](docs/VOXTELL_NEGATIVE_PROMPT_POLICY.md)
- [Selected-model-aware M-step](docs/SELECTED_MODEL_AWARE_MSTEP_V7.md)
- [Model inventory and trainability](docs/MODEL_INVENTORY_AND_TRAINABILITY_V7.md)
- [Command cheatsheet](docs/COMMAND_CHEATSHEET.md)

## License and data governance

This repository integrates code paths and model families with different
licenses and data-use conditions. Review the license, citation, checkpoint, and
dataset terms for every enabled component before redistribution or deployment.
Do not commit private clinical data, protected health information, or
non-redistributable model weights.
