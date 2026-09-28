# MedAI Agent Loop

## Handoff quick start

This is the GitHub source repository for the MedAI Agent Loop, a multi-teacher
3D medical image segmentation pipeline with strict 373-target organ identity,
pseudo-label selection, LabelCritic audit support, and VoxTell-style student
training.

For a new maintainer:

```bash
git clone https://github.com/Xiang-mira/medical_agent.git
cd medical_agent

python -m venv .venv
source .venv/bin/activate
pip install -e agent-harness

python run_medai_cli.py --json doctor
python run_medai_cli.py --json model-inventory
PYTHONPATH=agent-harness pytest -q agent-harness/tests
```

Read [Repository Handoff](docs/REPOSITORY_HANDOFF.md) first for the clean repo
map, branch policy, data/checkpoint policy, and maintainer checklist.

## Current formal status and guardrails

This repository is a registry-driven, multi-model pseudo-label refinement
system for 3D medical image segmentation. The current formal target space is
373 exact organ/structure prompts.

The current safe operating contract is:

- **LabelCritic uses the pinned official source** in
  `third_party/LabelCritic-main` for AP projection and pairwise comparison.
  Project code is only a wrapper/adapter. The official benchmark data is not
  available, so LabelCritic remains `audit_only` unless a licensed benchmark
  with checksums is added and gates pass.
- **VoxTell inference/baseline uses the official VoxTell source and assets** in
  `third_party/VoxTell` plus `checkpoints/VoxTell/voxtell_v1.1`.
  The current M-step is **not official VoxTell finetuning**. It must be reported
  as: `project prompt-distillation trainer initialized from official VoxTell assets`.
- **Positive training labels are recovered only by family-free geometric teacher
  consensus** or later approved gates. `evidence_family` is provenance/audit
  metadata only; it cannot vote, choose a winner, raise training weight, or
  enter VLM prompts.
- **Smoke/debug outputs are not scientific evidence.** Any run using `mock_seg`,
  LabelCritic `stub`, smoke overrides, or dry-run artifacts is plumbing only and
  must not enter a formal manifest, M-step claim, or Round2 claim.
- **Uncalibrated LabelCritic selection is disabled in the formal path.**
  `MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION=1` is ignored by formal
  selection and recorded as ignored metadata.
- **Projection fallback is audit-only.** If a LabelCritic grade/projection path
  falls back to a project slice projection, the grade is skipped and cannot
  affect training eligibility.

## Architecture

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
family-free 3D teacher-consensus check
                         |
                         v
LabelCritic official pairwise audit when calibrated gates allow/require it
                         |
                         +----> select one original teacher medoid or withhold for review
                         |
                         v
consensus core manifest + single-teacher ablation + LabelCritic audit manifest
                         |
                         v
project prompt-distillation M-step initialized from official VoxTell assets
                         |
                         v
Student inference, next E-step, convergence check, early stop
```

## Technical specifications

- **Exact organ identity:** comparisons, critic decisions, dashboard
  rows, and training items are keyed by `(case_id, canonical_id)`.
  `liver`, `liver_segment_1`, `pancreas`, and `pancreas_head` are distinct
  targets. Parent masks define ROIs only and cannot substitute for child masks.
- **Hierarchical inference:** major organs run first. Child structures run inside
  parent-mask ROIs with a configurable physical margin and are restored to the
  original CT geometry. A missing parent blocks dependent child inference.
- **Family is audit-only:** correlated checkpoints may be recorded as evidence
  families for provenance and dashboards, but family membership never selects a
  winner, increases training eligibility, replaces LabelCritic, or enters VLM
  prompts.
- **Geometric consensus:** multi-teacher positives require QC-passing original
  teacher masks, complete-link 3D Dice agreement at `>=0.95`, a unique largest
  cluster, and an original teacher medoid winner. Fusion masks cannot win.
- **Quality control:** geometry, non-empty-mask, containment,
  connected-component, and volume checks are applied before selection.
- **Audit records:** selection records retain evidence components,
  conflicts, missing evidence, grades, model lineage, critic signals, and
  training weights.

## Repository layout

```text
README.md                       project overview and execution guide
run_medai_cli.py                cross-platform CLI entry point
run_medai_cli.bat               Windows CLI launcher
agent-harness/
  README.md                     Python package and test layout
  cli_anything/medai/
    medai_cli.py                 JSON-oriented command-line interface
    core/
      multimodel_loop.py         E-step orchestration and artifact writing
      hierarchical_roi.py       parent-first ROI planning and restoration
      organ_taxonomy.py          canonical identity and hierarchy utilities
      auto_label_core.py         audit-only evidence scoring and grading
      labelcritic_wrapper.py     organ-specific candidate ranking
      voxtell_student.py         prompt-student manifests and inference
configs/
  README.md                      configuration index
  model_registry.yaml            model routes, checkpoints, and capabilities
  organ_taxonomy.json            strict parent/child taxonomy
  student_3d_prompt_target_organs.json
                                  373 exact formal targets and prompt metadata
  autolabel_core.yaml            evidence weights, thresholds, and grades
  teacher_branch_map.yaml        teacher branch routing
data/
  README.md                      repository demo data
data_manifest/
  README.md                      case-manifest schema and inventory
scripts/
  README.md                      script categories and primary entry points
  run_em_training.py             single formal multi-round EM entry point
  train_voxtell_prompt_student.py
  build_organ_taxonomy.py
  audit_organ_identity.py
  audit_organ_mappings.py
  check_gpu_resources.py
docs/
  README.md                      documentation index
  REPOSITORY_HANDOFF.md          maintainer handoff, branch policy, data policy
  guides/                        operating-system and packaging guides
  archive/                       superseded implementation documents
  raw_materials/                 source materials
third_party/
  README.md                      vendored external projects
wrappers/
  README.md                      compatibility wrapper index
```

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

## Installation and basic checks

From the repository root:

```bash
pip install -e agent-harness
python run_medai_cli.py --json doctor
python run_medai_cli.py --json model-inventory
```

Before GPU execution:

```bash
PYTHONPATH=agent-harness python scripts/check_gpu_resources.py
```


## Hugging Face model assets

Large model weights are released in the Hugging Face model-assets repository, not in GitHub:

```text
https://huggingface.co/Xiang-mira/MedIA-Agentic-AI
```

The HF release currently includes 23 teacher models and one VoxTell-style student model. It intentionally excludes `mock_seg` and `epai_finetuned`. Internal/JHU/private assets are listed for lab collaboration but should not be externally advertised.

Download on a new GPU/HPC machine:

```bash
git lfs install
git clone https://huggingface.co/Xiang-mira/MedIA-Agentic-AI checkpoints/MedIA-Agentic-AI
export HF_ASSET_ROOT="$PWD/checkpoints/MedIA-Agentic-AI"
```

Or use:

```bash
python examples/download_from_hf.py \
  --repo-id Xiang-mira/MedIA-Agentic-AI \
  --local-dir checkpoints/MedIA-Agentic-AI
```

See [Hugging Face model release guide](docs/HUGGINGFACE_MODEL_RELEASE.md) and [HF model manifest](configs/hf_model_manifest.yaml) for the full model list, supported organs, input/output formats, CLI commands, expected GPU memory, and HPC examples.

## DISCOVERY / OnDemand HPC migration

DISCOVERY does not currently provide the project workflow with a direct personal
SSH/SCP entry point. Use the OnDemand Web Portal, then either OnDemand Shell or
OnDemand VS Code Server. Do not assume `ssh discovery`, `scp`, VS Code
Remote-SSH, or Codex SSH remote projects are available.

Migration is intentionally split by asset type:

```text
GitHub                                      source code, configs, docs, manifests
Xiang-mira/MedIA-Agentic-AI-Private-HPC     private restore checkpoints and selected formal state
HPC/project storage                         new CT datasets and any local PHI-bearing data
not migrated                                PanTS tarballs/data, caches, bad/smoke outputs
```

Create the software environment on the HPC instead of copying an old conda
folder:

```bash
git clone https://github.com/Xiang-mira/medical_agent.git
cd medical_agent

python -m venv .venv
source .venv/bin/activate
pip install -r agent-harness/requirements.txt
```

Download the private migration assets directly on the HPC:

```bash
huggingface-cli login
huggingface-cli download Xiang-mira/MedIA-Agentic-AI-Private-HPC \
  --local-dir checkpoints \
  --resume-download
```

The private repo is laid out so that teacher assets land under `checkpoints/`
with the same local paths expected by `configs/model_registry.yaml`. It also
contains `student_models/em_round1_25case_full_mstep_lr3e-5_20260711/` and
filtered lightweight formal state under `outputs/` inside the download root
(`checkpoints/student_models/...` and `checkpoints/outputs/...` when using the
command above). Copy or symlink those optional state folders into the project
root only when a continuation run needs them.

Qwen2-VL and Qwen2.5-VL are public upstream models and are not mirrored in the
private migration repo. Download them only if a LabelCritic/VLM workflow needs
them. For formal LabelCritic/VLM judging, prefer the public 70B/72B-class model
`Qwen/Qwen2-VL-72B-Instruct-AWQ` served through the OpenAI-compatible endpoint
used by LabelCritic. The local 7B downloads below are resource-limited fallback
or debugging options:

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir checkpoints/Qwen/Qwen2-VL-7B-Instruct \
  --resume-download

huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir checkpoints/Qwen/Qwen2.5-VL-7B-Instruct \
  --resume-download
```

On DISCOVERY HPC, keep the preferred LabelCritic model under the shared
bodymaps model root (exposed to the code tree through `checkpoints`) instead of
committing it to this private migration repository, for example:

```bash
hf download Qwen/Qwen2-VL-72B-Instruct-AWQ \
  --revision 712d5a5a210e7f0af603d2a949b577c68de2c6ef \
  --local-dir checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ
```

`medai-cli critic` and the multi-model loop use the LabelCritic/vLLM
OpenAI-compatible endpoint selected by `--base-url/--port` or
`--critic-base-url/--critic-port`. For `medai-cli run-loop`, pass
`--critic-vlm-model Qwen/Qwen2-VL-72B-Instruct-AWQ` when the served endpoint uses
that public model id. If neither an explicit VLM model nor `LABELCRITIC_MODEL_ID` is configured, the
wrapper asks the endpoint for `/v1/models` and uses the first served model id.
Formal runs should configure the explicit 72B AWQ model id.

Verify the restored private assets and code state:

```bash
python scripts/verify_hf_asset_manifest.py \
  --manifest docs/migration/hf_asset_manifest.tsv \
  --root checkpoints

python scripts/check_gpu_resources.py
python scripts/audit_teacher_readiness.py
pytest agent-harness/tests/test_imports.py \
       agent-harness/tests/test_student_trainset_pseudo_consistency.py
```

PanTS data and tarballs are not migrated. On the HPC, regenerate or rewrite
`data_manifest/*.csv` so `ct_path` and `annotation_folder` point to the new
available dataset paths. Do not rely on old absolute paths such as
`/home/teacher1/JHU-project1/medical_agent/data/PanTS`.

Migration audit files live under `docs/migration/`:

```text
HPC_MIGRATION_AUDIT.md
asset_manifest.json
github_file_manifest.tsv
hf_asset_manifest.tsv
hf_upload_plan.tsv
excluded_manifest.tsv
```

Regenerate them after changing restore assets:

```bash
python scripts/build_hpc_migration_manifests.py
```

Maintain the private HF repo with the large-folder staging uploader. Do not use
small `create_commit` batches for the filtered `outputs/` tree; the formal
state contains tens of thousands of small files and will hit Hugging Face's
repository commit rate limit. The staging directory uses hardlinks by default,
so it does not duplicate the 29+ GiB checkpoint payload on disk:

```bash
python scripts/stage_hf_private_assets.py \
  --manifest docs/migration/hf_asset_manifest.tsv \
  --stage-dir .hf_hpc_upload_staging \
  --prune-stage \
  --hardlink

python scripts/stage_hf_private_assets.py \
  --manifest docs/migration/hf_asset_manifest.tsv \
  --stage-dir .hf_hpc_upload_staging \
  --verify-only

python scripts/stage_hf_private_assets.py \
  --stage-dir .hf_hpc_upload_staging \
  --upload-large-folder \
  --repo-id Xiang-mira/MedIA-Agentic-AI-Private-HPC \
  --num-workers 2
```

If the upload is interrupted or the network returns an SSL EOF, rerun the final
command without deleting `.hf_hpc_upload_staging/.cache/`; the Hugging Face
large-folder uploader resumes from that cache. Use `--copy-fallback` only when
hardlinking fails and there is enough disk space to copy the restore payload.

After upload, check the remote repo against the manifest:

```bash
python scripts/check_hf_remote_manifest.py \
  --repo-id Xiang-mira/MedIA-Agentic-AI-Private-HPC \
  --manifest docs/migration/hf_asset_manifest.tsv
```

The older `scripts/upload_hf_private_assets.py --outputs-only` path is kept only
for debugging small subsets; it is not the recommended migration upload path.

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
`annotation_folder` is a legacy/debug pseudo-reference hook, not a mainline
accuracy reference. The formal loop builds 373-target pseudo-label rows from
teacher outputs, confirmed absent zero masks, and withheld uncertain targets.

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
  ATLAS-Net/                 # clone of https://huggingface.co/Koushik45048545309/Atlas-Net, including utils/atlas_postprocess.py
  VoxTell/voxtell_v1.1/
  Qwen/Qwen3-Embedding-4B/
```

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

Inspect registry coverage and routing:

```bash
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex

python run_medai_cli.py --json route-models \
  --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex

python run_medai_cli.py --json validate-373-target
```

## E-step execution

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
  --critic-base-url http://127.0.0.1 \
  --critic-port 8000 \
  --critic-vlm-model Qwen/Qwen2-VL-72B-Instruct-AWQ
```

For each case, major organs are inferred first. Child tasks use parent ROIs and
are restored to the original CT grid. Compatible crops may be merged to avoid
reloading the same checkpoint, but each child keeps an independent anatomical
support box. At least 95% of a merged-crop prediction must remain inside that
support; otherwise only that child is rerun on its independent ROI.

Primary teachers run first and backups are used when required for Round 1
bootstrap. Round 2 and later use the EM repair path: the previous selected
pseudo label is the immutable pseudo reference, and the previous round cleaned
student prediction competes against it through LabelCritic/verifier. Fresh
teacher replay is debug/bootstrap-only, not the formal Round2+ update path.

### E-step selection flow

1. Normalize teacher outputs to exact canonical NIfTI masks.
2. Run ShapeKit when enabled.
3. Apply structural and geometry QC.
4. In Round 1, exclude fusion, student, historical pseudo, and QC-failed masks
   from formal positive recovery. In Round2+, allow only `round_prev_selected`
   and postprocessed `student_prev` in the formal EM candidate pool.
5. Compute a full 3D Dice matrix over original teacher masks.
6. Accept only a unique complete-link geometric consensus cluster
   (`Dice >= 0.95`) and choose the original teacher medoid.
7. Run official LabelCritic pairwise/audit artifacts when needed. Until the
   official benchmark gate is ready, LabelCritic winners are audit-only:
   `audit_winner` may be written, but `formal_winner=null`,
   `training_weight=0`, and `should_enter_student_training=false`.
8. Withhold uncertain/rejected targets for review. No family vote, fusion
   fallback, uncalibrated LabelCritic winner, or project projection fallback may
   enter the training manifest.
9. Materialize one audit record for every case × 373 target, using an all-zero
   negative only when scan coverage proves absence.
10. Assign training weights only after the selection gate and lineage checks.

AutoLabelCore/family scoring is an audit and dashboard signal in the formal
path. It must not overwrite a LabelCritic abstention or geometric consensus
decision, and it must not make single-teacher labels formal high-confidence
positives.

### Split manifests after E-step

For repaired runs, split the full 373 manifest into three isolated products:

```bash
python scripts/split_labelcritic_repair_manifests.py \
  --input outputs/round1/full_case_373_manifest.json \
  --output-dir outputs/round1/labelcritic_repair_split_manifests
```

- `consensus_core_manifest.json`: geometric consensus positives plus reliable
  `negative_absent` labels.
- `single_teacher_ablation_manifest.json`: core plus strict single-teacher
  low-weight ablation labels.
- `labelcritic_audit_manifest.json`: LabelCritic pairwise/audit records with
  zero training weight.

## E-step output artifacts

```text
outputs/round1/
  run_summary.json
  inference_results.json
  dice_metrics.csv
  round_metrics.csv
  training_manifest.json
  review_queue.jsonl
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

`dice_metrics.csv` records E-step candidate/selected pseudo-label consistency.
The formal EM route does not read external fine-label roots or report accuracy.
Current evaluation tables include metric-contract columns such as
`metric_target`, `metric_subject`, `metric_comparison`, and
`metric_interpretation`; mainline targets are `selected_pseudo_label`,
`teacher_candidate`, `student_candidate`, and `absent_negative_zero_mask`.

Audit or rescore AutoLabelCore results:

```bash
python scripts/audit_autolabel_core.py \
  --selection outputs/round1/cases/<case_id>/pseudo_label_selection.json \
  --output outputs/round1/cases/<case_id>/autolabel_core_audit.json

python scripts/rescore_autolabel_v3.py --help
```

Run the total repair audit after E-step, training, and post-processing:

```bash
python scripts/audit_repair_plan.py \
  --selection-manifest outputs/round1/full_case_373_manifest.json \
  --student-manifest outputs/round1/mstep/voxtell_prompt_student_manifest.json \
  --sampling-audit outputs/round1/mstep/student/sampling_audit.json \
  --postprocess-csv outputs/round1/student_postprocessed/student_containment_postprocess_per_mask.csv \
  --metric-artifact outputs/round1/round_metrics.csv
```

Generate the 373-class prior and apply parent-ROI containment:

```bash
python scripts/build_organ_ct_appearance_373.py
python scripts/apply_organ_type_postprocess.py \
  --input-root outputs/round1/student_predictions \
  --output-root outputs/round1/student_predictions_postprocessed \
  --parent-root outputs/round1/cases
```

Known limitations: the official LabelCritic benchmark gate is currently blocked
because licensed benchmark data/checksums are unavailable; generated 373-organ
descriptions are project extensions; LabelCritic projections are 2D views of 3D
anatomy; absent negatives require explicit FOV/coverage evidence; all mainline
metrics measure pseudo-label construction/refinement quality, not true
segmentation accuracy.

## Formal multi-round EM run

`scripts/run_em_training.py` is the formal end-to-end entry point for the
project EM route. It drives the registry-based teacher pool, hierarchical
E-step, pseudo-label selection, the **project prompt-distillation trainer
initialized from official VoxTell assets**, student evaluation, cache reuse,
and convergence stopping.

Do not describe this route as official VoxTell finetuning. Official VoxTell is
used for source code, checkpoint initialization, prompt embeddings, pretrained
baseline inference, and optional baseline-only encoder-transfer experiments;
the prompt-conditioned M-step trainer is project code.

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

Environment controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDAI_NUM_ROUNDS` | `3` | Maximum EM rounds |
| `MEDAI_STUDENT_BACKEND` | `voxtell_style_3d_prompt` | Current 373-target student |
| `MEDAI_TEACHER_INFERENCE_MODE` | `hierarchical_roi` | Parent-first teacher inference |
| `MEDAI_ROI_MARGIN_MM` | `20` | Physical margin around parent masks |
| `MEDAI_INFER_TIMEOUT_SEC` | `3600` | Per-teacher inference timeout |
| `MEDAI_ENABLE_SHAPEKIT` | enabled | Formal mask post-processing requirement |
| `MEDAI_ENABLE_CRITIC` | enabled | LabelCritic support |
| `MEDAI_LABELCRITIC_ALLOW_UNCALIBRATED_SELECTION` | ignored in formal selection | Historical debug flag; formal selector records and ignores it |
| `MEDAI_CONVERGENCE_AUTOSTOP` | enabled | Stop when student change stabilizes |
| `MEDAI_CONVERGENCE_DSC_DELTA` | `0.01` | Round-over-round stop threshold |
| `MEDAI_CONVERGENCE_MIN_ROUNDS` | `2` | Minimum rounds before early stop |
| `MEDAI_MANAGE_OWN_VLLM` | disabled | Permit control of this run's own VLM process |

When convergence criteria are met, the runner writes
`convergence_stop.json`.

## VoxTell usage contract

There are two separate VoxTell paths:

1. **Official VoxTell inference/baseline**
   - Source: `third_party/VoxTell`.
   - Assets: `checkpoints/VoxTell/voxtell_v1.1` and the official prompt
     embedding bank.
   - Adapter: `OfficialVoxTellPretrainedAdapter`.
   - Default role: `baseline_only`; it must not silently enter the formal
     pseudo-label manifest.

2. **Project prompt-distillation M-step**
   - Script: `scripts/train_voxtell_prompt_student.py`.
   - Required wording: `project prompt-distillation trainer initialized from
     official VoxTell assets`.
   - Not official `voxtell-finetune`.
   - Any report, paper draft, or experiment log must preserve this distinction.

Trainer validation:

```bash
python scripts/train_voxtell_prompt_student.py \
  --manifest outputs/round1/mstep/voxtell_prompt_student_manifest.json \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --output-dir outputs/round1/mstep \
  --dry-run
```

Prompt-cache metadata includes the text-model identity, prompt hash, cache
format, and encoder policy. Negative-prompt configuration is defined in
[VoxTell negative prompt policy](docs/VOXTELL_NEGATIVE_PROMPT_POLICY.md).

Audit the official VoxTell source/assets:

```bash
python scripts/audit_voxtell_vendor.py --json-out outputs/audit_voxtell_vendor.json
python scripts/audit_voxtell_official_assets.py --output outputs/audit_voxtell_official_assets.json
```

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

Trainability is determined by the registry entry, training state, checkpoint
metadata, model dependencies, and available resources.

## Verification

Run the test and audit suite:

```bash
PYTHONPATH=agent-harness pytest -q \
  agent-harness/tests/test_autolabel_core.py \
  agent-harness/tests/test_hierarchical_identity.py \
  agent-harness/tests/test_atlasnet_integration.py

PYTHONPATH=agent-harness python scripts/verify_final_v9_integrity.py
PYTHONPATH=agent-harness python scripts/audit_373_organ_routing.py
PYTHONPATH=agent-harness python scripts/audit_teacher_readiness.py
PYTHONPATH=agent-harness python scripts/audit_labelcritic_vendor.py \
  --output outputs/audit_labelcritic_vendor.json
PYTHONPATH=agent-harness python scripts/audit_voxtell_vendor.py \
  --json-out outputs/audit_voxtell_vendor.json
PYTHONPATH=agent-harness python scripts/audit_voxtell_official_assets.py \
  --output outputs/audit_voxtell_official_assets.json
```

CLI checks:

```bash
python run_medai_cli.py --json doctor
python run_medai_cli.py --json validate-373-target
python run_medai_cli.py --json model-inventory
python run_medai_cli.py --json registry-candidates \
  --organs pancreas,liver,aorta,pancreatic_duct
```

## Review and sample utilities

```bash
python run_medai_cli.py --json itksnap-review \
  --review-queue outputs/round1/review_queue.jsonl \
  --output-script outputs/round1/open_itksnap_review.sh \
  --max-cases 10

python run_medai_cli.py --json build-samples \
  --run-output outputs/round1 \
  --output-jsonl outputs/round1/case_samples.jsonl
```

## Further documentation

- [Documentation index](docs/README.md)
- [Script index](scripts/README.md)
- [Configuration index](configs/README.md)
- [Hierarchical ROI and strict organ identity](docs/HIERARCHICAL_ROI_AND_ORGAN_IDENTITY.md)
- [Multi-model routing architecture](docs/ARCHITECTURE_multimodel_routing.md)
- [Auto fine-label and VoxTell implementation](docs/AUTO_FINE_LABEL_373_VOXTELL_IMPLEMENTATION.md)
- [VoxTell/Qwen text encoder architecture](docs/VOXTELL_QWEN_TEXT_ENCODER_ARCHITECTURE.md)
- [VoxTell negative prompt policy](docs/VOXTELL_NEGATIVE_PROMPT_POLICY.md)
- [Selected-model-aware M-step](docs/SELECTED_MODEL_AWARE_MSTEP_V7.md)
- [Model inventory and trainability](docs/MODEL_INVENTORY_AND_TRAINABILITY_V7.md)
- [Command cheatsheet](docs/COMMAND_CHEATSHEET.md)
