# Teacher Meeting Implementation Audit - 2026-06-05

This audit records verified progress against
`docs/TEACHER_MEETING_IMPLEMENTATION_PLAN_2026_06_05.md` and the current Chinese
execution plan
`docs/TEACHER_MEETING_IMPLEMENTATION_PLAN_CN_2026_06_05.md`.

## Verified Complete In This Pass

### Priority 1: E-step Selection Path

Status: completed for the core code path.

Evidence:

- `multimodel_loop.py` compiles.
- Multi-candidate smoke test passed with two preseeded teacher candidates.
- The smoke test wrote:
  - `selection_metadata.json`
  - source-aware `training_manifest.json`
  - `vlm_decisions.jsonl`
- LabelCritic decisions are now appended to durable JSONL output.
- Selection schema is stable for no-candidate, single-candidate, LabelCritic,
  and fallback branches.
- ShapeKit fallback is recorded with `shapekit_fallback` review/quality flags.

Smoke command category:

```text
PYTHONPATH=agent-harness python <inline estep_metadata_smoke>
```

Observed smoke result:

```json
{
  "status": "success",
  "total_updated": 1,
  "total_labelcritic_decisions": 1,
  "manifest_items": 1,
  "vlm_decisions": 1
}
```

### Priority 2: Auditable Manifest

Status: completed for generic M-step and VoxTell M-step manifests.

Evidence:

- `mstep_runner.build_training_manifest()` now carries:
  - `ct_path`
  - `selected_model`
  - `candidate_models`
  - `selection_method`
  - `selection_status`
  - `labelcritic_records`
  - `shapekit_status`
  - `dataset_role`
  - `ground_truth_status`
  - `review_flags`
  - `quality_flags`
  - `source_metadata_available`
- `VoxTellStudent.build_training_manifest()` now reads E-step selection metadata
  and writes the same source/quality fields.
- Manifest smoke test passed with one metadata-backed organ and one organ
  without metadata.

Observed smoke result:

```json
{
  "mstep_items": 2,
  "voxtell_items": 2
}
```

### Priority 3: Round Student Inference Entry

Status: completed for the standalone helper.

Evidence:

- `scripts/run_student_infer_then_round2.py` no longer imports or uses
  `VISTA3DStudent`.
- It now uses `VoxTellStudent`.
- It reads `configs/student_3d_prompt_target_organs.json`.
- It writes predictions to
  `outputs/round<round>/student_predictions/<case_id>/<organ>.nii.gz`, matching
  the Round2 `student_prev` injection contract.
- Dry-run smoke passed for one case and one prompt.

Observed smoke:

```text
Round 1 VoxTell-style 3D prompt student inference
Cases: 1, prompts: 1
Dry run: True
```

### Priority 4: ShapeKit Formal Default

Status: completed for default configuration.

Evidence:

- `scripts/run_em_training.py` now defaults `ENABLE_SHAPEKIT=True`.
- `MEDAI_FAST_SMOKE=1` disables ShapeKit for smoke/debug only.
- E-step metadata distinguishes `skipped_dry_run` from
  `skipped_debug_only`.

Observed config checks:

```text
default ENABLE_SHAPEKIT -> True
MEDAI_FAST_SMOKE=1 ENABLE_SHAPEKIT -> False
```

### Priority 5: Student Failure Mining

Status: completed for first implementation and smoke validation.

Evidence:

- Added `scripts/mine_student_failure_cases.py`.
- It compares student predictions against selected pseudo labels.
- It outputs:
  - `student_failure_cases.csv`
  - `student_failure_all_comparisons.csv`
  - `student_failure_organ_summary.csv`
  - `student_failure_cases.json`
- It records Dice, empty masks, shape mismatch, volume ratio, selected source
  model, candidate models, selection method, ShapeKit status, and flags.
- It explicitly labels metrics as
  `student_vs_selected_pseudo_label_consistency`.
- Smoke test found a deliberately too-small student mask as a review item.

Observed smoke result:

```json
{
  "status": "success",
  "num_comparisons": 1,
  "num_review_items": 1
}
```

### Legacy Route Safety

Status: improved.

Evidence:

- `scripts/repair_round1.py` is guarded behind
  `MEDAI_ALLOW_VISTA3D_LEGACY=1`.
- `scripts/rerun_round2_mstep.py` is guarded behind
  `MEDAI_ALLOW_VISTA3D_LEGACY=1`.
- `scripts/evaluate_all_organs.py` is guarded behind
  `MEDAI_ALLOW_VISTA3D_LEGACY=1`.
- Default `scripts/run_em_training.py` remains
  `MEDAI_STUDENT_BACKEND=voxtell_style_3d_prompt`.
- VISTA3D remains available as teacher/reference/legacy only.

### Real-CT Tiny End-to-End Check

Status: completed for a controlled 2-case smoke subset.

Scope:

- 2 real CTs from `data_manifest/case_list_50_tumor.csv`.
- 5 organs: liver, pancreas, spleen, kidney_left, kidney_right.
- Lightweight `mock_seg` teacher.
- ShapeKit enabled.
- LabelCritic backend set to `stub` because this check validates pipeline
  structure, not VLM quality.

Evidence:

- `mock_seg` was repaired to generate deterministic synthetic masks from real
  CT shape when `data/PanTS/demo_masks/<case_id>` is absent.
- Direct mock smoke produced 7 masks for a real PanTS CT.
- E-step completed successfully:

```json
{
  "status": "success",
  "num_cases": 2,
  "total_updated": 10,
  "total_labelcritic_decisions": 0
}
```

- The resulting generic training manifest had:
  - 10 items;
  - `source_metadata_available=True` for all items;
  - `selected_model=mock_seg` for all items;
  - `shapekit_status=success` for all items;
  - `selection_method=single_teacher_default` for all items.
- `review_queue.jsonl` correctly records low-reference-Dice items when a
  reference annotation is available.
- A focused 1-case/1-organ rerun verified that `review_flags` and
  `quality_flags` include `low_reference_dice`.

### VoxTell M-step Contract Check

Status: completed for dry-run.

Evidence:

- `voxtell-student-manifest` built a 10-item prompt/mask manifest from the
  real-CT tiny E-step output.
- All 10 items had CT image paths and source/quality metadata.
- `scripts/train_voxtell_prompt_student.py --dry-run --max-items 4` succeeded:
- Dry-run now writes both `voxtell_prompt_train_plan.json` and
  `voxtell_prompt_train_result.json` for stable automated audit.

```json
{
  "stage": "train_voxtell_prompt_student",
  "status": "dry_run",
  "num_manifest_items": 4,
  "model_files_ok": true,
  "epochs": 1,
  "learning_rate": 5e-05
}
```

### Student Inference and Failure-Mining Contract Check

Status: completed for dry-run/planning.

Evidence:

- `scripts/run_student_infer_then_round2.py --dry-run` planned VoxTell-style
  3D prompt inference for 2 real CT cases and 2 prompts.
- `scripts/mine_student_failure_cases.py` compared expected student outputs
  against selected pseudo labels and correctly emitted review items for missing
  student predictions.

Observed failure-mining dry-run result:

```json
{
  "status": "success",
  "num_comparisons": 4,
  "num_review_items": 4
}
```

### Reusable Tiny Regression Script

Status: completed.

Evidence:

- Added `scripts/run_teacher_plan_tiny_check.py`.
- It runs:
  - subset case-list creation;
  - E-step with real CT inputs and `mock_seg`;
  - optional ShapeKit;
  - VoxTell prompt manifest build;
  - VoxTell trainer dry-run;
  - VoxTell student inference dry-run;
  - student failure mining.
- Quick validation passed with 1 real CT and 2 organs:

```json
{
  "status": "success",
  "estep_total_updated": 2,
  "manifest_items": 2,
  "all_items_have_source_metadata": true,
  "shapekit_statuses": ["skipped_debug_only"]
}
```

### Real Teacher Subset Check

Status: completed for one real nnUNet teacher family.

Scope:

- Teacher: `epai_20250421`.
- Case: `PanTS_00000026`.
- Organs checked downstream: liver, pancreas, spleen, kidney_left, kidney_right.

Evidence:

- Registry/checkpoint dry-run checks succeeded for:
  - `vista3d`
  - `epai_20250421`
  - `vsmtrans`
  - `totalsegmentator`
  - `unest`
- Added `scripts/audit_teacher_readiness.py` to make this dry-run readiness
  check reproducible.
- Subset readiness audit across 9 teacher entries completed:

```json
{
  "status": "success",
  "num_models": 9,
  "models_ready_for_command": 9,
  "models_missing_checkpoint_path": []
}
```
- Required command-line tools/imports are present:
  - `TotalSegmentator`
  - `nnUNetv2_predict`
  - `nnUNetv2_predict_from_modelfolder`
  - `totalsegmentator`
  - `nnunetv2`
  - `monai`
  - `nibabel`
  - `torch`
- Real ePAI inference completed successfully on one real CT:

```json
{
  "status": "success",
  "runtime_sec": 78.766,
  "num_masks": 24
}
```

- Real-teacher subset E-step with `epai_20250421 + mock_seg` completed:

```json
{
  "status": "success",
  "num_cases": 1,
  "total_updated": 5,
  "total_labelcritic_decisions": 5
}
```

- The resulting manifest had:
  - 5 items;
  - all items with `source_metadata_available=True`;
  - all items with candidates `["epai_20250421", "mock_seg"]`;
  - all items selected from `epai_20250421`;
  - all items with `shapekit_status=success`;
  - 5 durable `vlm_decisions.jsonl` lines.

Notes:

- This check used `critic_backend=stub`, so the selection method is recorded as
  `label_critic_fallback`. Real VLM LabelCritic still needs a separate run.
- The selected fallback was correct for the smoke because ePAI had higher
  reference consistency than synthetic mock masks.
- Added `scripts/run_real_teacher_subset_check.py` so this ePAI+mock
  multi-candidate validation can be rerun.

### Real LabelCritic/VLM Check

Status: completed for a controlled ePAI-vs-mock A/B pair and for the main
E-step pipeline.

Setup:

- VLM server: `vllm.entrypoints.openai.api_server`.
- Model: local `checkpoints/Qwen/Qwen2-VL-7B-Instruct`.
- Endpoint: `http://localhost:8000/v1/models`.
- GPU: A100 80GB; the server was stopped after the check to release memory.

Fix applied:

- Patched `third_party/LabelCritic-main/ErrorDetector.py` so missing
  `IPython.display` no longer crashes headless server execution. The fallback
  `display()` is a no-op and does not affect LabelCritic decisions.
- Added `scripts/run_labelcritic_pair_check.py` to make pairwise LabelCritic
  checks reproducible with either `stub` or real `labelcritic` backend.

Evidence:

- `scripts/run_labelcritic_pair_check.py --backend stub` succeeded with
  `projection_status=success`.
- `scripts/run_labelcritic_pair_check.py --backend labelcritic` succeeded after
  the IPython fallback patch:

```json
{
  "stage": "labelcritic_pair_check",
  "status": "success",
  "backend": "labelcritic",
  "decision": {
    "winner": "uncertain",
    "parse_status": "vlm_undecided"
  }
}
```

- Main E-step pipeline with real LabelCritic completed on preseeded ePAI +
  mock candidates for liver/pancreas:

```json
{
  "status": "success",
  "total_updated": 2,
  "total_labelcritic_decisions": 2
}
```

- The resulting manifest had:
  - 2 items;
  - `labelcritic_records[*].status=success`;
  - `vlm_decisions.jsonl` with 2 durable VLM records;
  - `shapekit_status=success` for both organs;
  - `selected_model=epai_20250421`.

Notes:

- The real VLM returned `uncertain` for the tested pairs, which is acceptable
  behavior and correctly triggers fallback/review. The important verified point
  is that the real backend, parser, durable records, and fallback semantics all
  work end-to-end.

## Verification Commands Run

```bash
python -m py_compile \
  scripts/run_em_training.py \
  scripts/run_student_infer_then_round2.py \
  scripts/mine_student_failure_cases.py \
  scripts/run_teacher_plan_tiny_check.py \
  scripts/run_real_teacher_subset_check.py \
  scripts/run_labelcritic_pair_check.py \
  scripts/audit_teacher_readiness.py \
  scripts/repair_round1.py \
  scripts/rerun_round2_mstep.py \
  scripts/evaluate_all_organs.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/mstep_runner.py \
  agent-harness/cli_anything/medai/core/voxtell_student.py \
  agent-harness/cli_anything/medai/medai_cli.py
```

All listed files compiled successfully.

## Still Not Fully Complete

- Full formal E-step with all enabled teachers, all 373 organs, and real
  ShapeKit has not been run in this pass.
- Real VoxTell fine-tuning has not been run in this pass.
- Real VoxTell inference was not run in this pass; only inference dry-run was
  validated.
- The JHU expert fine-label dataset is still unavailable, so true accuracy
  evaluation remains blocked by data access.
- Full 21-model Drive alignment and runnable status should be re-audited before
  the formal all-teacher run.
- Some explicit legacy CLI commands for VISTA3D still exist for reproducibility;
  they are not the default student mainline but should be documented as legacy
  if exposed in user-facing docs.

## Next Best Step

Run a larger non-smoke pipeline check on a tiny real subset:

```text
2 real CT cases
5-10 representative organs
enabled teacher subset
ShapeKit enabled
LabelCritic stub first, then real LabelCritic when VLM server is ready
VoxTell manifest build
VoxTell trainer dry-run
student inference dry-run
failure mining
```

After that passes, scale to the formal 50-case, 373-organ run.

## Continuation Audit - Teacher Meeting Plan Consolidation

Status: completed for planning/source-of-truth consolidation, not for formal
pipeline completion.

User/meeting updates incorporated:

- Current exact engineering target remains `373 organs`, not 377/384.
- Missing or non-one-to-one classes are skipped rather than approximated with
  coarse labels.
- The account exists, but the JHU expert fine-label dataset is not yet
  available; therefore current metrics remain pseudo-label consistency only.
- Teacher outputs must be described as pseudo-label candidates, not true ground
  truth.
- The teacher pipeline is fixed inference/test only; no teacher checkpoint is
  trained during E-step.
- Label Critic must operate immediately after teacher candidate generation for
  multi-candidate case-organ items.
- ShapeKit remains mandatory for formal selected outputs.
- Student stays on the 3D prompt-based/VoxTell-style path; 2D conversion and
  VISTA3D-127-class student framing remain out of the mainline.
- Round2 must compare `student_prev` against Round1 best pseudo label and must
  not automatically overwrite Round1 if the student collapses.
- Hard-case mining must find case-organ pairs the student repeatedly fails to
  learn, using Dice against Round1 selected pseudo label plus empty-mask,
  shape-mismatch, volume-outlier, LabelCritic-uncertain, and ShapeKit-fallback
  signals.

Document update:

- Updated
  `docs/TEACHER_MEETING_IMPLEMENTATION_PLAN_CN_2026_06_05.md` with:
  - a meeting-requirement checklist mapping each teacher instruction to the
    concrete project behavior;
  - the current continuation order, starting with real Label Critic decision
    quality investigation before scaling to formal 50 CT x 373 organs.

Fresh static audit commands:

```bash
python scripts/audit_373_organ_routing.py
python scripts/audit_21_model_drive_alignment.py
```

Observed results:

```json
{
  "routing_status": "success",
  "target_organs": 373,
  "route_requested_organs": 373,
  "statically_unresolvable_target_organs": 0,
  "drive_alignment_status": "success",
  "target_model_count": 21,
  "registry_present_count": 21,
  "enabled_count": 21,
  "models_with_missing_required_files": {},
  "models_with_live_drive_missing_paths": {},
  "organs_with_best_enabled_model": 381,
  "policy_skipped_organ_count": 8,
  "static_exact_merge_candidate_organs": 373,
  "organs_without_best_enabled_model": 3
}
```

Known caveats still present:

- DAPS is present locally/live with `checkpoint_best.pth`, but the older cached
  Drive size manifest still does not contain DAPS entries, so size/checksum
  equivalence is not proven from that old manifest.
- VISTA3D intentionally uses Drive weight evidence plus the
  VISTA3D-Inference-Pipeline wrapper files, so some wrapper files are unmatched
  to the old Drive size manifest by design.
- MOOSE has a Drive-internal command/folder naming inconsistency:
  `run_MOOSE.sh` expects `nnUNetResEncUNetLPlans`, while available checkpoint
  folders are named with `nnUNetPlans`; the registry follows the checkpoint
  folder layout.

Current next engineering step:

- Investigate real Label Critic decision quality. The pipeline and durable
  metadata already work, but recent real VLM checks frequently return
  `vlm_undecided`/`uncertain`. Before scaling to 50 CT x 373 organs, determine
  whether this is genuine uncertainty or a prompt/parser/dual-confirmation
  issue. Any alternative Label Critic prompt mode should be configurable and
  conservative formal fallback behavior must remain auditable.

## Continuation Audit - LabelCritic Decision-Quality Diagnostics

Status: completed for safe diagnostic switches and pair-level evidence; not
completed for improving real LabelCritic winner rate.

Problem investigated:

- Real LabelCritic runs were technically successful but frequently produced
  `winner=uncertain`.
- This blocked the teacher-meeting requirement that multi-teacher candidates be
  judged by Label Critic before ShapeKit/dataset assembly.

Findings:

- Wrapper parsing was not the primary issue. The underlying LabelCritic CSVs
  often either had no comparison rows or had rows with undecided answers.
- For the checked pancreas and liver ePAI-vs-mock pairs, the CSV evidence was:
  - default dual-confirmation mode: `answer=0.5`, `answer_1=-1`,
    `answer_2=-1`;
  - non-dual mode: `answer=-1`;
  - `--no-dice-check` did not convert the pancreas pair into a decisive
    `1/2` winner.
- The stdout evidence indicates Qwen2-VL / LabelCritic sometimes judges the
  organ as not present in the projected body region and enters the
  `Annotation should be zero` branch. This is a prompt/projection/VLM judgment
  limitation, not a reliable winner decision.
- A parallel read-only subagent audit reached the same conclusion: the largest
  sources of `uncertain` are projection-Dice skipping/header-only CSVs and
  dual/prompt parser outputs of `-1`/`0.5`, not a downstream wrapper winner
  mapping bug.

Fixes applied:

- `third_party/LabelCritic-main/RunAPI_single.py`
  - Added `--no_dice_check`.
  - Added `--no_dual_confirmation`.
  - Added `--conservative_dual`.
  - Kept the existing `--simple_prompt_ablation` and made it reachable from
    callers.
  - Default behavior remains the prior formal conservative path.
- `third_party/LabelCritic-main/CompareOrgan.py`
  - Added CLI passthrough for the diagnostic switches.
  - Added `--run_id` so the caller can scope logs to the exact run.
  - Added `--base_output` and `--base_csv` passthrough so artifacts are written
    inside the wrapper work directory instead of shared global folders.
  - Prints when no comparison rows were produced.
- `agent-harness/cli_anything/medai/core/labelcritic_wrapper.py`
  - Passes `run_id`, isolated output/csv directories, and diagnostic options to
    `CompareOrgan.py`.
  - Writes `labelcritic_options` into every result JSON.
  - Parses the scoped run block instead of relying on stale shared-log tails.
  - Distinguishes `parse_status=no_comparison_rows` from
    `parse_status=vlm_undecided` when CSV evidence is available.
- `scripts/run_labelcritic_pair_check.py`
  - Exposes `--no-dice-check`, `--no-dual-confirmation`,
    `--simple-prompt-ablation`, and `--conservative-dual` for controlled
    pair-level experiments.
- `third_party/LabelCritic-main/ErrorDetector.py`
  - Records `skipped_by_projection_dice` in stdout for easier diagnosis.

Verification:

```bash
python -m py_compile \
  third_party/LabelCritic-main/RunAPI_single.py \
  third_party/LabelCritic-main/CompareOrgan.py \
  third_party/LabelCritic-main/ErrorDetector.py \
  agent-harness/cli_anything/medai/core/labelcritic_wrapper.py \
  scripts/run_labelcritic_pair_check.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py

python scripts/audit_373_organ_routing.py
```

Both checks passed.

Pair-level real LabelCritic evidence:

```text
pancreas ePAI vs mock, default:
  output: outputs/audit_21_models/labelcritic_default_after_skip_parse_v2/pancreas_epai_vs_mock.json
  csv: answer=0.5, answer_1=-1, answer_2=-1
  decision: uncertain / vlm_undecided

pancreas ePAI vs mock, --no-dice-check:
  output: outputs/audit_21_models/labelcritic_no_dice_check_ablation_v2/pancreas_epai_vs_mock.json
  csv: answer=0.5, answer_1=-1, answer_2=-1
  decision: uncertain / vlm_undecided

pancreas ePAI vs mock, --no-dual-confirmation:
  output: outputs/audit_21_models/labelcritic_no_dual_ablation/pancreas_epai_vs_mock.json
  csv: answer=-1
  decision: uncertain / vlm_undecided

liver ePAI vs mock, default:
  output: outputs/audit_21_models/labelcritic_liver_epai_mock_default/liver_epai_vs_mock.json
  csv: answer=0.5, answer_1=-1, answer_2=-1
  decision: uncertain / vlm_undecided

liver ePAI vs mock, --no-dual-confirmation:
  output: outputs/audit_21_models/labelcritic_liver_epai_mock_no_dual/liver_epai_vs_mock.json
  csv: answer=-1
  decision: uncertain / vlm_undecided
```

Safety decision:

- Do not force a winner from `-1` or `0.5`.
- Do not make non-dual or no-dice-check mode the formal default.
- Formal multi-teacher selection must continue to treat these cases as
  `label_critic_fallback` and add review metadata.

Next step:

- Improve LabelCritic input quality rather than weakening selection safety:
  inspect generated projection images, test alternate axis/window/prompt wording,
  and consider bypassing the organ-presence gating prompt for already
  organ-specific candidate masks. Only promote a mode to formal use if it
  produces stable, auditable 1/2 decisions on multiple organs/cases.

## Continuation Audit - LabelCritic Organ-Presence Gate Ablation

Status: completed for pair-level diagnosis; not promoted to formal selection.

Hypothesis tested:

- Prior real LabelCritic runs often entered the `Annotation should be zero`
  branch because the initial body-region prompt judged the requested organ as
  absent from the projection.
- We tested whether bypassing that organ-presence gate would turn
  `uncertain` into stable `1/2` winner decisions.

Fixes applied:

- Added diagnostic-only `skip_organ_presence_gate` support:
  - `third_party/LabelCritic-main/RunAPI_single.py`
  - `third_party/LabelCritic-main/CompareOrgan.py`
  - `third_party/LabelCritic-main/ErrorDetector.py`
  - `agent-harness/cli_anything/medai/core/labelcritic_wrapper.py`
  - `scripts/run_labelcritic_pair_check.py`
- The default remains `False`; formal LabelCritic behavior is unchanged.
- Result JSON now records `labelcritic_options.skip_organ_presence_gate`.

Verification:

```bash
python -m py_compile \
  third_party/LabelCritic-main/RunAPI_single.py \
  third_party/LabelCritic-main/CompareOrgan.py \
  third_party/LabelCritic-main/ErrorDetector.py \
  agent-harness/cli_anything/medai/core/labelcritic_wrapper.py \
  scripts/run_labelcritic_pair_check.py

python scripts/audit_373_organ_routing.py
```

Both checks passed.

Real VLM ablation evidence:

```text
pancreas ePAI vs mock, --skip-organ-presence-gate:
  output: outputs/audit_21_models/labelcritic_pancreas_skip_gate/pancreas_epai_vs_mock.json
  csv: answer=0.5, answer_1=2, answer_2=2
  decision: uncertain / vlm_undecided

liver ePAI vs mock, --skip-organ-presence-gate:
  output: outputs/audit_21_models/labelcritic_liver_skip_gate/liver_epai_vs_mock.json
  csv: answer=0.5, answer_1=1, answer_2=1
  decision: uncertain / vlm_undecided

liver ePAI vs mock, --skip-organ-presence-gate --simple-prompt-ablation:
  output: outputs/audit_21_models/labelcritic_liver_skip_gate_simple/liver_epai_vs_mock.json
  csv: answer=0.5, answer_1=2, answer_2=2
  decision: uncertain / vlm_undecided
```

Interpretation:

- Skipping the organ-presence gate changes the failure mode: the VLM no longer
  just returns `-1/-1`; it produces concrete directional answers.
- However, both reversed-order prompts often return the same overlay number
  (`1/1` or `2/2`). In LabelCritic's dual-confirmation design, same-number
  answers after reversing image order indicate order/position bias or
  inconsistency, not a reliable candidate winner.
- Therefore the correct formal behavior is still `uncertain ->
  label_critic_fallback + review`, not forced winner selection.

Safety decision:

- Do not promote `skip_organ_presence_gate` to formal default.
- Do not reinterpret same-number dual answers as a winner.
- Keep this switch available only for diagnostics while improving the
  comparison protocol.

Next step:

- Investigate a more reliable LabelCritic comparison protocol:
  - inspect the generated side-by-side projection images for position bias;
  - test explicit response templates that force `overlay 1` / `overlay 2` /
    `tie` without long free-form summaries;
  - consider using one fixed side-by-side image plus a separate swapped-image
    audit rather than two independent reversed conversations;
  - only accept a mode if reversed-order checks produce order-consistent
    decisions on several organs/cases.

## Continuation Audit - LabelCritic Strict-Choice Prompt Diagnostic

Status: completed for pair-level diagnostic implementation; not promoted to
formal default.

Hypothesis tested:

- Previous `skip_organ_presence_gate` runs showed the VLM could produce
  directional answers, but often returned same-number dual answers such as
  `1/1` or `2/2`, which LabelCritic correctly treats as order-inconsistent.
- We tested whether a shorter forced-choice prompt could reduce free-form
  summary ambiguity and produce usable LabelCritic winners.

Fixes applied:

- Added diagnostic-only `strict_choice_prompt` support:
  - `third_party/LabelCritic-main/RunAPI_single.py`
  - `third_party/LabelCritic-main/CompareOrgan.py`
  - `third_party/LabelCritic-main/ErrorDetector.py`
  - `agent-harness/cli_anything/medai/core/labelcritic_wrapper.py`
  - `scripts/run_labelcritic_pair_check.py`
- The strict-choice prompt forces the model to choose only `overlay 1`,
  `overlay 2`, or `tie`.
- In strict-choice mode, LabelCritic uses the two-image prompt path so both
  overlays are visible in the same VLM request.
- The default remains unchanged; this is not yet formal behavior.

Verification:

```bash
python -m py_compile \
  third_party/LabelCritic-main/RunAPI_single.py \
  third_party/LabelCritic-main/CompareOrgan.py \
  third_party/LabelCritic-main/ErrorDetector.py \
  agent-harness/cli_anything/medai/core/labelcritic_wrapper.py \
  scripts/run_labelcritic_pair_check.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py

python scripts/audit_373_organ_routing.py
```

Both checks passed.

Real VLM evidence:

```text
pancreas ePAI vs mock, --skip-organ-presence-gate --strict-choice-prompt:
  output: outputs/audit_21_models/labelcritic_pancreas_strict_choice/pancreas_epai_vs_mock.json
  csv: answer=2, answer_1=2, answer_2=0.5
  parsed winner: mask2

liver ePAI vs mock, --skip-organ-presence-gate --strict-choice-prompt:
  output: outputs/audit_21_models/labelcritic_liver_strict_choice/liver_epai_vs_mock.json
  csv: answer=1, answer_1=0.5, answer_2=2
  parsed winner: mask1

liver ePAI vs mock, --strict-choice-prompt only:
  output: outputs/audit_21_models/labelcritic_liver_strict_choice_no_skip_gate/liver_epai_vs_mock.json
  csv: answer=0.5, answer_1=-1, answer_2=-1
  parsed winner: uncertain
```

Interpretation:

- `strict_choice_prompt` plus `skip_organ_presence_gate` is the first tested
  mode that produced explicit winners on both liver and pancreas pair checks.
- It still does not satisfy the strongest dual-confirmation condition because
  one of the two reversed-order answers was uncertain in both successful pairs.
  The current non-conservative dual logic accepts the decisive side.
- Therefore this mode is promising for further validation, but not strong
  enough to enable as formal default without more organs/cases and manual visual
  sanity checks.

Safety decision:

- Keep formal default conservative for now.
- Treat `strict_choice_prompt + skip_organ_presence_gate` as an experimental
  candidate mode.
- If used in a small replay, all selected items from this mode should carry
  explicit `labelcritic_options` metadata and remain reviewable until the mode
  is validated across a broader subset.

Next step:

- Run a small multi-organ replay using this experimental mode, not the full
  formal 50 CT x 373 run.
- Inspect generated projection images for a few successful and failed
  decisions.
- Compare strict-choice winners against existing reference/fallback signals
  where available before considering promotion to formal LabelCritic mode.

## Continuation Audit - Closed-Loop Metadata and Gap Completion

This continuation closes the remaining implementation gaps from the teacher
meeting plan:

- Resume now skips a case only when raw predictions, selection metadata, final
  updated masks, and ShapeKit/fallback metadata are complete. Raw predictions
  alone no longer cause a case to be skipped.
- E-step selection metadata now records `prompt`, `source_model`,
  `labelcritic_decision_path`, `dataset_type=pseudo_label_dataset`,
  `ground_truth_status=pseudo_label_candidate`, and normalized
  `quality_status`.
- ShapeKit statuses distinguish successful post-processing from
  `fallback_original`, `unsupported_target`, and debug/dry-run skips. Unsupported
  targets keep the selected pseudo label but are explicitly review/gap items.
- The E-step now writes `pseudo_label_gap_report.json/csv` so missing selected
  masks, ShapeKit fallbacks, unsupported targets, and policy gaps are not
  silently dropped.
- The E-step now writes a run-level `shapekit_report.json` plus per-case
  `annotation_versions/<case_id>/shapekit_report.json` files.
- VoxTell prompt student manifests and the generic M-step manifest now share the
  same pseudo-label source/selection/ShapeKit fields.
- Student failure mining now includes Round1 source metadata, LabelCritic
  winner/fallback information, decision paths, ShapeKit status, and explicit
  review reasons in addition to Dice/empty/volume/shape signals.
- Student failure mining uses `metric_family=pseudo_consistency` for pseudo-label
  comparisons and reserves `fine_label_eval` outputs behind the explicit
  `--fine-label-root` option for future JHU expert labels.

## Continuation Audit - ShapeKit Before LabelCritic Order

The E-step order has been changed from selected-mask ShapeKit post-processing to
candidate-level ShapeKit pre-processing:

```text
teacher/student candidate masks
  -> candidate-level ShapeKit attempt
  -> LabelCritic compares post-ShapeKit candidates
  -> winner copied to annotation_versions/<case_id>/updated/
```

Rationale:

- LabelCritic should compare the masks that would actually enter the pseudo-label
  dataset. If LabelCritic compares raw masks and ShapeKit changes the winner
  afterward, the decision can become stale.
- Candidate-level ShapeKit makes all candidates anatomically normalized before
  visual comparison.
- If ShapeKit fails or has no safe target for a candidate, that candidate falls
  back to its raw mask and records `candidate_shapekit_status`,
  `selected_candidate_shapekit_status`, and review/gap flags.
- The manifest records `comparison_input_stage=post_shapekit_candidate` for
  ShapeKit-enabled non-dry-run comparisons.

Recommended future improvement:

- Add a candidate QC score between ShapeKit and LabelCritic: geometry alignment,
  empty-mask checks, volume-ratio priors by organ, connected-component sanity,
  and optional pairwise Dice clustering. LabelCritic should then adjudicate only
  among candidates that pass basic QC, or receive QC metadata in its prompt.

## Continuation Audit - Strict-Choice Shared Replay Contract

Status: completed for capped shared E-step replay; still experimental.

Fixes applied:

- Added `labelcritic_options` passthrough to
  `agent-harness/cli_anything/medai/core/multimodel_loop.py` so shared
  E-step candidate selection can use diagnostic LabelCritic modes without
  hard-coding them globally.
- Added diagnostic LabelCritic flags to
  `scripts/replay_teacher_candidates_with_labelcritic.py`.
- Implemented `--max-critic-decisions` as a real replay-scope cap by truncating
  the organ list to the first organ(s) expected to generate pairwise decisions.
  This prevents quick diagnostics from unintentionally replaying all 10 organs.

Stub contract check:

```text
command:
  python scripts/replay_teacher_candidates_with_labelcritic.py \
    --critic-backend stub \
    --output-dir outputs/audit_21_models/replay_strict_choice_stub_contract_capped \
    --max-critic-decisions 1 \
    --labelcritic-skip-organ-presence-gate \
    --labelcritic-strict-choice-prompt \
    --no-shapekit

result:
  status=success
  kept_organs=["liver"]
  manifest_items=2
  total_labelcritic_decisions=2
  labelcritic_options.strict_choice_prompt=true
  labelcritic_options.skip_organ_presence_gate=true
```

Real LabelCritic replay without disabling projection-Dice skip:

```text
output:
  outputs/audit_21_models/replay_strict_choice_real_capped_liver

result:
  status=success
  kept_organs=["liver"]
  total_labelcritic_decisions=2
  both critic CSVs had header only
  parse_status=no_comparison_rows
  selection_method=label_critic_fallback for both cases
```

Real LabelCritic replay with no-dice-check:

```text
command:
  python scripts/replay_teacher_candidates_with_labelcritic.py \
    --critic-backend labelcritic \
    --output-dir outputs/audit_21_models/replay_strict_choice_real_capped_liver_no_dice \
    --max-critic-decisions 1 \
    --labelcritic-skip-organ-presence-gate \
    --labelcritic-strict-choice-prompt \
    --labelcritic-no-dice-check \
    --no-shapekit \
    --timeout-sec 600

result:
  status=success
  kept_organs=["liver"]
  total_labelcritic_decisions=3
  manifest_items=2
  review_queue_lines=1
```

Detailed real replay outcome:

```text
PanTS_00000026/liver:
  candidates: epai_20250421, vsmtrans, cads551
  epai_20250421 vs vsmtrans:
    csv: answer=2, answer_1=2, answer_2=0.5
    winner=vsmtrans
  vsmtrans vs cads551:
    csv: answer=1, answer_1=0.5, answer_2=2
    winner=vsmtrans
  final selection_method=label_critic
  selected_model=vsmtrans
  review_flags=[]

PanTS_00000029/liver:
  candidates: epai_20250421, vsmtrans, cads551
  epai_20250421 vs vsmtrans:
    csv: answer=0.5, answer_1=0.5, answer_2=0.5
    winner=uncertain
  final selection_method=label_critic_fallback
  selected_model=epai_20250421
  review_flags=["selection_fallback"]
```

Interpretation:

- The strict-choice mode can now drive the real shared E-step to a true
  `label_critic` selection for at least one real case-organ item.
- `--labelcritic-no-dice-check` is necessary in this replay because otherwise
  high 2D projection Dice causes LabelCritic to skip comparison and produce
  header-only CSVs.
- The mode remains unstable: one of two liver cases still falls back, and the
  successful decisions rely on one decisive side plus one uncertain reversed
  side, not fully decisive dual confirmation.

Safety decision:

- Do not enable this mode as formal default yet.
- Use it only for small capped replay experiments.
- Require projection-image inspection and broader multi-organ validation before
  promoting it into the main formal E-step.

## Continuation Audit - Strict-Choice 2-Case 5-Organ Replay

Status: completed for a small real replay; not sufficient for formal default.

Scope:

- Cases: `PanTS_00000026`, `PanTS_00000029`.
- Organs: liver, pancreas, spleen, kidney_left, kidney_right.
- Candidate sources replayed from existing teacher predictions:
  `epai_20250421`, `vsmtrans`, `cads551`, `cads552`, `cads553`,
  `cads554`, `nnunet_private`.
- LabelCritic backend: real `labelcritic` via local Qwen2-VL vLLM.
- Diagnostic mode:
  - `--labelcritic-strict-choice-prompt`
  - `--labelcritic-skip-organ-presence-gate`
  - `--labelcritic-no-dice-check`
- ShapeKit: disabled for this diagnostic replay only
  (`shapekit_status=skipped_debug_only`).

Command:

```bash
python scripts/replay_teacher_candidates_with_labelcritic.py \
  --critic-backend labelcritic \
  --output-dir outputs/audit_21_models/replay_strict_choice_real_2case_5organ_no_dice \
  --organs liver,pancreas,spleen,kidney_left,kidney_right \
  --labelcritic-skip-organ-presence-gate \
  --labelcritic-strict-choice-prompt \
  --labelcritic-no-dice-check \
  --no-shapekit \
  --timeout-sec 600
```

Summary:

```json
{
  "status": "success",
  "manifest_items": 10,
  "multi_candidate_items": 10,
  "total_labelcritic_decisions": 12,
  "vlm_decision_lines": 12,
  "review_queue_lines": 8,
  "selected_models": ["cads551", "epai_20250421", "nnunet_private", "vsmtrans"]
}
```

Selection outcome:

```text
label_critic selected: 2 / 10 manifest items
label_critic_fallback: 8 / 10 manifest items

pairwise decisions with winner: 4 / 12
pairwise uncertain: 8 / 12
```

Successful LabelCritic selections:

```text
PanTS_00000026/liver:
  candidate_models: epai_20250421, vsmtrans, cads551
  selected_model: cads551
  selection_method: label_critic
  review_flags: []

PanTS_00000029/pancreas:
  candidate_models: epai_20250421, vsmtrans, cads551
  selected_model: epai_20250421
  selection_method: label_critic
  review_flags: []
```

Fallback cases:

```text
PanTS_00000026:
  pancreas, spleen, kidney_left, kidney_right

PanTS_00000029:
  liver, spleen, kidney_left, kidney_right
```

Pairwise parse-status distribution:

```text
better_path_parse: 4
vlm_undecided: 8
```

By organ:

```text
liver: 2 winners, 1 uncertain
pancreas: 2 winners, 1 uncertain
spleen: 0 winners, 2 uncertain
kidney_left: 0 winners, 2 uncertain
kidney_right: 0 winners, 2 uncertain
```

Interpretation:

- The experimental strict-choice mode is a real improvement over the earlier
  all-uncertain behavior: it can produce actual `label_critic` selections in
  the shared E-step.
- The mode is still not robust enough for formal default use. In this 10-item
  replay, only 20% of final case-organ selections were true LabelCritic
  selections; 80% still required fallback/review.
- The failure pattern is organ-dependent. Liver and pancreas show promise;
  kidney and spleen remain mostly undecided under this projection/prompt setup.

Safety decision:

- Keep current formal behavior conservative.
- Continue using fallback + review queue whenever LabelCritic is inconclusive.
- Do not expand to 50 CT x 373 organs with this mode until additional
  projection/prompt work improves stability, especially for kidney/spleen and
  left/right structures.

Next step:

- Inspect projection artifacts for successful liver/pancreas cases and failed
  kidney/spleen cases.
- Decide whether organ-specific prompt templates or alternate projection axes
  are needed before running a larger replay.

## Continuation Audit - Phase 1/2 Repairs

Status: completed for static model/routing audit, not for full formal
50-case inference.

Evidence:

- Added `docs/TEACHER_MEETING_IMPLEMENTATION_PLAN_CN_2026_06_05.md` as the
  Chinese execution plan matching the teacher meeting requirements.
- Strengthened `scripts/audit_21_model_drive_alignment.py`:
  - fixes local path mapping to the cached Drive size manifest;
  - reads `outputs/audit_21_models/live_drive_listing_gdown.json` when
    available;
  - reports live Drive missing paths separately from cached-size-manifest gaps;
  - reports command-parameter mismatches against Drive run scripts/README;
  - records whether actual registry parameters match the checkpoint folder name.
- Generated live Drive listing with `gdown.download_folder(skip_download=True)`,
  read-only/no downloads:

```json
{
  "status": "success",
  "count": 997,
  "output": "outputs/audit_21_models/live_drive_listing_gdown.json"
}
```

- Latest 21-model alignment audit:

```json
{
  "target_model_count": 21,
  "registry_present_count": 21,
  "enabled_count": 21,
  "models_with_missing_required_files": {},
  "models_with_drive_size_mismatches": {},
  "live_drive_listing_available": true,
  "models_with_live_drive_missing_paths": {},
  "organs_with_best_enabled_model": 381,
  "policy_skipped_organ_count": 8,
  "static_exact_merge_candidate_organs": 373,
  "organs_without_best_enabled_model": 3
}
```

Notes:

- DAPS is present in the live Drive listing with `checkpoint_best.pth`,
  `debug.json`, `dataset.json`, `dataset_fingerprint.json`, and `plans.json`.
- The older cached Drive size CSV does not include DAPS, so live presence is
  verified but Drive size/checksum equivalence for DAPS remains unproven without
  a fresh metadata export or re-download comparison.
- VISTA3D is intentionally split across Drive weight evidence
  `VISTA3D/model_bundle.pt` and the local/GitHub VISTA3D-Inference-Pipeline
  files used by the wrapper.
- MOOSE has a Drive-internal inconsistency: `run_MOOSE.sh` says
  `nnUNetResEncUNetLPlans`, while the live Drive checkpoint folders are named
  `nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres`. The registry keeps
  `nnUNetPlans` because that matches the provided checkpoint folder layout.

Routing repair:

- Changed `organ_router.route_organs()` default behavior to use
  `configs/student_3d_prompt_target_organs.json` when no explicit organ list is
  passed.
- Updated `segment-all` help text so default routing is documented as the
  current 373 exact targets, not the full xlsx 384 organs.
- Verified default routing:

```json
{
  "total_requested_organs": 373,
  "no_enabled_candidates": 0,
  "missing_organs": 0,
  "contains_skipped_or_no_route_organs": []
}
```

- Added `scripts/audit_373_organ_routing.py`.
- Latest 373-organ routing audit:

```json
{
  "status": "success",
  "target_organs": 373,
  "unique_target_organs": 373,
  "route_requested_organs": 373,
  "selected_model_keys": 22,
  "statically_unresolvable_target_organs": 0,
  "blocking_keys": []
}
```

Clarification:

- The 22 selected routing keys are the 21 Drive inventory entries plus the
  official `totalsegmentator` CLI route. This is expected because
  TotalSegmentator must be handled through its official license/CLI path rather
  than mixed into the private Drive nnUNet checkpoints.

Regression:

- Ran a 1-case/2-organ tiny mainline check after the routing default change:

```json
{
  "status": "success",
  "estep_total_updated": 2,
  "manifest_items": 2,
  "all_items_have_source_metadata": true,
  "shapekit_statuses": ["skipped_debug_only"]
}
```

This was a fast regression with ShapeKit disabled for speed; it does not replace
the formal ShapeKit-enabled E-step requirement.

Teacher readiness:

- Ran command-construction readiness for the 22 model keys selected by the
  current 373-organ routing pool: the 21 Drive inventory entries plus official
  `totalsegmentator`.

```json
{
  "status": "success",
  "num_models": 22,
  "models_ready_for_command": 22,
  "models_missing_checkpoint_path": [],
  "dry_run_statuses": ["dry_run"],
  "dependencies": {
    "TotalSegmentator": "/home/teacher1/miniconda3/bin/TotalSegmentator",
    "nnUNetv2_predict": "/home/teacher1/miniconda3/bin/nnUNetv2_predict",
    "nnUNetv2_predict_from_modelfolder": "/home/teacher1/miniconda3/bin/nnUNetv2_predict_from_modelfolder"
  }
}
```

## Continuation Audit - Phase 3 Real Teacher Candidate Selection

Status: completed for a small real-teacher subset, not for full-scale formal
E-step.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `epai_20250421`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: liver, pancreas, spleen.
- ShapeKit: enabled.
- LabelCritic backend: `stub` for deterministic wiring validation.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["epai_20250421", "vsmtrans", "mock_seg"],
  "organs": ["liver", "pancreas", "spleen"],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 3
  },
  "manifest_items": 3,
  "selected_models": ["vsmtrans"],
  "candidate_model_sets": [["epai_20250421", "vsmtrans", "mock_seg"]],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 3,
  "all_items_have_source_metadata": true
}
```

Fix applied:

- `multimodel_loop.py` now writes `selection_fallback` items to
  `review_queue.jsonl` even when the fallback-selected mask has high reference
  Dice. High pseudo-label/reference consistency must not hide the fact that
  LabelCritic was inconclusive or stubbed.
- `mstep_runner.py` and `voxtell_student.py` now preserve `fallback_reason` in
  generic and VoxTell training manifests.

Verification after the fix:

```json
{
  "manifest_items": 3,
  "fallback_reasons_present": true,
  "review_queue_lines": 3,
  "review_reasons": ["fallback pseudo-label selection requires review"],
  "shapekit_statuses": ["success"]
}
```

Notes:

- This validates real multi-teacher candidate collection, durable LabelCritic
  records, ShapeKit success, source-aware manifest metadata, and explicit
  review queue behavior.
- It does not validate real VLM quality because `critic_backend=stub` was used.
- At this checkpoint no vLLM server is running on `localhost:8000`, and the A100
  GPU was idle; a short real-LabelCritic check can be run later when needed.

VoxTell handoff validation:

- Built a VoxTell prompt-student manifest from the same real-teacher E-step.
- Verified the manifest preserves:
  - `selected_model`
  - `candidate_models`
  - `selection_method`
  - `selection_status`
  - `fallback_reason`
  - `labelcritic_records`
  - `shapekit_status`
  - `review_flags`
  - `quality_flags`
  - `dataset_role = pseudo_label`
  - `ground_truth_status = pseudo_label_candidate`

```json
{
  "status": "success",
  "num_items": 3,
  "num_cases": 1,
  "num_items_missing_image": 0,
  "source_metadata_available": true,
  "fallback_reason_present": true
}
```

- VoxTell trainer dry-run on this manifest succeeded:

```json
{
  "stage": "train_voxtell_prompt_student",
  "status": "dry_run",
  "num_manifest_items": 3,
  "model_files_ok": true,
  "epochs": 1,
  "learning_rate": 5e-05
}
```

## Continuation Audit - Expanded Real Teacher Subset

Status: completed for the next-scale small subset.

Scope:

- Cases: `PanTS_00000026`, `PanTS_00000029`.
- Real teachers: `epai_20250421`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: liver, pancreas, spleen, kidney_left, kidney_right.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["epai_20250421", "vsmtrans", "mock_seg"],
  "organs": ["liver", "pancreas", "spleen", "kidney_left", "kidney_right"],
  "estep": {
    "status": "success",
    "total_updated": 10,
    "total_labelcritic_decisions": 10
  },
  "manifest_items": 10,
  "selected_models": ["epai_20250421", "vsmtrans"],
  "candidate_model_sets": [["epai_20250421", "vsmtrans", "mock_seg"]],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 10,
  "all_items_have_source_metadata": true
}
```

Detailed checks:

```json
{
  "manifest_items": 10,
  "cases": ["PanTS_00000026", "PanTS_00000029"],
  "organs": ["kidney_left", "kidney_right", "liver", "pancreas", "spleen"],
  "selection_statuses": ["fallback"],
  "fallback_reason_missing": [],
  "review_queue_lines": 10,
  "review_reasons": {
    "fallback pseudo-label selection requires review": 10
  },
  "vlm_decision_lines": 10,
  "shapekit_statuses": ["success"]
}
```

VoxTell handoff on expanded subset:

```json
{
  "status": "success",
  "num_items": 10,
  "num_cases": 2,
  "num_items_missing_image": 0,
  "metadata_fields_missing": 0,
  "review_flags": [["selection_fallback"]]
}
```

- VoxTell trainer dry-run succeeded on the expanded prompt manifest:

```json
{
  "stage": "train_voxtell_prompt_student",
  "status": "dry_run",
  "num_manifest_items": 5,
  "model_files_ok": true,
  "learning_rate": 5e-05
}
```

Notes:

- This confirms the E-step chain is stable across multiple real CT cases for
  abdominal organs using two real teachers.
- It still uses stub LabelCritic, so it validates pipeline wiring and metadata,
  not real VLM judgment quality.
- This is still far below the formal 50-case, 373-organ requirement.

## Continuation Audit - CADS551 Added To Abdominal Teacher Pool

Status: completed for one real CT.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `cads551`, `vsmtrans`, `epai_20250421`.
- Synthetic comparator: `mock_seg`.
- Organs: liver, pancreas, spleen, kidney_left, kidney_right.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads551", "vsmtrans", "epai_20250421", "mock_seg"],
  "estep": {
    "status": "success",
    "total_updated": 5,
    "total_labelcritic_decisions": 5
  },
  "manifest_items": 5,
  "selected_models": ["cads551", "vsmtrans"],
  "candidate_model_sets": [["cads551", "vsmtrans", "epai_20250421", "mock_seg"]],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 5,
  "review_queue_lines": 5,
  "all_items_have_source_metadata": true
}
```

Notes:

- `cads551` produced real masks successfully and can participate in the abdominal
  candidate pool dictated by xlsx routing.
- In stub LabelCritic mode, the first uncertain pairwise comparison triggers
  fallback to the reference-Dice best candidate. This is explicitly recorded as
  `selection_fallback` with `fallback_reason` and review queue entries.

## Continuation Audit - DAPS Real Teacher Check

Status: completed for DAPS wrapper/checkpoint validation.

Scope:

- Case: `PanTS_00000026`.
- Teacher: `daps`.
- Comparator: `mock_seg`.
- Checkpoint: `checkpoint_best.pth`.
- Initial organs: bronchus, thyroid_left, thyroid_right, heart_myocardium,
  breast_left, artery_subclavian_left.
- Focus rerun organs: heart_myocardium, breast_left.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Static routing facts:

- DAPS covers 30 current 373 target organs.
- DAPS `dataset.json` contains 30 non-background labels, matching the registry
  `covered_organs`.
- Registry uses:
  - dataset id `1347`
  - trainer `nnUNetTrainer`
  - plans `nnUNetResEncUNetLPlans`
  - folds `all`
  - checkpoint `checkpoint_best.pth`

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["daps", "mock_seg"],
  "organs": ["heart_myocardium", "breast_left"],
  "estep": {
    "status": "success",
    "total_updated": 2,
    "total_labelcritic_decisions": 0
  },
  "manifest_items": 2,
  "selected_models": ["daps"],
  "candidate_model_sets": [["daps"]],
  "shapekit_statuses": ["fallback_original"],
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- DAPS real inference succeeded and produced 13 non-empty masks on
  `PanTS_00000026`.
- For the broader 6-organ probe, bronchus/thyroid/subclavian candidates were
  missing because no non-empty mask was produced for this abdomen CT; these were
  recorded as `missing_candidate` review items.
- `heart_myocardium` and `breast_left` were successfully selected from DAPS.
- ShapeKit fell back to the selected DAPS masks because this non-abdominal
  selection set lacked ShapeKit's required affine reference `liver.nii.gz`.

Fix applied:

- `multimodel_loop.py` now writes ShapeKit fallback review queue items with
  `selected_model`, `candidate_models`, `selection_method`, `selection_status`,
  `review_flags`, `quality_flags`, and `shapekit_reason`.

Verification after the fix:

```json
{
  "manifest_items": 2,
  "shapekit_statuses": ["fallback_original"],
  "review_queue_lines": 2,
  "review_flags": [["shapekit_fallback"]],
  "quality_flags": [["single_candidate", "shapekit_fallback"]]
}
```

## Continuation Audit - CADS552 Real Teacher Check

Status: completed for CADS552 wrapper/checkpoint validation on a small spine
subset.

Scope:

- Case: `PanTS_00000026`.
- Teacher: `cads552`.
- Comparator: `mock_seg`.
- Organs: `vertebrae_L1`, `vertebrae_L2`, `vertebrae_T12`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads552", "mock_seg"],
  "organs": ["vertebrae_L1", "vertebrae_L2", "vertebrae_T12"],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 0
  },
  "manifest_items": 3,
  "selected_models": ["cads552"],
  "candidate_model_sets": [["cads552"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 0,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `cads552` real inference succeeded on a real CT and produced 9 masks.
- The three requested vertebra organs were selected from `cads552` as
  `single_teacher_default` candidates.
- ShapeKit fell back to the selected teacher masks because the spine-only case
  layout did not contain ShapeKit's required affine reference
  `liver.nii.gz`.
- This fallback is expected for this non-abdominal subset and is recorded as a
  review risk rather than treated as a silent success.

Verification:

```json
{
  "manifest_items": 3,
  "review_queue_lines": 3,
  "selected_model": "cads552",
  "selection_method": "single_teacher_default",
  "shapekit_status": "fallback_original",
  "review_flags": ["shapekit_fallback"],
  "quality_flags": ["single_candidate", "shapekit_fallback"],
  "shapekit_reason": "ShapeKit requires affine reference mask 'liver.nii.gz' in each case."
}
```

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_cads552_spine_shapekit \
  --num-cases 1 \
  --models cads552,mock_seg \
  --organs vertebrae_L1,vertebrae_L2,vertebrae_T12 \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  scripts/run_real_teacher_subset_check.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/mstep_runner.py \
  agent-harness/cli_anything/medai/core/voxtell_student.py
```

## Continuation Audit - ShapeKit Affine Reference Auto-Config Fix

Status: completed for the main ShapeKit wrapper.

Problem:

- ShapeKit's default config uses `liver.nii.gz` as the affine reference.
- The wrapper previously failed precheck whenever a selected pseudo-label subset
  did not contain `liver.nii.gz`.
- That behavior is too brittle for the teacher-meeting requirement that all
  selected outputs go through ShapeKit, because small validation subsets and
  non-abdominal organ groups may legitimately omit liver.

Fix applied:

- Updated `agent-harness/cli_anything/medai/core/shapekit_runner.py`.
- The wrapper still does not mutate `third_party/ShapeKit-main`; it edits only a
  temporary runtime copy.
- With `auto_config=True`, the wrapper now:
  - checks affine reference availability per case;
  - keeps `liver.nii.gz` when every case has it;
  - otherwise auto-selects a mask that exists in every case;
  - writes the chosen reference into the temporary ShapeKit config;
  - records `affine_reference_original`, `affine_reference_used`,
    `affine_reference_source`, and missing default-reference cases in
    `config_check`.
- If no common mask exists across cases, ShapeKit still fails explicitly and
  records the reason.

Direct validation:

```json
{
  "status": "success",
  "target_organs_used": ["bladder", "colon", "duodenum"],
  "affine_reference_original": "liver.nii.gz",
  "affine_reference_used": "bladder.nii.gz",
  "affine_reference_source": "auto_selected_common_mask",
  "missing_default_reference_cases": ["PanTS_00000026"],
  "num_masks_total": 3
}
```

Impact:

- This reduces false ShapeKit fallback in small real-teacher subset checks.
- It also makes the formal 373-organ workflow more robust when a case-organ
  subset lacks liver but still has valid selected masks.

## Continuation Audit - CADS553 Multi-Candidate Real Teacher Check

Status: completed after the ShapeKit affine-reference fix.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `cads553`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: `colon`, `duodenum`, `bladder`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads553", "vsmtrans", "mock_seg"],
  "organs": ["colon", "duodenum", "bladder"],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 2
  },
  "manifest_items": 3,
  "selected_models": ["vsmtrans"],
  "candidate_model_sets": [["cads553", "vsmtrans"], ["vsmtrans"]],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 2,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `cads553` real inference succeeded on a real CT and produced 14 masks.
- `vsmtrans` real inference succeeded and produced 25 masks.
- `colon` and `duodenum` formed true multi-teacher candidate sets from
  `cads553` and `vsmtrans`.
- `bladder` had a single available candidate from `vsmtrans`.
- The stub LabelCritic produced two durable records for the two multi-candidate
  organs; because the backend is intentionally stubbed, both selections are
  recorded as `label_critic_fallback` and queued for review.
- ShapeKit succeeded for all three selected masks after the affine-reference
  wrapper fix.

Manifest verification:

```json
{
  "manifest_items": 3,
  "shapekit_statuses": ["success"],
  "review_queue_lines": 2,
  "selection_fallback_organs": ["colon", "duodenum"],
  "shapekit_fallback_items": 0,
  "all_items_have_source_metadata": true
}
```

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_cads553_vsmtrans_mock_shapekit_after_affinefix \
  --num-cases 1 \
  --models cads553,vsmtrans,mock_seg \
  --organs colon,duodenum,bladder \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  agent-harness/cli_anything/medai/core/shapekit_runner.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  scripts/run_real_teacher_subset_check.py

python scripts/audit_373_organ_routing.py
```

## Continuation Audit - ShapeKit Calibration-Only Mask Fix

Status: completed for the main E-step/ShapeKit handoff.

Problem:

- The previous ShapeKit affine-reference fix allowed non-liver subsets to choose
  an alternate affine reference.
- However, some ShapeKit organ-specific post-processors use `liver` not only as
  the affine reference, but also as an anatomical calibration mask for left/right
  reassignment.
- `femur`, `kidney`, `lung`, and `adrenal_gland` family post-processing can
  depend on this liver calibration mask.
- A femur-only CADS554 subset produced femur masks, but ShapeKit printed a
  traceback because `post_processing_femur()` called
  `reassign_left_right_based_on_liver()` with no liver mask.

Fix applied:

- Updated `agent-harness/cli_anything/medai/core/multimodel_loop.py`.
- Before calling ShapeKit, the E-step now injects a `liver.nii.gz`
  calibration-only auxiliary mask when selected organs include a known
  liver-dependent ShapeKit family and a liver prediction is available from one
  of the already-run teachers.
- The auxiliary liver mask is written only to `selected_pre_shapekit` so ShapeKit
  can use it internally.
- It is explicitly recorded as:
  - `dataset_role = shapekit_calibration_only`
  - `included_in_training_manifest = false`
- It does not become a selected pseudo-label and does not appear as an item in
  `training_manifest.json`.

Smoke validation:

```json
{
  "organ": "liver",
  "source_model": "mock",
  "dataset_role": "shapekit_calibration_only",
  "included_in_training_manifest": false,
  "liver_exists_in_selected_pre_shapekit": true
}
```

## Continuation Audit - CADS554 Femur Multi-Candidate Real Teacher Check

Status: completed after the ShapeKit calibration-only mask fix.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `cads554`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: `femur_left`, `femur_right`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads554", "vsmtrans", "mock_seg"],
  "organs": ["femur_left", "femur_right"],
  "estep": {
    "status": "success",
    "total_updated": 2,
    "total_labelcritic_decisions": 2
  },
  "manifest_items": 2,
  "selected_models": ["vsmtrans"],
  "candidate_model_sets": [["cads554", "vsmtrans"]],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 2,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `cads554` real inference succeeded on a real CT and produced 16 masks.
- `vsmtrans` real inference succeeded and produced 25 masks.
- `femur_left` and `femur_right` formed true multi-teacher candidate sets from
  `cads554` and `vsmtrans`.
- The stub LabelCritic produced two durable records; both selections are
  recorded as `label_critic_fallback` and queued for review because the backend
  is intentionally stubbed.
- ShapeKit succeeded for both femur masks after the calibration-only liver mask
  was injected.

Manifest and ShapeKit verification:

```json
{
  "manifest_items": 2,
  "manifest_organs": ["femur_left", "femur_right"],
  "shapekit_statuses": ["success"],
  "selected_pre_shapekit_masks": ["femur_left.nii.gz", "femur_right.nii.gz", "liver.nii.gz"],
  "calibration_masks": [
    {
      "organ": "liver",
      "source_model": "vsmtrans",
      "dataset_role": "shapekit_calibration_only",
      "included_in_training_manifest": false
    }
  ],
  "shapekit_fallback_items": 0,
  "review_queue_lines": 4
}
```

Notes:

- The four review queue entries are expected in this stub run:
  low reference Dice and LabelCritic fallback for each femur side.
- No ShapeKit fallback remains after the fix.
- This is not true accuracy evidence; it validates teacher runnable status,
  multi-candidate selection wiring, ShapeKit handoff, and manifest integrity.

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_cads554_femur_vsmtrans_mock_shapekit_after_calibrationfix \
  --num-cases 1 \
  --models cads554,vsmtrans,mock_seg \
  --organs femur_left,femur_right \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/shapekit_runner.py \
  scripts/run_real_teacher_subset_check.py

python scripts/audit_373_organ_routing.py
```

## Continuation Audit - ShapeKit Unsupported-Target Flag

Status: completed for E-step metadata.

Problem:

- Some current 373 target organs are valid pseudo-label targets but are not
  supported by the local ShapeKit config/post-processing targets.
- Example: rib subclasses such as `rib_left_6` and `rib_right_6`.
- ShapeKit correctly reports `No safe ShapeKit target organs detected`, but the
  old manifest only marked this as a generic `shapekit_fallback`.
- For formal 373-organ accounting, unsupported ShapeKit targets should be
  distinguishable from crashes, missing files, or calibration failures.

Fix applied:

- Updated `agent-harness/cli_anything/medai/core/multimodel_loop.py`.
- When ShapeKit fallback reason is `No safe ShapeKit target organs detected`,
  the E-step now adds:
  - `shapekit_unsupported_target` to `review_flags`;
  - `shapekit_unsupported_target` to `quality_flags`;
  - `shapekit_unsupported_target: true` to the review queue item.
- The mask remains usable as the selected teacher pseudo-label candidate, but it
  is clearly marked as not post-processed by ShapeKit.

## Continuation Audit - CADS555 Rib Real Teacher Check

Status: completed for CADS555 wrapper/checkpoint validation and unsupported
ShapeKit target metadata.

Scope:

- Case: `PanTS_00000026`.
- Teacher: `cads555`.
- Comparator: `mock_seg`.
- Requested organs: `rib_left_1`, `rib_left_6`, `rib_right_1`, `rib_right_6`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads555", "mock_seg"],
  "organs": ["rib_left_1", "rib_left_6", "rib_right_1", "rib_right_6"],
  "estep": {
    "status": "success",
    "total_updated": 2,
    "total_labelcritic_decisions": 0
  },
  "manifest_items": 2,
  "selected_models": ["cads555"],
  "candidate_model_sets": [["cads555"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 0,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `cads555` real inference succeeded on a real CT and produced 14 rib masks.
- For the chosen abdomen CT, `rib_left_6` and `rib_right_6` produced candidate
  masks and were selected from `cads555`.
- `rib_left_1` and `rib_right_1` produced no candidate masks in this case and
  were recorded as `missing_candidate`.
- ShapeKit does not currently provide a safe target group for rib subclasses,
  so the two selected rib masks fell back to the selected teacher masks.
- This fallback is now clearly marked as `shapekit_unsupported_target`.

Manifest and review verification:

```json
{
  "manifest_items": 2,
  "manifest_organs": ["rib_left_6", "rib_right_6"],
  "selected_model": "cads555",
  "selection_method": "single_teacher_default",
  "shapekit_status": "fallback_original",
  "shapekit_reason": "No safe ShapeKit target organs detected",
  "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
  "quality_flags": ["single_candidate", "shapekit_fallback", "shapekit_unsupported_target"],
  "missing_candidate_organs": ["rib_left_1", "rib_right_1"],
  "review_queue_lines": 4
}
```

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_cads555_ribs_mock_shapekit_after_unsupportedflag \
  --num-cases 1 \
  --models cads555,mock_seg \
  --organs rib_left_1,rib_left_6,rib_right_1,rib_right_6 \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/shapekit_runner.py \
  scripts/run_real_teacher_subset_check.py

python scripts/audit_373_organ_routing.py
```

## Continuation Audit - CADS556 Mixed-Structure Real Teacher Check

Status: completed for CADS556 wrapper/checkpoint validation on a mixed organ
subset.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `cads556`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: `prostate`, `rectum`, `sternum`, `spinal_canal`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads556", "vsmtrans", "mock_seg"],
  "organs": ["prostate", "rectum", "sternum", "spinal_canal"],
  "estep": {
    "status": "success",
    "total_updated": 4,
    "total_labelcritic_decisions": 2
  },
  "manifest_items": 4,
  "selected_models": ["cads556", "vsmtrans"],
  "candidate_model_sets": [["cads556"], ["cads556", "vsmtrans"]],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 2,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `cads556` real inference succeeded on a real CT and produced 15 masks.
- Requested target outputs were available for all four organs.
- `prostate` and `rectum` formed multi-teacher candidate sets from `cads556`
  and `vsmtrans`.
- `sternum` and `spinal_canal` were single-candidate selections from `cads556`.
- The stub LabelCritic produced two durable records; both multi-candidate
  selections are recorded as `label_critic_fallback` and queued for review.
- ShapeKit completed successfully.

Manifest verification:

```json
{
  "manifest_items": 4,
  "manifest_organs": ["prostate", "rectum", "spinal_canal", "sternum"],
  "prostate": {
    "selected_model": "vsmtrans",
    "candidate_models": ["cads556", "vsmtrans"],
    "selection_method": "label_critic_fallback",
    "shapekit_status": "success"
  },
  "rectum": {
    "selected_model": "cads556",
    "candidate_models": ["cads556", "vsmtrans"],
    "selection_method": "label_critic_fallback",
    "shapekit_status": "success"
  },
  "spinal_canal": {
    "selected_model": "cads556",
    "candidate_models": ["cads556"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "success"
  },
  "sternum": {
    "selected_model": "cads556",
    "candidate_models": ["cads556"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "success"
  },
  "review_queue_lines": 2
}
```

ShapeKit detail:

- Auto-config selected `target_organs_used = ["prostate"]`.
- `rectum`, `sternum`, and `spinal_canal` were carried through in ShapeKit's
  output layout, but should be interpreted as ShapeKit pass-through outputs
  rather than dedicated ShapeKit post-processing targets.
- No ShapeKit fallback occurred in this run.

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_cads556_mixed_vsmtrans_mock_shapekit \
  --num-cases 1 \
  --models cads556,vsmtrans,mock_seg \
  --organs prostate,rectum,sternum,spinal_canal \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/shapekit_runner.py \
  scripts/run_real_teacher_subset_check.py

python scripts/audit_373_organ_routing.py
```

## Continuation Audit - Runtime Alias Candidate Lookup Fix

Status: completed for E-step candidate collection.

Problem:

- Static routing audit correctly used `configs/model_label_aliases.json` to
  prove that local labels can resolve to global organ names.
- Runtime E-step candidate collection still looked only for
  `<global_organ>.nii.gz`.
- This created a mismatch: an organ could pass static alias audit but still be
  treated as missing during real E-step.
- Example:
  - global organ: `subcutaneous_adipose_tissue`
  - local CADS559/SAROS output: `subcutaneous_tissue.nii.gz`
  - alias config: `subcutaneous_tissue -> subcutaneous_adipose_tissue`
  - before fix: real E-step missed the candidate.

Fix applied:

- Updated `agent-harness/cli_anything/medai/core/multimodel_loop.py`.
- Candidate lookup now checks:
  - direct global mask name first;
  - then reverse `local_to_global` aliases from
    `configs/model_label_aliases.json`.
- `dice_metrics.csv` now includes an `alias_match` field so runtime alias use is
  auditable.

Validation:

```json
{
  "global_organ": "subcutaneous_adipose_tissue",
  "local_mask": "subcutaneous_tissue.nii.gz",
  "model": "cads559",
  "alias_match": "local_alias:subcutaneous_tissue",
  "candidate_exists": true
}
```

## Continuation Audit - CADS559/SAROS Coarse Region Real Teacher Check

Status: completed after the runtime alias candidate lookup fix.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `cads559`, `saros_nnunet`.
- Synthetic comparator: `mock_seg`.
- Organs: `abdominal_cavity`, `muscle`,
  `subcutaneous_adipose_tissue`, `mediastinum`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["cads559", "saros_nnunet", "mock_seg"],
  "organs": ["abdominal_cavity", "muscle", "subcutaneous_adipose_tissue", "mediastinum"],
  "estep": {
    "status": "success",
    "total_updated": 4,
    "total_labelcritic_decisions": 4
  },
  "manifest_items": 4,
  "selected_models": ["cads559"],
  "candidate_model_sets": [["cads559", "saros_nnunet"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 4,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `cads559` real inference succeeded and produced 8 coarse region/tissue masks.
- `saros_nnunet` real inference succeeded and produced 8 coarse region/tissue
  masks.
- All four requested organs formed multi-teacher candidate sets from CADS559 and
  SAROS after runtime alias lookup was fixed.
- `subcutaneous_adipose_tissue` uses the local output
  `subcutaneous_tissue.nii.gz` for both CADS559 and SAROS; this is now visible
  in `dice_metrics.csv` as `alias_match = local_alias:subcutaneous_tissue`.
- The stub LabelCritic produced four durable records; all selections are
  recorded as `label_critic_fallback` and queued for review.
- ShapeKit does not have safe targets for these coarse region/tissue masks, so
  all four selected masks are marked with `shapekit_unsupported_target`.

Manifest and alias verification:

```json
{
  "manifest_items": 4,
  "manifest_organs": [
    "abdominal_cavity",
    "mediastinum",
    "muscle",
    "subcutaneous_adipose_tissue"
  ],
  "subcutaneous_adipose_tissue": {
    "candidate_models": ["cads559", "saros_nnunet"],
    "candidate_paths": [
      ".../cads559/.../segmentations/subcutaneous_tissue.nii.gz",
      ".../saros_nnunet/.../segmentations/subcutaneous_tissue.nii.gz"
    ],
    "alias_match": "local_alias:subcutaneous_tissue"
  },
  "review_flags": [
    "selection_fallback",
    "shapekit_fallback",
    "shapekit_unsupported_target"
  ]
}
```

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_cads559_saros_coarse_shapekit_after_aliascsv \
  --num-cases 1 \
  --models cads559,saros_nnunet,mock_seg \
  --organs abdominal_cavity,muscle,subcutaneous_adipose_tissue,mediastinum \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/shapekit_runner.py \
  scripts/run_real_teacher_subset_check.py

python scripts/audit_373_organ_routing.py
```

## Continuation Audit - CADS557/CADS558 Direct Forward Probes

Status: completed for wrapper/checkpoint forward-pass validation, but not for
useful E-step candidate validation because the currently available CT data is
PanTS abdomen rather than brain/head-neck CT.

Available data check:

- Current local case list and CT files are PanTS abdomen cases.
- No dedicated head/neck or brain CT case was found in the current workspace
  during this pass.

CADS557 scope:

- Model: `cads557`.
- Dataset: `Dataset557_Brain257`.
- Case: `PanTS_00000026`.
- Mode: direct `nnunetv2_predict_and_split.py` probe.

CADS557 evidence:

```json
{
  "status": "success",
  "case_id": "PanTS_00000026",
  "num_local_labels": 9,
  "num_written": 0,
  "written_masks": [],
  "labels_available": [
    "white matter",
    "gray matter",
    "csf",
    "scalp",
    "eye balls",
    "compact bone",
    "spongy bone",
    "blood",
    "head muscles"
  ]
}
```

CADS558 scope:

- Model: `cads558`.
- Dataset: `Dataset558_OAR258`.
- Case: `PanTS_00000026`.
- Mode: direct `nnunetv2_predict_and_split.py` probe.

CADS558 evidence:

```json
{
  "status": "success",
  "case_id": "PanTS_00000026",
  "num_local_labels": 29,
  "num_written": 0,
  "written_masks": [],
  "sample_labels_available": [
    "OAR_A_Carotid_L",
    "OAR_A_Carotid_R",
    "OAR_Arytenoid",
    "OAR_Bone_Mandible",
    "OAR_Brainstem",
    "OAR_BuccalMucosa",
    "OAR_Cavity_Oral",
    "OAR_Cochlea_L",
    "OAR_Cochlea_R",
    "OAR_Cricopharyngeus"
  ]
}
```

Interpretation:

- Both wrappers/checkpoints are runnable on the server.
- Both models produced no non-empty masks on the abdomen CT probe, which is
  consistent with their brain/head-neck target domains.
- This is not evidence that the models cannot produce their target organs; it is
  evidence that the current local CT sample is not suitable for validating these
  domains end-to-end.
- A head/neck or brain CT case is required to validate CADS557/CADS558 E-step
  candidate generation, alias mapping, and ShapeKit/review metadata on actual
  non-empty target masks.

Validation commands:

```bash
python scripts/nnunetv2_predict_and_split.py \
  --image /home/teacher1/JHU-project1/medical_agent/data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz \
  --output outputs/audit_21_models/direct_probe_cads557_pants26 \
  --dataset-id 557 \
  --nnunet-results checkpoints/CADS_series/CADS_series \
  --dataset-json checkpoints/CADS_series/CADS_series/Dataset557_Brain257/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/dataset.json \
  --trainer nnUNetTrainerNoMirroring \
  --plans nnUNetResEncUNetLPlans \
  --configuration 3d_fullres \
  --folds all \
  --checkpoint-name checkpoint_final.pth \
  --per-model-dir outputs/audit_21_models/direct_probe_cads557_pants26/per_model/cads557

python scripts/nnunetv2_predict_and_split.py \
  --image /home/teacher1/JHU-project1/medical_agent/data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz \
  --output outputs/audit_21_models/direct_probe_cads558_pants26 \
  --dataset-id 558 \
  --nnunet-results checkpoints/CADS_series/CADS_series \
  --dataset-json checkpoints/CADS_series/CADS_series/Dataset558_OAR258/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/dataset.json \
  --trainer nnUNetTrainerNoMirroring \
  --plans nnUNetResEncUNetLPlans \
  --configuration 3d_fullres \
  --folds all \
  --checkpoint-name checkpoint_final.pth \
  --per-model-dir outputs/audit_21_models/direct_probe_cads558_pants26/per_model/cads558

python -m py_compile \
  scripts/nnunetv2_predict_and_split.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  scripts/run_real_teacher_subset_check.py

python scripts/audit_373_organ_routing.py
```

## Continuation Audit - MOOSE888 Real Teacher Check

Status: completed for MOOSE888 wrapper/checkpoint validation and local plans
parameter validation.

Context:

- The Drive audit reports a command-parameter mismatch for MOOSE:
  `run_MOOSE.sh` expects `nnUNetResEncUNetLPlans`, while the local/live
  checkpoint folder is named
  `nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres`.
- The registry intentionally uses `plans = nnUNetPlans` to match the actual
  checkpoint folder.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `moose888`, `cads553`.
- Synthetic comparator: `mock_seg`.
- Organs: `heart_myocardium`, `inferior_vena_cava`,
  `portal_vein_and_splenic_vein`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["moose888", "cads553", "mock_seg"],
  "organs": ["heart_myocardium", "inferior_vena_cava", "portal_vein_and_splenic_vein"],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 1
  },
  "manifest_items": 3,
  "selected_models": ["moose888"],
  "candidate_model_sets": [["moose888"], ["moose888", "cads553"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 1,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `moose888` real inference succeeded on a real CT and produced 12 masks.
- This confirms that the local registry's `nnUNetPlans` setting is runnable for
  the mounted MOOSE888 checkpoint layout.
- `heart_myocardium` formed a multi-teacher candidate set from `moose888` and
  `cads553`.
- `inferior_vena_cava` and `portal_vein_and_splenic_vein` were selected from
  `moose888` as single-teacher candidates in this run.
- Runtime alias lookup correctly resolved MOOSE's local
  `portal_splenic_vein.nii.gz` output to global
  `portal_vein_and_splenic_vein`.
- ShapeKit does not have safe targets for these selected cardiac/vascular masks
  in this subset, so the masks fell back to selected teacher outputs and are
  marked as `shapekit_unsupported_target`.

Manifest and alias verification:

```json
{
  "manifest_items": 3,
  "manifest_organs": [
    "heart_myocardium",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein"
  ],
  "portal_vein_and_splenic_vein": {
    "selected_model": "moose888",
    "candidate_models": ["moose888"],
    "alias_match": "local_alias:portal_splenic_vein",
    "prediction": ".../moose888/.../segmentations/portal_splenic_vein.nii.gz"
  },
  "shapekit_status": "fallback_original",
  "shapekit_reason": "No safe ShapeKit target organs detected"
}
```

Conclusion:

- MOOSE888 should keep `plans = nnUNetPlans` in `configs/model_registry.yaml`
  because that matches the actual local checkpoint folder and has now run
  successfully.
- The Drive script mismatch remains documented as an upstream/Drive internal
  inconsistency, not a local runnable blocker.

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_moose888_cardiac_cads553_mock_shapekit \
  --num-cases 1 \
  --models moose888,cads553,mock_seg \
  --organs heart_myocardium,inferior_vena_cava,portal_vein_and_splenic_vein \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900

python -m py_compile \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  scripts/run_real_teacher_subset_check.py \
  scripts/audit_21_model_drive_alignment.py

python scripts/audit_373_organ_routing.py
python scripts/audit_21_model_drive_alignment.py
```

## Continuation Audit - MOOSE666 Real Teacher Check

Status: completed for MOOSE666 wrapper/checkpoint validation and local plans
parameter validation.

Context:

- The Drive audit reports the same MOOSE command-parameter mismatch for
  MOOSE666 as for MOOSE888: `run_MOOSE.sh` expects
  `nnUNetResEncUNetLPlans`, while the local/live checkpoint folder is named
  `nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres`.
- The registry intentionally uses `plans = nnUNetPlans` to match the actual
  checkpoint folder.
- This check validates local runnable behavior and E-step integration; it does
  not validate true segmentation accuracy.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `moose666`, `cads554`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: `femur_left`, `femur_right`, `scapula_left`, `clavicula_left`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["moose666", "cads554", "vsmtrans", "mock_seg"],
  "organs": ["femur_left", "femur_right", "scapula_left", "clavicula_left"],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 2
  },
  "manifest_items": 3,
  "selected_models": ["cads554", "moose666"],
  "candidate_model_sets": [
    ["cads554"],
    ["moose666", "cads554", "vsmtrans"]
  ],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 2,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `moose666` real inference succeeded on a real CT and produced usable femur
  masks.
- `femur_left` and `femur_right` formed multi-teacher candidate sets from
  `moose666`, `cads554`, and `vsmtrans`.
- LabelCritic stub returned `uncertain`, so the E-step used explicit
  `label_critic_fallback` and wrote the fallback reason to
  `selection_metadata.json`, `training_manifest.json`, `vlm_decisions.jsonl`,
  and `review_queue.jsonl`.
- ShapeKit completed successfully for all three selected manifest items.
- `clavicula_left` was selected from `cads554` as a single-teacher candidate.
- `scapula_left` produced no candidate on this abdomen CT subset and was
  explicitly recorded in `review_queue.jsonl` as `missing_candidate`; it was
  not silently merged.

Manifest verification:

```json
{
  "manifest_items": 3,
  "manifest_organs": ["clavicula_left", "femur_left", "femur_right"],
  "femur_left": {
    "selected_model": "moose666",
    "candidate_models": ["moose666", "cads554", "vsmtrans"],
    "selection_method": "label_critic_fallback",
    "selection_status": "fallback",
    "shapekit_status": "success",
    "review_flags": ["selection_fallback"]
  },
  "femur_right": {
    "selected_model": "moose666",
    "candidate_models": ["moose666", "cads554", "vsmtrans"],
    "selection_method": "label_critic_fallback",
    "selection_status": "fallback",
    "shapekit_status": "success",
    "review_flags": ["selection_fallback"]
  },
  "clavicula_left": {
    "selected_model": "cads554",
    "candidate_models": ["cads554"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "success"
  }
}
```

Conclusion:

- MOOSE666 should keep `plans = nnUNetPlans` in
  `configs/model_registry.yaml`, matching the actual local checkpoint folder.
- The Drive script mismatch remains documented as an upstream/Drive internal
  inconsistency, not a local runnable blocker, because both MOOSE666 and
  MOOSE888 have now run with the local registry setting.

Validation commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_moose666_bones_cads554_vsmtrans_mock_shapekit \
  --num-cases 1 \
  --models moose666,cads554,vsmtrans,mock_seg \
  --organs femur_left,femur_right,scapula_left,clavicula_left \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900
```

## Continuation Audit - Teacher Meeting Plan Completeness Review

Status: completed for read-only planning review and documentation update.

Review result:

- The current Chinese execution plan preserves the exact target of 373 organs.
- It does not revert to 377/384 as the current engineering target.
- It explicitly keeps the 8 SAROS coarse/non-one-to-one organs and 3
  no-enabled-route organs skipped.
- It keeps teacher outputs labeled as pseudo-label candidates, not expert
  ground truth.
- It keeps LabelCritic before ShapeKit and before M-step training.
- It keeps VISTA3D as teacher/reference/legacy and not the 127-class student
  core.
- It keeps the student route 3D prompt-based/VoxTell-style and excludes 2D
  conversion.

Documentation changes made after review:

- Updated the current-data-access wording: the account exists, while JHU
  fine-label dataset access/data remains unavailable.
- Added the formal-case preference that the 50 CTs should prioritize cases with
  tumor annotation, while tumor masks remain context/side-channel information
  rather than 373 organ targets.
- Added ITK-SNAP sanity-check/display examples as an explicit review and
  presentation requirement.
- Added RadThinking-style patient trace / reasoning trace / VQA side-output as
  a non-blocking research asset.
- Added low-quality Dice threshold guidance:
  `student_vs_selected_pseudo_label Dice < 0.5` as strong review and
  `0.5-0.8` as warning/recheck, with empty masks, shape mismatch, and extreme
  volume ratios sent directly to review.
- Added the reporting requirement to track whether low-quality case-organ
  counts decrease across iterations, while keeping this framed as
  pseudo-label consistency rather than true accuracy.

## Continuation Audit - Regression After MOOSE666 Check

Status: passed.

Commands and observed results:

```bash
python scripts/audit_373_organ_routing.py
```

```json
{
  "status": "success",
  "target_organs": 373,
  "unique_target_organs": 373,
  "route_requested_organs": 373,
  "selected_model_keys": 22,
  "statically_unresolvable_target_organs": 0,
  "blocking_keys": []
}
```

```bash
python scripts/audit_21_model_drive_alignment.py
```

```json
{
  "status": "success",
  "target_model_count": 21,
  "registry_present_count": 21,
  "enabled_count": 21,
  "models_with_missing_required_files": {},
  "models_with_live_drive_missing_paths": {},
  "organs_with_best_enabled_model": 381,
  "policy_skipped_organ_count": 8,
  "static_exact_merge_candidate_organs": 373,
  "organs_without_best_enabled_model": 3,
  "meets_381_target": true
}
```

Known residual notes from the same Drive audit:

- DAPS and VISTA3D still appear under
  `models_with_required_files_unmatched_to_drive_manifest` because the cached
  size manifest cannot prove those paths, although live Drive missing paths are
  empty.
- MOOSE666 and MOOSE888 still show Drive command-parameter mismatches for
  `plans`, but real CT runs have now validated the local `nnUNetPlans`
  registry setting.

```bash
python scripts/audit_teacher_readiness.py \
  --models airrc,atm,cads551,cads552,cads553,cads554,cads555,cads556,cads557,cads558,cads559,daps,epai_20250421,lvp,moose666,moose888,nnunet_private,saros_nnunet,totalsegmentator,unest,vista3d,vsmtrans \
  --output outputs/audit_21_models/teacher_readiness_373_selected_keys_after_moose666.json
```

```json
{
  "status": "success",
  "num_models": 22,
  "models_ready_for_command": 22,
  "models_missing_checkpoint_path": []
}
```

```bash
python -m py_compile \
  scripts/nnunetv2_predict_and_split.py \
  agent-harness/cli_anything/medai/core/multimodel_loop.py \
  agent-harness/cli_anything/medai/core/shapekit_runner.py \
  agent-harness/cli_anything/medai/core/mstep_runner.py \
  agent-harness/cli_anything/medai/core/voxtell_student.py \
  scripts/run_real_teacher_subset_check.py \
  scripts/audit_373_organ_routing.py \
  scripts/audit_21_model_drive_alignment.py \
  scripts/audit_teacher_readiness.py
```

Compile result: passed.

Current next engineering priority:

1. Continue real subset validation for remaining teacher families not yet
   meaningfully tested on anatomy-matched CTs: especially ATM, AirRC, LVP,
   UNEST, TotalSegmentator, VISTA3D, and CADS557/CADS558 with brain/head-neck
   CTs when available.
2. Build a stronger small closed loop:
   2-5 tumor-annotation CTs, 10-20 representative organs, multiple real
   teachers, ShapeKit enabled, LabelCritic stub first, then real VLM on a tiny
   sample.
3. From that loop, produce the pseudo-label manifest, VoxTell small training,
   student re-inference, Round2 student-vs-Round1 competition, and failure
   mining report.
4. Add ITK-SNAP examples and patient trace / reasoning trace samples for the
   next teacher-facing progress report.

## Continuation Audit - LVP Real Teacher Check

Status: completed for LVP wrapper/checkpoint validation, direct label split,
single-teacher candidate selection, and ShapeKit unsupported-target metadata.

Scope:

- Case: `PanTS_00000026`.
- Real teacher: `lvp`.
- Synthetic comparator: `mock_seg`.
- Organs: `liver_hepatic_vein`, `liver_portal_vein`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Pre-run registry/label verification:

- `configs/model_registry.yaml` routes `lvp` to Dataset1381 with
  `nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres`.
- Local `dataset.json` labels directly match current 373 target names:
  - `liver_hepatic_vein`
  - `liver_portal_vein`
- No alias entry is required for LVP because both labels are direct global
  organ names.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["lvp", "mock_seg"],
  "organs": ["liver_hepatic_vein", "liver_portal_vein"],
  "estep": {
    "status": "success",
    "total_updated": 2,
    "total_labelcritic_decisions": 0
  },
  "manifest_items": 2,
  "selected_models": ["lvp"],
  "candidate_model_sets": [["lvp"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 0,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `lvp` real inference succeeded on a real CT.
- The wrapper wrote two non-empty masks:
  - `liver_hepatic_vein.nii.gz` with 64,861 voxels.
  - `liver_portal_vein.nii.gz` with 21,996 voxels.
- Both organs were selected as `single_teacher_default` from `lvp`.
- `mock_seg` did not provide those liver-vessel masks, as expected.
- ShapeKit had no safe target group for these liver-vessel subclasses and
  fell back to the selected teacher masks.
- The fallback was not silent: both manifest items and review queue entries
  include `shapekit_fallback` and `shapekit_unsupported_target`.

Manifest verification:

```json
[
  {
    "organ": "liver_hepatic_vein",
    "selected_model": "lvp",
    "candidate_models": ["lvp"],
    "selection_method": "single_teacher_default",
    "selection_status": "selected",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "quality_flags": ["single_candidate", "shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  },
  {
    "organ": "liver_portal_vein",
    "selected_model": "lvp",
    "candidate_models": ["lvp"],
    "selection_method": "single_teacher_default",
    "selection_status": "selected",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "quality_flags": ["single_candidate", "shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  }
]
```

Conclusion:

- LVP is now validated as runnable on the server for its two current 373-target
  liver-vessel organs.
- No wrapper, alias, or merge code change was required in this pass.
- This is not true accuracy evidence; no JHU expert fine labels were used.

Validation command:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_lvp_liver_vessels_mock_shapekit \
  --num-cases 1 \
  --models lvp,mock_seg \
  --organs liver_hepatic_vein,liver_portal_vein \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900
```

Regression after LVP check:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Known residual notes unchanged:

- DAPS and VISTA3D remain unmatched to the older cached Drive size manifest,
  but live Drive missing paths are empty.
- MOOSE666 and MOOSE888 still have a Drive-script `plans` mismatch; real runs
  validate the local `nnUNetPlans` setting.
- CADS557/CADS558 still require anatomy-matched brain/head-neck CTs for
  meaningful non-empty E-step validation.

## Continuation Audit - Private AbdomenAtlas nnUNet Real Teacher Check

Status: completed for `nnunet_private` wrapper/checkpoint validation,
multi-teacher candidate selection, alias lookup, ShapeKit success, and
source-aware manifest metadata.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `nnunet_private`, `vsmtrans`, `cads551`.
- Synthetic comparator: `mock_seg`.
- Organs: `aorta`, `gall_bladder`, `kidney_left`,
  `portal_vein_and_splenic_vein`, `pancreas_head`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Pre-run registry/label verification:

- `nnunet_private` maps to Dataset224 AbdomenAtlas1.1 with
  `nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres`.
- The local Dataset224 `dataset.json` contains 34 direct target labels,
  including all five organs used in this check.
- This model had no prior real subset summary in `outputs/audit_21_models`, so
  this pass adds its first real E-step evidence.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["nnunet_private", "vsmtrans", "cads551", "mock_seg"],
  "organs": [
    "aorta",
    "gall_bladder",
    "kidney_left",
    "portal_vein_and_splenic_vein",
    "pancreas_head"
  ],
  "estep": {
    "status": "success",
    "total_updated": 5,
    "total_labelcritic_decisions": 4
  },
  "manifest_items": 5,
  "selected_models": ["nnunet_private", "vsmtrans"],
  "candidate_model_sets": [
    ["nnunet_private"],
    ["nnunet_private", "vsmtrans", "cads551"],
    ["nnunet_private", "vsmtrans", "cads551", "mock_seg"]
  ],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 4,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- `nnunet_private` real inference succeeded on a real CT and produced 34 masks.
- The five requested organs all appeared in the final pseudo-label manifest.
- `aorta`, `gall_bladder`, `kidney_left`, and
  `portal_vein_and_splenic_vein` formed multi-teacher candidate sets and
  generated durable LabelCritic stub records in `vlm_decisions.jsonl`.
- `pancreas_head` was selected as a single-teacher `nnunet_private` candidate.
- ShapeKit completed successfully for all five selected masks.
- The CADS551 local label `gallbladder` was correctly resolved to global
  `gall_bladder` through `model_label_aliases.json`.
- `portal_vein_and_splenic_vein` was collected from all three real teachers in
  this run, confirming direct-name handling for this model family.

Manifest verification:

```json
{
  "manifest_items": 5,
  "organs": {
    "aorta": {
      "selected_model": "vsmtrans",
      "candidate_models": ["nnunet_private", "vsmtrans", "cads551", "mock_seg"],
      "selection_method": "label_critic_fallback",
      "shapekit_status": "success"
    },
    "gall_bladder": {
      "selected_model": "nnunet_private",
      "candidate_models": ["nnunet_private", "vsmtrans", "cads551"],
      "alias_match_observed": "local_alias:gallbladder",
      "selection_method": "label_critic_fallback",
      "shapekit_status": "success"
    },
    "kidney_left": {
      "selected_model": "nnunet_private",
      "candidate_models": ["nnunet_private", "vsmtrans", "cads551", "mock_seg"],
      "selection_method": "label_critic_fallback",
      "shapekit_status": "success"
    },
    "portal_vein_and_splenic_vein": {
      "selected_model": "nnunet_private",
      "candidate_models": ["nnunet_private", "vsmtrans", "cads551"],
      "selection_method": "label_critic_fallback",
      "shapekit_status": "success"
    },
    "pancreas_head": {
      "selected_model": "nnunet_private",
      "candidate_models": ["nnunet_private"],
      "selection_method": "single_teacher_default",
      "shapekit_status": "success"
    }
  }
}
```

Conclusion:

- `nnunet_private` is now validated as runnable on the server and integrated
  into the E-step candidate/merge/ShapeKit path for representative abdominal
  targets.
- No wrapper, alias, or merge code change was required in this pass.
- LabelCritic remains stub in this validation, so all multi-teacher selections
  are explicitly marked as `label_critic_fallback` and queued for review.
- This is not true accuracy evidence; no JHU expert fine labels were used.

Validation command:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_nnunet_private_abdomen_vsmtrans_cads551_mock_shapekit \
  --num-cases 1 \
  --models nnunet_private,vsmtrans,cads551,mock_seg \
  --organs aorta,gall_bladder,kidney_left,portal_vein_and_splenic_vein,pancreas_head \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900
```

Regression after `nnunet_private` check:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

## Continuation Audit - UNEST Real Teacher Check

Status: completed for UNEST MONAI-bundle wrapper validation, kidney-substructure
mask splitting, single-teacher candidate selection, and ShapeKit
unsupported-target metadata.

Scope:

- Case: `PanTS_00000026`.
- Real teacher: `unest`.
- Synthetic comparator: `mock_seg`.
- Organs: `kidney_cortex`, `kidney_medulla`,
  `kidney_pelvicalyceal_system`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Pre-run wrapper/label verification:

- `scripts/unest_predict_and_split.py` uses the local MONAI bundle at
  `checkpoints/UNEST/UNEST/renalStructures_UNEST_segmentation`.
- The local bundle includes `models/model.pt`, `configs/metadata.json`, and
  `configs/inference.json`.
- The wrapper splits the combined UNEST label map into:
  - `kidney_cortex`
  - `kidney_medulla`
  - `kidney_pelvicalyceal_system`
- All three labels are present in the current 373 target organ list.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["unest", "mock_seg"],
  "organs": [
    "kidney_cortex",
    "kidney_medulla",
    "kidney_pelvicalyceal_system"
  ],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 0
  },
  "manifest_items": 3,
  "selected_models": ["unest"],
  "candidate_model_sets": [["unest"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 0,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- UNEST real inference succeeded on a real CT through the MONAI bundle path.
- The wrapper produced three per-organ masks:
  - `kidney_cortex.nii.gz`
  - `kidney_medulla.nii.gz`
  - `kidney_pelvicalyceal_system.nii.gz`
- All three were selected as `single_teacher_default` from `unest`.
- ShapeKit does not have safe target groups for these kidney substructures, so
  each selected mask fell back to the teacher output.
- The fallback was explicit: manifest and review queue entries include
  `shapekit_fallback` and `shapekit_unsupported_target`.

Manifest verification:

```json
[
  {
    "organ": "kidney_cortex",
    "selected_model": "unest",
    "candidate_models": ["unest"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  },
  {
    "organ": "kidney_medulla",
    "selected_model": "unest",
    "candidate_models": ["unest"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  },
  {
    "organ": "kidney_pelvicalyceal_system",
    "selected_model": "unest",
    "candidate_models": ["unest"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  }
]
```

Conclusion:

- UNEST is now validated as runnable on the server and integrated into the
  E-step candidate/manifest path for its three current 373-target kidney
  substructures.
- No wrapper, alias, or merge code change was required in this pass.
- This is not true accuracy evidence; no JHU expert fine labels were used.

Validation command:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_unest_kidney_substructures_mock_shapekit \
  --num-cases 1 \
  --models unest,mock_seg \
  --organs kidney_cortex,kidney_medulla,kidney_pelvicalyceal_system \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900
```

Regression after UNEST check:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

## Continuation Audit - TotalSegmentator Official-CLI License-Safe Routing Fix

Status: completed for the TotalSegmentator E-step wrapper path, official CLI
subtask selection, license filtering, standardized output layout, and one real
non-licensed subtask validation.

Problem found:

- `registered_infer.run_registered_model()` previously invoked
  `run_totalseg_with_contract()` without passing organ-specific subtask context
  from the current E-step request.
- As a result, TotalSegmentator could default to all subtasks in
  `configs/totalseg_subtask_organs.json`, including academic-license tasks such
  as `appendicular_bones`, `brain_structures`, `coronary_arteries`, and
  `heartchambers_highres`.
- This violated the project requirement that TotalSegmentator be handled through
  the official TotalSegmentator route with license constraints respected.
- A second layout issue was found: the runner stored subtask outputs under
  `segmentations/totalseg_<task>/`, while the E-step candidate collector looks
  for direct files under `segmentations/*.nii.gz`. This could make successful
  TotalSegmentator outputs appear as missing candidates.

Code changes:

- `agent-harness/cli_anything/medai/core/registered_infer.py`
  - Added TotalSegmentator subtask resolution from `requested_organs`,
    explicit `subtasks`, or explicit `subtask`.
  - Added default skipping of academic-license subtasks unless
    `allow_licensed_totalseg` is explicitly set.
  - Records `selected_subtasks` and `skipped_subtasks` in inference results.
- `agent-harness/cli_anything/medai/core/totalseg_runner.py`
  - Added support for multiple explicit subtasks.
  - Preserves official subtask output folders.
  - Also copies produced masks to the standard
    `case/segmentations/*.nii.gz` layout consumed by the E-step.
  - Records `segmentation_output`, `num_masks`, `sample_masks`, and `subtasks`
    in the result.
- `agent-harness/cli_anything/medai/core/multimodel_loop.py`
  - Passes the current requested organ list into `run_registered_model()` so
    TotalSegmentator only runs relevant subtasks in E-step checks.
- `agent-harness/cli_anything/medai/medai_cli.py`
  - Groups repeated `totalsegmentator` routed subtasks into a single model run
    for `segment-all`, avoiding repeated overwrites.
  - Passes `requested_organs` and grouped `subtasks` to registered inference.
- `scripts/audit_teacher_readiness.py`
  - Records `dry_run_selected_subtasks` and `dry_run_skipped_subtasks`.

Dry-run validation:

```json
{
  "requested_organs": ["liver_segment_1", "liver_segment_2"],
  "selected_subtasks": ["liver_segments"],
  "skipped_subtasks": []
}
```

License-filter validation:

```json
{
  "requested_organs": ["coronary_artery"],
  "status": "skipped",
  "reason": "No TotalSegmentator subtasks remain after license/request filtering.",
  "skipped_subtasks": [
    {
      "subtask": "coronary_arteries",
      "reason": "academic license required; skipped by default"
    }
  ]
}
```

`segment-all` grouped-routing validation:

```json
{
  "requested_organs": ["liver_segment_1", "liver_segment_2", "coronary_artery"],
  "model_runs": [
    {
      "model_key": "totalsegmentator",
      "subtasks": ["liver_segments", "coronary_arteries"],
      "infer": {
        "status": "dry_run",
        "selected_subtasks": ["liver_segments"],
        "skipped_subtasks": [
          {
            "subtask": "coronary_arteries",
            "reason": "academic license required; skipped by default"
          }
        ]
      }
    }
  ]
}
```

## Continuation Audit - TotalSegmentator Real Teacher Check

Status: completed for official CLI execution, non-licensed subtask routing,
standard E-step candidate layout, and manifest/review metadata.

Scope:

- Case: `PanTS_00000026`.
- Real teacher: `totalsegmentator`.
- Synthetic comparator: `mock_seg`.
- Organs: `liver_segment_1`, `liver_segment_2`.
- TotalSegmentator subtask actually run: `liver_segments`.
- Academic-license subtasks: not run.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["totalsegmentator", "mock_seg"],
  "organs": ["liver_segment_1", "liver_segment_2"],
  "estep": {
    "status": "success",
    "total_updated": 2,
    "total_labelcritic_decisions": 0
  },
  "manifest_items": 2,
  "selected_models": ["totalsegmentator"],
  "candidate_model_sets": [["totalsegmentator"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 0,
  "all_items_have_source_metadata": true
}
```

Inference result verification:

```json
{
  "backend": "TotalSegmentator",
  "status": "success",
  "selected_subtasks": ["liver_segments"],
  "skipped_subtasks": [],
  "num_masks": 8,
  "sample_masks": [
    "liver_segment_1.nii.gz",
    "liver_segment_2.nii.gz",
    "liver_segment_3.nii.gz",
    "liver_segment_4.nii.gz",
    "liver_segment_5.nii.gz",
    "liver_segment_6.nii.gz",
    "liver_segment_7.nii.gz",
    "liver_segment_8.nii.gz"
  ],
  "subtask_results": {
    "liver_segments": {
      "status": "success",
      "return_code": 0,
      "num_masks": 8
    }
  }
}
```

Observed behavior:

- The official installed `TotalSegmentator` CLI was used.
- Only the request-relevant `liver_segments` subtask ran.
- The runner preserved official subtask outputs under
  `segmentations/totalseg_liver_segments/`.
- The runner also wrote standardized direct masks under
  `segmentations/*.nii.gz`, allowing the E-step to find
  `liver_segment_1` and `liver_segment_2`.
- Both requested organs were selected as `single_teacher_default` from
  `totalsegmentator`.
- ShapeKit does not have a safe target for these liver segment subclasses, so
  the masks fell back to teacher outputs and were explicitly marked
  `shapekit_unsupported_target`.

Manifest verification:

```json
[
  {
    "organ": "liver_segment_1",
    "selected_model": "totalsegmentator",
    "candidate_models": ["totalsegmentator"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  },
  {
    "organ": "liver_segment_2",
    "selected_model": "totalsegmentator",
    "candidate_models": ["totalsegmentator"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original",
    "shapekit_reason": "No safe ShapeKit target organs detected",
    "review_flags": ["shapekit_fallback", "shapekit_unsupported_target"],
    "ground_truth_status": "pseudo_label_candidate"
  }
]
```

Validation command:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_totalseg_liver_segments_mock_shapekit \
  --num-cases 1 \
  --models totalsegmentator,mock_seg \
  --organs liver_segment_1,liver_segment_2 \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 1200
```

Regression after TotalSegmentator fix:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Known residual notes unchanged:

- Licensed TotalSegmentator tasks are now explicitly skipped by default unless
  `allow_licensed_totalseg` is provided. To use them, an academic license must
  be configured according to official TotalSegmentator instructions.
- This is not true accuracy evidence; no JHU expert fine labels were used.

## Continuation Audit - VISTA3D Real Teacher Check

Status: completed for VISTA3D MONAI-bundle wrapper validation, label-map split,
runtime candidate participation, ShapeKit handoff, and source-aware manifest
metadata.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `vista3d`, `vsmtrans`.
- Synthetic comparator: `mock_seg`.
- Organs: `aorta`, `kidney_left`, `kidney_right`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Pre-run bundle/label verification:

- Local VISTA3D bundle path exists:
  `checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master`.
- Required local files exist:
  - `models/model.pt`
  - `configs/inference.json`
  - `configs/batch_inference.json`
  - `label_mappings/label_dict_127_abdomenAtlas3-1.json`
- The VISTA3D label map contains 127 entries.
- VISTA3D covers 74 organs in the current 373-target list.
- `scripts/vista3d_predict_and_split.py --dry-run` succeeds with the local
  bundle and label map.

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["vista3d", "vsmtrans", "mock_seg"],
  "organs": ["aorta", "kidney_left", "kidney_right"],
  "estep": {
    "status": "success",
    "total_updated": 3,
    "total_labelcritic_decisions": 3
  },
  "manifest_items": 3,
  "selected_models": ["vsmtrans"],
  "candidate_model_sets": [
    ["vista3d", "vsmtrans", "mock_seg"],
    ["vsmtrans", "mock_seg"]
  ],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 3,
  "all_items_have_source_metadata": true
}
```

Inference result verification:

```json
{
  "model_key": "vista3d",
  "status": "success",
  "num_masks": 71,
  "sample_masks": [
    "kidney_left.nii.gz",
    "kidney_right.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
    "gall_bladder.nii.gz",
    "liver.nii.gz",
    "pancreas.nii.gz",
    "spleen.nii.gz"
  ],
  "return_code": 0,
  "timed_out": false
}
```

Observed behavior:

- VISTA3D real inference succeeded through the local MONAI bundle and produced
  71 non-empty per-organ masks.
- `kidney_left` and `kidney_right` participated as VISTA3D candidates and were
  compared against `vsmtrans` by LabelCritic stub.
- `aorta` did not participate as a VISTA3D candidate in this specific case
  because the split output did not contain a non-empty `aorta.nii.gz`, even
  though `aorta` exists in the VISTA3D label map. This is recorded as
  `candidate_exists=False` rather than silently merged.
- All final selected masks came from `vsmtrans` in this validation because
  LabelCritic was stubbed and the reference-consistency fallback selected the
  highest available candidate.
- ShapeKit succeeded for all three selected masks.

Manifest verification:

```json
{
  "aorta": {
    "selected_model": "vsmtrans",
    "candidate_models": ["vsmtrans", "mock_seg"],
    "selection_method": "label_critic_fallback",
    "shapekit_status": "success"
  },
  "kidney_left": {
    "selected_model": "vsmtrans",
    "candidate_models": ["vista3d", "vsmtrans", "mock_seg"],
    "selection_method": "label_critic_fallback",
    "shapekit_status": "success"
  },
  "kidney_right": {
    "selected_model": "vsmtrans",
    "candidate_models": ["vista3d", "vsmtrans", "mock_seg"],
    "selection_method": "label_critic_fallback",
    "shapekit_status": "success"
  }
}
```

Dice/candidate observations:

```json
{
  "aorta": {
    "vista3d_candidate_exists": false,
    "reason": "Model prediction missing; cannot verify."
  },
  "kidney_left": {
    "vista3d_candidate_exists": true,
    "vista3d_reference_dice": 0.127394
  },
  "kidney_right": {
    "vista3d_candidate_exists": true,
    "vista3d_reference_dice": 0.585937
  }
}
```

Conclusion:

- VISTA3D is now validated as runnable on the server through the local bundle
  and integrated as a teacher/reference candidate source.
- The run confirms that VISTA3D remains a teacher/reference component and does
  not constrain the student target space to 127 classes.
- No wrapper or alias code change was required in this pass.
- Empty or missing VISTA3D outputs are handled by existing candidate-exists
  metadata and do not silently enter the merge.
- This is not true accuracy evidence; no JHU expert fine labels were used.

Validation command:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_vista3d_abdomen_vsmtrans_mock_shapekit \
  --num-cases 1 \
  --models vista3d,vsmtrans,mock_seg \
  --organs aorta,kidney_left,kidney_right \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 1200
```

Regression after VISTA3D check:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Known residual note:

- The Drive alignment audit still lists VISTA3D under
  `models_with_required_files_unmatched_to_drive_manifest` because the older
  cached size manifest cannot prove those files. However, live Drive missing
  paths are empty and this real run validates the local runnable bundle.

## Continuation Audit - ATM / AirRC Real Teacher Probe

Status: completed for registry command construction, real nnUNet wrapper
execution, per-label split, candidate collection, LabelCritic/fallback metadata,
ShapeKit fallback metadata, and 373-routing regression.

Scope:

- Case: `PanTS_00000026`.
- Real teachers: `atm`, `airrc`.
- Synthetic comparator: `mock_seg`.
- Organs: `airway_tree`, `airway_wall`, `lung_pulmonary_arteries`,
  `lung_pulmonary_veins`.
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Important limitation:

- This run used the available PanTS abdomen CT as a server-side runnable probe.
  ATM/AirRC are airway/chest-oriented teachers, so this validates wrapper,
  output split, routing, and merge metadata, not anatomical accuracy. A chest CT
  is still needed for meaningful airway/lung-vessel quality validation.

Registry facts:

```json
{
  "atm": {
    "dataset_id": 1370,
    "trainer": "nnUNetTrainer",
    "plans": "nnUNetResEncUNetLPlans",
    "folds": "all",
    "checkpoint_name": "checkpoint_final",
    "covered_organs": ["airway_tree"]
  },
  "airrc": {
    "dataset_id": 1380,
    "trainer": "nnUNetTrainer",
    "plans": "nnUNetResEncUNetLPlans",
    "folds": "all",
    "checkpoint_name": "checkpoint_final",
    "covered_organs": [
      "airway_tree",
      "airway_wall",
      "lung_pulmonary_arteries",
      "lung_pulmonary_veins"
    ]
  }
}
```

Evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "models": ["atm", "airrc", "mock_seg"],
  "organs": [
    "airway_tree",
    "airway_wall",
    "lung_pulmonary_arteries",
    "lung_pulmonary_veins"
  ],
  "estep": {
    "status": "success",
    "total_updated": 4,
    "total_labelcritic_decisions": 1
  },
  "manifest_items": 4,
  "selected_models": ["airrc", "atm"],
  "candidate_model_sets": [["airrc"], ["atm", "airrc"]],
  "shapekit_statuses": ["fallback_original"],
  "vlm_decision_lines": 1,
  "all_items_have_source_metadata": true,
  "accuracy_warning": "This check validates pipeline wiring, not true segmentation accuracy."
}
```

Observed behavior:

- `atm` completed real inference and produced `airway_tree.nii.gz`.
- `airrc` completed real inference and produced four target masks:
  `airway_tree.nii.gz`, `airway_wall.nii.gz`,
  `lung_pulmonary_arteries.nii.gz`, and `lung_pulmonary_veins.nii.gz`.
- `airway_tree` had two real candidates (`atm`, `airrc`) and entered the
  LabelCritic/fallback path.
- `airway_wall`, `lung_pulmonary_arteries`, and `lung_pulmonary_veins` were
  single-teacher AirRC defaults.
- ShapeKit did not refine these airway/lung-vessel targets and returned
  `fallback_original`; the manifest records `shapekit_fallback` and
  `shapekit_unsupported_target` review/quality flags instead of silently
  pretending success.

Manifest verification:

```json
{
  "airway_tree": {
    "selected_model": "atm",
    "candidate_models": ["atm", "airrc"],
    "selection_method": "label_critic_fallback",
    "selection_status": "fallback",
    "fallback_reason": "LabelCritic inconclusive for atm vs airrc",
    "shapekit_status": "fallback_original",
    "review_flags": [
      "selection_fallback",
      "shapekit_fallback",
      "shapekit_unsupported_target"
    ]
  },
  "airway_wall": {
    "selected_model": "airrc",
    "candidate_models": ["airrc"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original"
  },
  "lung_pulmonary_arteries": {
    "selected_model": "airrc",
    "candidate_models": ["airrc"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original"
  },
  "lung_pulmonary_veins": {
    "selected_model": "airrc",
    "candidate_models": ["airrc"],
    "selection_method": "single_teacher_default",
    "shapekit_status": "fallback_original"
  }
}
```

Validation command:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/real_teacher_subset_1case_atm_airrc_airway_abdomen_probe_shapekit \
  --num-cases 1 \
  --models atm,airrc,mock_seg \
  --organs airway_tree,airway_wall,lung_pulmonary_arteries,lung_pulmonary_veins \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 900
```

Regression after ATM/AirRC check:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Known residual notes:

- This is not true accuracy evidence; no JHU expert fine labels were used.
- A chest CT should be used later to validate ATM/AirRC anatomical quality and
  non-empty geometry in-domain.
- The current E-step writes per-case source metadata as
  `pseudo_label_selection.json`; some older audit text also mentions
  `selection_metadata.json`. The manifest fields are present, but naming should
  be standardized or documented to avoid operator confusion.

## Continuation Audit - Round2 Competition and Legacy Guard Repair

Status: completed for the main logic risk identified by code review:
Round2+ now explicitly injects the previous round's selected pseudo labels as
candidate masks, and VISTA3D legacy student routes are blocked unless explicitly
enabled for historical reproduction.

Problem fixed:

- Before this repair, Round2+ injected `student_prev` predictions but did not
  explicitly inject the previous round's selected + ShapeKit-final pseudo label.
  That was weaker than the teacher meeting requirement: student output must
  compete against the first-round best output, and student must not automatically
  replace it.
- The `vista3d_legacy` backend could still be activated by environment variable
  without an explicit legacy guard, and CLI help text still described VISTA3D as
  a student/M-step route.

Code changes:

- `scripts/run_em_training.py`
  - Added `ensure_current_student_backend_allowed()`.
  - `MEDAI_STUDENT_BACKEND=vista3d_legacy` now raises unless
    `MEDAI_ALLOW_VISTA3D_LEGACY=1` is set.
  - Round2+ now injects:
    - `round_prev_selected` from
      `outputs/round<N-1>/estep/annotation_versions`;
    - `student_prev` from
      `outputs/round<N-1>/student_predictions`.
  - The legacy VISTA3D dataset builder now prefers current E-step selected
    pseudo labels, not direct student predictions, so student predictions first
    have to pass candidate selection.
- `agent-harness/cli_anything/medai/core/multimodel_loop.py`
  - Added `_resolve_preseeded_case_dir()` to support preseeded layouts:
    `<base>/<case_id>/*.nii.gz`,
    `<base>/<case_id>/updated/*.nii.gz`, and
    `<base>/<case_id>/segmentations/*.nii.gz`.
  - This makes `round_prev_selected` compatible with the existing
    `annotation_versions/<case>/updated` layout.
- `agent-harness/cli_anything/medai/medai_cli.py`
  - Updated `vista3d-segment` and `vista3d-finetune` descriptions to
    legacy/reference wording.
  - Added JSON `legacy_warning` fields to both commands.

Targeted smoke evidence:

```json
{
  "preseeded_layout_resolution": {
    "status": "success",
    "checked_layouts": ["direct", "updated", "segmentations"]
  },
  "vista3d_legacy_guard_default": {
    "status": "guarded",
    "message_prefix": "MEDAI_STUDENT_BACKEND=vista3d_legacy is disabled by default"
  },
  "vista3d_legacy_guard_explicit": {
    "status": "allowed_with_explicit_legacy_flag"
  }
}
```

Regression after repair:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Residual notes:

- This repair proves candidate injection mechanics and guard behavior, but a
  real Round2 run with trained student masks is still required before claiming
  the full Round2 teacher-vs-student competition has been validated at scale.
- The Drive alignment audit still carries the known cached-manifest caveats for
  DAPS/VISTA3D and the MOOSE `run_MOOSE.sh` plans-name mismatch; live Drive
  missing paths remain empty and real runnable checks have validated the local
  registry choices.

## Continuation Audit - CADS / TotalSegmentator Provenance Clarification

Status: completed for registry provenance metadata and audit visibility.

Problem addressed:

- CADS datasets `Dataset551_Totalseg251` through `Dataset555_Totalseg255`
  contain the string `Totalseg` in their Drive dataset names.
- The project also has a separate official `totalsegmentator` registry entry.
- Without an explicit provenance note, this could be misread as mixing official
  TotalSegmentator licensing/CLI requirements into CADS private nnUNet weights.

Clarification implemented:

- `configs/model_registry.yaml`
  - `cads551`-`cads559` now include:
    - `distribution_route: private_drive_nnunet_checkpoint`
    - `provenance_note`
    - `license_note`
  - These entries explicitly state that CADS is run with the Drive
    `run_CADS.sh` / `nnUNetv2_predict` parameters.
  - The `Totalseg` string in CADS dataset names is treated as a CADS/private
    label-set naming convention, not as the public official TotalSegmentator
    package.
- `scripts/audit_21_model_drive_alignment.py`
  - The JSON audit now carries registry provenance/license fields.
  - The Markdown report now includes a CADS private checkpoint /
    TotalSegmentator naming clarification section.

Current policy:

- CADS551-559: private Google Drive nnUNet checkpoints, run with Drive
  `run_CADS.sh` parameters.
- `totalsegmentator`: official installed TotalSegmentator CLI only, with
  requested-organ subtask filtering and academic-license subtasks skipped by
  default unless explicitly allowed.

Regression after clarification:

```json
{
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Residual note:

- If the data provider later states that CADS `Totalseg` checkpoints must also
  obey official TotalSegmentator licensing constraints, the CADS route should be
  disabled or moved to the official TotalSegmentator CLI path. Current evidence
  from Drive (`run_CADS.sh`) supports the private nnUNet route.

## Continuation Audit - ShapeKit Formal Guard

Status: completed for the public `run-loop` CLI.

Problem addressed:

- Teacher meeting requirement says formal selected outputs should pass through
  ShapeKit.
- The underlying metadata already recorded `skipped_debug_only`, but the public
  CLI still allowed `--no-enable-shapekit` in a non-dry-run command without an
  explicit debug acknowledgement.

Code change:

- `agent-harness/cli_anything/medai/medai_cli.py`
  - Added `--debug-allow-no-shapekit`.
  - `run-loop --no-enable-shapekit` now fails in non-dry-run formal mode unless
    the user explicitly passes `--debug-allow-no-shapekit`.
  - Dry-run still permits disabling ShapeKit for smoke checks.

Targeted behavior checks:

```json
{
  "formal_no_shapekit": {
    "exit_code": 1,
    "status": "failed",
    "reason": "Formal non-dry-run E-step requires ShapeKit."
  },
  "dry_run_no_shapekit": {
    "exit_code": 0,
    "allowed": true
  },
  "debug_no_shapekit": {
    "exit_code": 0,
    "allowed": true
  }
}
```

Residual note:

- `scripts/run_real_teacher_subset_check.py` and
  `scripts/run_teacher_plan_tiny_check.py` still allow `--no-enable-shapekit`
  because they are explicit audit/smoke helpers, not formal production entry
  points.

## Continuation Audit - Target Policy Metadata and Small Multi-Teacher Loop

Status: completed for per-run target-space policy metadata and a stronger
2-case, multi-real-teacher small loop.

Problem addressed:

- The accepted target config already contained the 384/381/8/3/373 accounting,
  but each E-step `run_summary.json` did not explicitly carry that policy.
- This made subset runs harder to explain: a run might request 10 organs, but
  the output did not directly state that the project-wide accepted exact target
  remains 373 and that 8 policy-skipped plus 3 no-route organs are intentionally
  excluded.

Code change:

- `agent-harness/cli_anything/medai/core/multimodel_loop.py`
  - Added `_load_target_space_policy()`.
  - Every `run_summary.json` now includes `target_space_policy` with:
    - accepted 373 target count;
    - original global/routing counts;
    - requested organs count;
    - whether the request is the full accepted target;
    - `policy_skipped_organs` (8 SAROS coarse/unresolvable organs);
    - `no_enabled_route_organs` (3 no-route organs);
    - pseudo-label / non-true-accuracy warning.

Target policy smoke evidence:

```json
{
  "status": "loaded",
  "accepted_current_exact_prompt_targets": 373,
  "requested_organs": 2,
  "requested_is_full_accepted_target": false,
  "ground_truth_status": "pseudo_label_candidate",
  "counts": {
    "global_label_space_organs": 384,
    "enabled_routed_organs": 381,
    "policy_skipped_organs": 8,
    "no_enabled_route_organs": 3,
    "current_exact_prompt_target_organs": 373,
    "accepted_current_exact_prompt_targets": 373,
    "historical_teacher_direction": 377
  },
  "policy_skipped_names": [
    "arm_left",
    "arm_right",
    "head",
    "intermuscular_adipose_tissue",
    "leg_left",
    "leg_right",
    "skeletal_muscle",
    "visceral_adipose_tissue"
  ],
  "no_route_names": ["arms", "legs", "muscle_fat"]
}
```

Small loop scope:

- Cases: `PanTS_00000026`, `PanTS_00000029`.
- Real teachers:
  - `epai_20250421`
  - `vsmtrans`
  - `cads551`
  - `cads552`
  - `cads553`
  - `cads554`
  - `nnunet_private`
- Synthetic comparator: `mock_seg`.
- Organs:
  - `liver`
  - `pancreas`
  - `spleen`
  - `kidney_left`
  - `kidney_right`
  - `aorta`
  - `gall_bladder`
  - `vertebrae_L1`
  - `femur_left`
  - `esophagus`
- ShapeKit: enabled.
- LabelCritic backend: `stub`.

Small loop evidence:

```json
{
  "stage": "real_teacher_subset_check",
  "status": "success",
  "num_cases": 2,
  "num_models": 8,
  "num_requested_organs": 10,
  "estep": {
    "status": "success",
    "total_updated": 19,
    "total_labelcritic_decisions": 17
  },
  "manifest_items": 19,
  "selected_models": [
    "cads551",
    "cads552",
    "epai_20250421",
    "nnunet_private",
    "vsmtrans"
  ],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 17,
  "all_items_have_source_metadata": true
}
```

Observed behavior:

- All selected manifest items include source metadata.
- 17 multi-candidate selections reached the LabelCritic/fallback record path.
- `vertebrae_L1` was correctly treated as a single-teacher CADS552 default.
- ShapeKit succeeded for all 19 final manifest items.
- One expected organ/case pair did not produce a final mask:
  `PanTS_00000029/femur_left`. This is visible as 19 items for 20 possible
  case-organ pairs and is not silently fabricated.
- `review_queue.jsonl` contains 19 items, mostly selection fallback/review
  records, and one low-reference-Dice risk in the first case.

Student contract checks on the small loop:

```json
{
  "voxtell_manifest": {
    "status": "success",
    "num_items": 19,
    "num_cases": 2,
    "num_items_missing_image": 0,
    "ground_truth_status": "pseudo_label_candidate"
  },
  "voxtell_train_dryrun": {
    "status": "dry_run",
    "num_manifest_items": 8,
    "model_files_ok": true
  },
  "student_infer_dryrun": {
    "status": "completed",
    "cases": 2,
    "prompts": 2
  },
  "failure_mining": {
    "status": "success",
    "num_comparisons": 4,
    "num_review_items": 4
  }
}
```

Commands:

```bash
python scripts/run_real_teacher_subset_check.py \
  --output-dir outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit \
  --num-cases 2 \
  --models epai_20250421,vsmtrans,cads551,cads552,cads553,cads554,nnunet_private,mock_seg \
  --organs liver,pancreas,spleen,kidney_left,kidney_right,aorta,gall_bladder,vertebrae_L1,femur_left,esophagus \
  --critic-backend stub \
  --enable-shapekit \
  --timeout-sec 1200
```

Regression after small loop:

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "route_requested_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed"
}
```

Residual notes:

- This is still not true accuracy evidence; no JHU expert fine labels were used.
- LabelCritic backend was `stub`; the chain records durable decisions and
  fallback semantics, but large-scale real VLM selection remains to be run.
- VoxTell training was dry-run contract validation, not a real student training
  run. Real training still requires/uses the configured VoxTell fine-tuning
  command and should be scaled after the small-loop contract is stable.

## 2026-06-05 Follow-up: Formal Mainline Guards And Real VoxTell Ministep

Status: additional code review and regression completed.

### Formal teacher pool alignment

`scripts/run_em_training.py` no longer uses legacy family aliases in the formal
E-step teacher list. The formal pool is now the concrete Drive-aligned model
keys plus the official TotalSegmentator route:

```text
cads551,cads552,cads553,cads554,cads555,cads556,cads557,cads558,cads559,
moose666,moose888,nnunet_private,saros_nnunet,atm,airrc,lvp,daps,
epai_20250421,vsmtrans,vista3d,unest,totalsegmentator
```

This fixes the previous risk that formal EM would call non-registry aliases such
as `cads`, `moose`, `moose3_0`, or `vsnet` instead of the required one-model-one
entry registry keys. A startup guard now fails fast if any formal teacher key is
missing from `configs/model_registry.yaml` or disabled.

### Formal ShapeKit and LabelCritic gates

`scripts/run_em_training.py` now mirrors the stricter CLI behavior for formal
runs:

- ShapeKit cannot be disabled unless `MEDAI_DEBUG_ALLOW_NO_SHAPEKIT=1` or
  `MEDAI_FAST_SMOKE=1` is explicitly set.
- LabelCritic cannot be disabled unless `MEDAI_DEBUG_ALLOW_NO_LABELCRITIC=1` or
  `MEDAI_FAST_SMOKE=1` is explicitly set.
- If LabelCritic is enabled but the vLLM server is offline, formal E-step now
  fails instead of silently falling back to non-critic selection. Debug/smoke
  fallback remains available only with the explicit debug flag.

Guard smoke evidence:

```text
MEDAI_ENABLE_SHAPEKIT=0 -> shape_guard_ok
MEDAI_ENABLE_CRITIC=0  -> critic_guard_ok
```

### Round2 competition audit

`multimodel_loop.py` now writes `round2_competition_audit` into
`run_summary.json` when `preseeded_model_dirs` are provided. This does not make
student outputs automatic winners. It records whether `round_prev_selected` and
`student_prev` actually appeared in candidate lists, how often each source was
selected, and example missing case-organ entries.

This addresses the teacher-meeting requirement that Round2 compare student
predictions against Round1 best pseudo labels, while making missing injected
candidate masks visible instead of silent.

### Real VoxTell ministep and real student inference evidence

The small-loop pseudo-label manifest was used for a real VoxTell-style 3D prompt
student ministep:

```bash
python scripts/train_voxtell_prompt_student.py \
  --manifest outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/voxtell_prompt_manifest.json \
  --model-dir checkpoints/VoxTell/voxtell_v1.1 \
  --text-encoding-model checkpoints/Qwen/Qwen3-Embedding-4B \
  --output-dir outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/voxtell_real_train_ministep \
  --max-items 4 --max-steps 2 --epochs 1 --freeze-encoder
```

Observed result:

```json
{
  "status": "success",
  "device": "cuda",
  "num_items": 4,
  "steps": 2,
  "generated": [
    "model_finetune.pth",
    "voxtell_finetuned_model/fold_0/checkpoint_final.pth"
  ]
}
```

Real student inference then produced a non-empty prompt mask:

```bash
python scripts/run_student_infer_then_round2.py \
  --round 1 \
  --case-list outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/case_list_subset.csv \
  --output-root outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/student_real_infer_ministep \
  --model-dir outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/voxtell_real_train_ministep/voxtell_finetuned_model \
  --max-cases 1 --prompts liver --timeout-sec 900
```

Observed output:

```text
student_predictions/round1/student_predictions/PanTS_00000026/liver.nii.gz
shape=(512, 362, 81), nonzero_voxels=341647
```

Failure mining against selected pseudo labels also ran on the real student
output:

```json
{
  "status": "success",
  "num_comparisons": 2,
  "review_items": [
    {
      "case_id": "PanTS_00000026",
      "organ": "liver",
      "student_vs_pseudo_dice": 0.669608,
      "volume_ratio": 0.504444,
      "metric_scope": "student_vs_selected_pseudo_label_consistency"
    },
    {
      "case_id": "PanTS_00000029",
      "organ": "liver",
      "reason": "missing student prediction because ministep inference used --max-cases 1"
    }
  ]
}
```

This remains pseudo-label consistency evidence, not true expert-label accuracy.
No JHU expert fine labels were used in this check.

### Regression after follow-up fixes

```json
{
  "routing_373": {
    "status": "success",
    "target_organs": 373,
    "statically_unresolvable_target_organs": 0
  },
  "drive_alignment": {
    "status": "success",
    "target_model_count": 21,
    "registry_present_count": 21,
    "enabled_count": 21,
    "models_with_missing_required_files": {},
    "models_with_live_drive_missing_paths": {},
    "static_exact_merge_candidate_organs": 373
  },
  "teacher_readiness": {
    "status": "success",
    "num_models": 22,
    "models_ready_for_command": 22,
    "models_missing_checkpoint_path": []
  },
  "py_compile": "passed for run_em_training.py and multimodel_loop.py"
}

Known residuals:

- The MOOSE command alignment audit still records the known `plans` mismatch
  between the older parsed `run_MOOSE.sh` expectation and the local runnable
  `nnUNetPlans` setup.
- Full formal `50 CT x 373 organs` generation, large-scale real LabelCritic,
  longer VoxTell training, full Round2 competition, ITK-SNAP samples, and JHU
  expert-label accuracy evaluation remain open.

## 2026-06-05 Follow-up: Real LabelCritic And Round2 Competition Script

Status: completed for a small real service check and reusable Round2 wiring
script.

### Real LabelCritic service restored

The Qwen2-VL vLLM server was restarted on the local A100 and verified through
the OpenAI-compatible endpoint:

```text
GET http://localhost:8000/v1/models -> 200
served model: checkpoints/Qwen/Qwen2-VL-7B-Instruct
```

The command used was:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model checkpoints/Qwen/Qwen2-VL-7B-Instruct \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.4 \
  --trust-remote-code
```

### Real LabelCritic pair check

A real A/B check compared the previous selected pseudo-label liver mask with
the VoxTell ministep student liver mask:

```bash
python scripts/run_labelcritic_pair_check.py \
  --backend labelcritic \
  --ct data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz \
  --mask-a outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/estep/annotation_versions/PanTS_00000026/updated/liver.nii.gz \
  --mask-b outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit/student_real_infer_ministep/round1/student_predictions/PanTS_00000026/liver.nii.gz \
  --organ liver \
  --output-json outputs/audit_21_models/labelcritic_real_student_vs_round1_liver/liver_round1_vs_student.json \
  --timeout-sec 300
```

Observed result:

```json
{
  "status": "success",
  "backend": "labelcritic",
  "service_online": true,
  "decision": {
    "winner": "uncertain",
    "confidence": 0.0,
    "parse_status": "vlm_undecided"
  }
}
```

This proves the real LabelCritic service path can run and produce durable
decision metadata. The uncertain decision is treated as a review/fallback signal,
not as permission for the student to overwrite Round1.

### Reusable Round2 competition check

Added `scripts/run_round2_competition_check.py`. It validates that
`round_prev_selected` and `student_prev` are injected as candidates, that
student output is not automatically selected, and that `round2_competition_audit`
is written to `run_summary.json`.

Real LabelCritic run:

```bash
python scripts/run_round2_competition_check.py \
  --output-dir outputs/audit_21_models/round2_competition_script_real_labelcritic_liver \
  --critic-backend labelcritic \
  --timeout-sec 600
```

Observed result:

```json
{
  "status": "success",
  "estep_status": "success",
  "total_updated": 2,
  "total_labelcritic_decisions": 2,
  "critic_backend": "labelcritic",
  "enable_shapekit": true,
  "manifest_items": 2,
  "manifest_items_with_both_preseeded_sources": 1,
  "round_prev_candidate_entries": 2,
  "student_candidate_entries": 1,
  "round_prev_selected_entries": 2,
  "student_selected_entries": 0,
  "vlm_decision_lines": 2,
  "review_queue_lines": 2
}
```

The first liver item had candidate models:

```text
round_prev_selected, student_prev, mock_seg
```

and selected `round_prev_selected` with `selection_method=label_critic_fallback`
because the real VLM decision was uncertain. The second case records
`student_prev` as missing because the ministep student inference intentionally
used `--max-cases 1`; this is visible in `round2_competition_audit`.

This is still a small wiring/quality-control check, not large-scale accuracy.
No JHU expert fine labels were used.

## 2026-06-05 Follow-up: Multi-Teacher Replay With Real LabelCritic

Status: completed for a 2-case, 4-organ, 5-teacher replay using existing raw
teacher predictions.

### Preseeded-only replay bug fixed

While preparing replay, a bug was found in `run_multimodel_annotation_loop`:
passing `models=[]` still triggered the default non-dry-run model
`totalsegmentator`. That is unsafe for a replay-only check because it can
unexpectedly launch heavy inference.

Fix:

```text
models is None -> use default model
models == []   -> run no new teacher models; use preseeded candidates only
```

Guard smoke evidence:

```json
{
  "status": "success",
  "models_requested": [],
  "total_updated": 2
}
```

No TotalSegmentator process was launched after the fix.

### Reusable replay script

Added `scripts/replay_teacher_candidates_with_labelcritic.py`.

Purpose:

```text
existing raw_predictions/<model>/<case>/segmentations
  -> preseeded teacher layout
  -> shared candidate collection
  -> real LabelCritic
  -> ShapeKit
  -> auditable manifest
```

This avoids rerunning expensive teacher models while testing the same selection,
metadata, LabelCritic, ShapeKit, and manifest path used by the formal E-step.

### Real multi-teacher replay check

Command:

```bash
python scripts/replay_teacher_candidates_with_labelcritic.py \
  --output-dir outputs/audit_21_models/replay_teacher_candidates_real_labelcritic_2case_4organ \
  --organs liver,pancreas,gall_bladder,esophagus \
  --models epai_20250421,vsmtrans,cads551,cads553,nnunet_private \
  --critic-backend labelcritic \
  --timeout-sec 600
```

Observed result:

```json
{
  "status": "success",
  "estep_status": "success",
  "total_updated": 8,
  "total_labelcritic_decisions": 8,
  "manifest_items": 8,
  "multi_candidate_items": 8,
  "selected_models": [
    "cads551",
    "epai_20250421",
    "nnunet_private",
    "vsmtrans"
  ],
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 8,
  "review_queue_lines": 8
}
```

Metadata audit:

```text
items=8
metadata_failures=[]
```

Each manifest item includes:

- `selected_model`
- `candidate_models`
- `selection_method`
- `selection_status`
- `labelcritic_records`
- `shapekit_status=success`
- `dataset_role=pseudo_label`
- `ground_truth_status=pseudo_label_candidate`
- `source_metadata_available=true`

All 8 real LabelCritic decisions were `uncertain` in this run. The pipeline
therefore used explicit `label_critic_fallback`, selected a fallback candidate,
and wrote review records with `selection_fallback`. This is the intended safe
behavior: uncertain VLM decisions do not silently accept a mask and do not claim
expert accuracy.

Residual after this check:

- Real LabelCritic is now service-verified and multi-item replay-verified, but
  large-scale stability on many organs/cases is still open.
- Full formal `50 CT x 373 organs`, longer VoxTell training, full Round2
  competition, ITK-SNAP samples, and JHU expert-label accuracy evaluation remain
  open.

## 2026-06-05 Follow-up: 10-Organ Real LabelCritic Replay And Left/Right Fix

Status: completed for the existing 2-case, 10-organ small-loop teacher subset.

### Expanded replay

The replay check was expanded from 4 organs to the full existing 10-organ
small-loop subset:

```bash
python scripts/replay_teacher_candidates_with_labelcritic.py \
  --output-dir outputs/audit_21_models/replay_teacher_candidates_real_labelcritic_2case_10organ_fixed \
  --organs liver,pancreas,spleen,kidney_left,kidney_right,aorta,gall_bladder,vertebrae_L1,femur_left,esophagus \
  --models epai_20250421,vsmtrans,cads551,cads552,cads553,cads554,nnunet_private \
  --critic-backend labelcritic \
  --timeout-sec 600
```

Observed result:

```json
{
  "status": "success",
  "estep_status": "success",
  "total_updated": 19,
  "total_labelcritic_decisions": 17,
  "manifest_items": 19,
  "multi_candidate_items": 17,
  "shapekit_statuses": ["success"],
  "vlm_decision_lines": 17,
  "review_queue_lines": 19
}
```

The only missing case-organ pair was expected:

```text
PanTS_00000029 / femur_left
```

because no preseeded teacher candidate existed for that pair in the existing
small-loop raw predictions.

### LabelCritic left/right projection bug fixed

The first 10-organ replay exposed a LabelCritic projection failure for left-side
organs such as `kidney_left` and `femur_left`. The upstream
`ProjectDatasetFlex_single.py` attempted to join any `*left*` projection with a
right-side projection directory, but a single-organ A/B comparison only projects
the requested organ. This caused `missing_log` / projection failures.

Fixes:

- `labelcritic_wrapper.py` prepares a projection-only right-side companion mask
  for left-side organs inside the temporary LabelCritic work folder.
- `third_party/LabelCritic-main/ProjectDatasetFlex_single.py` now skips the
  left/right join when the companion projection directory is absent instead of
  failing the whole comparison.

Single-pair verification:

```bash
python scripts/run_labelcritic_pair_check.py \
  --backend labelcritic \
  --ct data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz \
  --mask-a outputs/audit_21_models/replay_teacher_candidates_real_labelcritic_2case_10organ/preseeded_teachers/epai_20250421/PanTS_00000026/segmentations/kidney_left.nii.gz \
  --mask-b outputs/audit_21_models/replay_teacher_candidates_real_labelcritic_2case_10organ/preseeded_teachers/vsmtrans/PanTS_00000026/segmentations/kidney_left.nii.gz \
  --organ kidney_left \
  --output-json outputs/audit_21_models/labelcritic_left_right_companion_check_fixed/kidney_left_epai_vs_vsmtrans.json \
  --timeout-sec 300
```

Observed result:

```json
{
  "status": "success",
  "decision": {
    "winner": "uncertain",
    "parse_status": "vlm_undecided"
  }
}
```

### Fixed replay metadata audit

After the left/right fix, the full 10-organ replay was rerun and checked:

```json
{
  "manifest_items": 19,
  "missing_expected": [["PanTS_00000029", "femur_left"]],
  "metadata_failures": [],
  "selection_method_counts": {
    "label_critic_fallback": 17,
    "single_teacher_default": 2
  },
  "candidate_count_counts": {
    "4": 9,
    "3": 8,
    "1": 2
  },
  "vlm_status_counts": {
    "success": 17
  },
  "vlm_parse_counts": {
    "vlm_undecided": 17
  }
}
```

All manifest items include complete source metadata, `dataset_role=pseudo_label`,
`ground_truth_status=pseudo_label_candidate`, and `shapekit_status=success`.

Important interpretation:

- The real LabelCritic service and projection path are now stable for this
  10-organ small subset.
- All 17 VLM decisions were still `uncertain`, so the E-step correctly used
  explicit `label_critic_fallback` and wrote review queue records.
- These are pseudo-label consistency and pipeline-wiring checks, not true
  expert-label accuracy. No JHU fine labels were used.
