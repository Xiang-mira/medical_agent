# Teacher Meeting Implementation Plan - 3D Prompt Student Distillation

Date: 2026-06-05

This document is the project execution plan derived from the teacher meeting. It
keeps all requirements explicit so the codebase can be repaired step by step
without drifting back to the old 2D/VISTA3D-centered route.

## 0. Current Ground Rules

- Current exact target space is 373 organs.
- The historical/ideal teacher-covered direction was 377 organs, but the current
  executable target is 373 because unresolved or non-one-to-one classes are
  skipped instead of approximated.
- Teacher outputs are pseudo-label candidates, not true ground truth.
- Until the JHU fine-label dataset is available, evaluation can only claim
  pseudo-label consistency, not real segmentation accuracy.
- The student route must stay 3D prompt-based. Do not convert 3D CTs into a 2D
  slice segmentation pipeline.
- VISTA3D must not define the system label space. It is only one teacher or
  reference component in the teacher/model pool.
- VoxTell-style 3D prompt segmentation is the main student direction because it
  keeps 3D CT input and uses text prompts inside the 3D model.

## 1. Target Pipeline

The teacher's intended workflow should be implemented as this sequence:

```text
Inputs:
  many CT cases
  teacher model pool: current 13 active teachers, project 21-model inventory,
  and future expandable N teachers
  target organ space: current 373 exact organs

Step 1: Teacher inference
  Run each fixed teacher model on each CT.
  Teachers are not trained in this stage.
  Output candidate masks for all supported organs.

Step 2: Candidate collection
  For each case and each organ:
    collect all existing teacher/student candidate masks.
    if one candidate exists:
      keep it as the current pseudo-label candidate.
    if multiple candidates exist:
      send them to LabelCritic for early selection.
    if no exact candidate exists:
      skip the organ and record it for review/coverage reporting.

Step 3: LabelCritic selection
  LabelCritic is used immediately after candidate generation, not late in the
  M-step.
  It decides which candidate mask is most plausible when multiple models can
  output the same organ.
  Dice/reference signals may be metadata or fallback signals, but they should
  not replace LabelCritic for multi-candidate selection.

Step 4: ShapeKit post-processing
  All selected pseudo-label masks must pass through ShapeKit.
  ShapeKit is a mandatory post-selection cleanup stage in the formal pipeline.
  If ShapeKit fails for a mask, the system may fall back to the selected
  pre-ShapeKit mask, but this must be logged as a review item.

Step 5: Pseudo-label dataset assembly
  Build a pseudo-labeled dataset from selected and ShapeKit-processed masks.
  Each item must record CT path, organ, prompt, final mask path, selected source
  model, candidate models, LabelCritic decision metadata, ShapeKit status, and
  quality flags.

Step 6: 3D prompt student training
  Train a VoxTell-style 3D prompt-based student on the pseudo-label dataset.
  The student learns from the selected/processed teacher ensemble output, not
  from VISTA3D label IDs.

Step 7: Student re-inference
  Run the trained student on CT cases and save one binary mask per prompted
  organ.

Step 8: Round-2 candidate competition
  Compare student predictions against the first-round best pseudo-labels.
  Do not automatically trust the student. If it learns poorly, keep the first
  round output.
  If student and first-round output conflict, use LabelCritic again.

Step 9: Student failure mining
  Compute student-vs-round1-best Dice per case and organ.
  Identify organs/cases the student repeatedly cannot learn.

Step 10: Manual review/correction
  Send hard cases, LabelCritic-inconclusive cases, ShapeKit fallback cases, and
  persistent low-Dice cases to manual review.
  Manual work should target hard examples, not all 373 organs from scratch.
```

## 2. Module-Level Implementation Plan

### 2.1 Teacher Inference Layer

Purpose:

- Use all fixed teacher models as inference engines.
- Never describe this stage as teacher training.
- Keep the teacher model pool expandable from the current active set to the full
  21-model inventory and future additional teachers.

Required behavior:

- Each model has a single CLI or wrapper entry.
- Each wrapper writes masks into a consistent case/segmentations layout.
- Model output labels are normalized through model-label alias rules.
- TotalSegmentator-family models must continue to use the official
  TotalSegmentator route because of license differences.
- VISTA3D remains a teacher/reference wrapper only, not the student label-space
  authority.

Current project status:

- `scripts/run_em_training.py` defines the current active teacher pool.
- `agent-harness/cli_anything/medai/core/registered_infer.py` and
  `configs/model_registry.yaml` are the main inference registry path.
- The 21-model Drive alignment work remains part of the model inventory
  requirement, but the teacher meeting pipeline focuses on the active teacher
  pool used for pseudo-label construction.

Acceptance criteria:

- For a smoke case, every enabled teacher either writes expected masks or records
  a structured failure.
- No teacher wrapper trains or modifies teacher checkpoints during E-step.
- Output paths are deterministic and mergeable.

### 2.2 Organ Routing and Target Space

Purpose:

- Make the system know exactly which organs are currently targetable and which
  models can produce each organ.

Required behavior:

- Current student target count is 373 exact organs.
- Skipped classes are not approximated unless explicitly approved.
- The code must distinguish:
  - global organs in the xlsx/label space;
  - enabled routed organs;
  - policy-skipped SAROS coarse/non-one-to-one organs;
  - no-enabled-route organs;
  - current exact prompt student targets.

Current project status:

- `configs/student_3d_prompt_target_organs.json` currently records:
  - `global_label_space_organs = 384`
  - `enabled_routed_organs = 381`
  - `policy_skipped_organs = 8`
  - `no_enabled_route_organs = 3`
  - `current_exact_prompt_target_organs = 373`
  - `historical_teacher_direction = 377`

Acceptance criteria:

- All E-step and student-training code uses the 373 current exact target list by
  default.
- No active student path falls back to VISTA3D's 127 labels.
- Skipped organs appear in reports as skipped, not silently missing.

### 2.3 LabelCritic Placement

Purpose:

- Answer the teacher's question: when multiple models can segment the same
  organ, how do we choose the best output?

Required behavior:

- LabelCritic runs immediately after candidate masks are collected.
- Single-candidate organs are accepted as pseudo-label candidates but still
  marked as pseudo labels.
- Multi-candidate organs are selected by LabelCritic.
- If LabelCritic is disabled, fails, or is inconclusive, fallback selection must
  be explicit and reviewable.

Current project status:

- `agent-harness/cli_anything/medai/core/multimodel_loop.py` has been modified
  toward this behavior with `_select_candidate`.
- Additional validation is still needed to confirm all LabelCritic decisions are
  written to durable JSONL/manifest outputs.

Acceptance criteria:

- For a multi-candidate fake/smoke organ, the run writes:
  - candidate model list;
  - LabelCritic pairwise decision records;
  - selected model;
  - fallback reason if any.
- The final training manifest can trace each mask back to its selected source.

### 2.4 ShapeKit Stage

Purpose:

- Enforce the teacher's requirement that all selected outputs go through
  ShapeKit before pseudo-label dataset assembly.

Required behavior:

- ShapeKit runs after LabelCritic selection, not only after individual model
  inference.
- The input to ShapeKit is the assembled selected pseudo-label case layout.
- If ShapeKit fails or does not produce a specific organ, fallback is allowed
  only with explicit metadata.

Current project status:

- `multimodel_loop.py` now assembles selected masks under
  `selected_pre_shapekit` and then calls ShapeKit once per selected case.
- `scripts/run_em_training.py` currently sets `ENABLE_SHAPEKIT = False`, which
  conflicts with the teacher's formal pipeline. This can be allowed only for
  fast smoke/debug runs, not for final pseudo-label generation.

Acceptance criteria:

- Formal E-step runs must enable ShapeKit.
- `pseudo_label_selection.json` must include per-organ ShapeKit status.
- `review_queue.jsonl` must contain ShapeKit fallback/failure cases.

### 2.5 Pseudo-Label Dataset Manifest

Purpose:

- Build the training dataset that the student actually consumes.

Required behavior:

- Manifest item fields should include:
  - `case_id`
  - `ct_path` or `image`
  - `organ`
  - `prompt`
  - `mask` or `final_mask`
  - `selected_model`
  - `candidate_models`
  - `selection_method`
  - `labelcritic_records`
  - `shapekit_status`
  - `dataset_role = pseudo_label`
  - `ground_truth_status = pseudo_label_candidate`
  - quality/review flags

Current project status:

- `mstep_runner.build_training_manifest()` currently only writes
  `case_id`, `organ`, and `mask_path`.
- `VoxTellStudent.build_training_manifest()` is used by
  `scripts/run_em_training.py`, but it must be checked to ensure it preserves
  enough source/quality metadata.

Acceptance criteria:

- The M-step manifest is not just a list of masks. It is auditable.
- We can answer for any training item: which teacher produced it, whether
  LabelCritic selected it, whether ShapeKit modified/fell back, and whether it
  is review-risky.

### 2.6 3D Prompt-Based Student

Purpose:

- Replace the old VISTA3D-centered student route with a VoxTell-style 3D
  prompt-based student.

Required behavior:

- Keep 3D CT input.
- Use organ text prompts inside the 3D model.
- Produce one binary 3D mask per organ prompt.
- Train on the 373 exact prompt target organs.
- Do not use VISTA3D label IDs as the student label space.

Current project status:

- `scripts/train_voxtell_prompt_student.py` exists as the project-specific
  fine-tuning entry around the VoxTell model.
- `agent-harness/cli_anything/medai/core/voxtell_student.py` exists as the
  wrapper.
- `scripts/run_em_training.py` defaults to
  `MEDAI_STUDENT_BACKEND=voxtell_style_3d_prompt`.
- `scripts/run_student_infer_then_round2.py` has been replaced with the
  VoxTell-style 3D prompt inference route.
- Older repair helpers that still use VISTA3D/127-class student logic are
  guarded behind `MEDAI_ALLOW_VISTA3D_LEGACY=1` for historical reproduction
  only.

Acceptance criteria:

- A dry run can build a VoxTell prompt manifest for the 373 target organs.
- A training dry run validates model directory, text encoder path, and manifest.
- Student inference writes `<case_id>/<organ>.nii.gz` for prompted organs.
- Round2 injects VoxTell student predictions, not VISTA3D/127 predictions.

### 2.7 Round2+ Iteration

Purpose:

- Use the trained student as another candidate, without blindly trusting it.

Required behavior:

- Round2+ should inject the previous student prediction as `student_prev`.
- Candidate selection should compare:
  - first-round best teacher pseudo-label;
  - previous student output;
  - any available teacher candidates if still needed.
- LabelCritic should arbitrate conflicts.
- If student collapses, keep the earlier best pseudo-label.

Current project status:

- `run_em_training.py` supports `preseeded_model_dirs` injection for
  `student_prev`.
- The separate `run_student_infer_then_round2.py` script now writes VoxTell
  prompt-student predictions in the same `student_predictions/<case>/<organ>`
  layout expected by Round2 candidate injection.

Acceptance criteria:

- Round2 can run with previous VoxTell student masks as candidates.
- Manifest records when the selected source is `student_prev`.
- The system never assumes Round2 student is automatically better.

### 2.8 Student Failure Mining

Purpose:

- Find the examples/classes the student cannot learn, as the teacher requested.

Required behavior:

- Compare student predictions against the first-round selected pseudo-labels.
- Compute Dice, empty-mask status, volume ratio, and shape/geometry mismatch
  flags per case and organ.
- Produce a hard-case table for manual review.

Required output:

- `student_failure_cases.csv`
- `student_failure_cases.json`
- per-organ summary with repeated failure counts.

Current project status:

- `scripts/evaluate_all_organs.py` exists but appears closer to a generic
  student/teacher overlap evaluator.
- A dedicated failure-mining script should be added or the existing script
  should be upgraded to use the Round1 best pseudo-label manifest.

Acceptance criteria:

- We can list case/organ pairs where student-vs-Round1 Dice is persistently low.
- Review reasons distinguish:
  - student empty mask;
  - pseudo-label possibly wrong;
  - ShapeKit fallback;
  - LabelCritic inconclusive;
  - volume outlier;
  - geometry mismatch.

### 2.9 Fine-Label Dataset and Reporting

Purpose:

- Prevent overstating accuracy before real fine labels are available.

Required behavior:

- Keep following up on the JHU edu account and fine-label dataset access.
- Until fine labels are available, report results as pseudo-label consistency.
- After fine labels are available, re-evaluate student and teacher outputs
  against true expert labels.

Acceptance criteria:

- Documentation and logs do not call teacher pseudo-labels true ground truth.
- Metrics are named correctly:
  - `student_vs_pseudo_label_dice`
  - `teacher_candidate_dice_if_reference_available`
  - not `true_accuracy` unless using fine labels.

## 3. Immediate Repair Order

### Priority 1: Stabilize the E-step Selection Path

Tasks:

- Compile and smoke-test `multimodel_loop.py`.
- Verify that selected masks are assembled before ShapeKit.
- Verify that single-candidate, multi-candidate, LabelCritic fallback, no-mask,
  and ShapeKit fallback paths all produce structured metadata.
- Write LabelCritic decision records to durable output, not only nested JSON.

Why first:

- This is the core correction from the meeting. If candidate selection is wrong,
  the student will learn unstable labels.

### Priority 2: Make the Manifest Auditable

Tasks:

- Upgrade manifest writing so training items keep source/quality metadata.
- Ensure VoxTell training manifest includes CT path, prompt, mask, selected
  source, candidate list, ShapeKit status, and review flags.

Why second:

- The student dataset must be explainable. Otherwise we cannot debug which
  model produced a bad label.

### Priority 3: Replace Legacy Round2 Student Inference

Tasks:

- Replace or retire `scripts/run_student_infer_then_round2.py`.
- Use `VoxTellStudent.segment()` and the 373 target prompt config.
- Save predictions in the directory layout expected by `preseeded_model_dirs`.

Why third:

- Round2 must never accidentally fall back to the old VISTA3D/127 path.

### Priority 4: Enable ShapeKit for Formal Runs

Tasks:

- Keep ShapeKit optional only for dry-run/smoke speed.
- Make formal pseudo-label generation use `ENABLE_SHAPEKIT = True` or a CLI flag
  that defaults to true for final runs.

Why fourth:

- The teacher explicitly required all outputs to pass ShapeKit.

### Priority 5: Add Student Failure Mining

Tasks:

- Add a dedicated script comparing student predictions to Round1 selected
  pseudo-labels.
- Output hard-case CSV/JSON and per-organ repeated failure summaries.

Why fifth:

- This directly implements the teacher's instruction to find cases the student
  cannot learn.

### Priority 6: Run End-to-End Smoke Test

Tasks:

- Use a tiny case/organ subset first.
- Include one single-candidate organ and one multi-candidate organ.
- Run E-step selection, ShapeKit path, manifest build, VoxTell training dry-run,
  student inference dry-run, and failure mining.

Why sixth:

- It validates the complete logic without spending full GPU time.

## 4. Reporting Language to Use

Use:

- "Teacher models are fixed inference engines in E-step."
- "Teacher outputs are pseudo-label candidates."
- "For organs with multiple candidates, LabelCritic selects the best current
  pseudo-label."
- "All selected outputs are passed through ShapeKit before dataset assembly."
- "The student is a 3D prompt-based VoxTell-style model."
- "Current exact target space is 373 organs."
- "Without JHU fine labels, metrics measure consistency with pseudo labels, not
  real accuracy."

Avoid:

- "Teacher models train on 50 cases."
- "Teacher output is ground truth."
- "Student is worse/better in true accuracy" without fine labels.
- "VISTA3D defines the target labels."
- "127 organs" as the main system target.
- "2D conversion" as the next student direction.

## 5. Open Risks

- JHU fine-label dataset is still unavailable, so true accuracy evaluation is
  blocked.
- ShapeKit may fail or not output every selected organ; fallback must be logged.
- LabelCritic may be unavailable if the VLM server is down; fallback selection
  must be explicit.
- VoxTell/Qwen memory requirements are high; training may require environment
  isolation or GPU memory tuning.
- The current codebase still contains legacy VISTA3D student scripts, so they
  must be clearly separated from the default mainline.
- Full 21-model Drive alignment and the active 13-teacher training pool must not
  be confused: the inventory may contain 21 models, while the current training
  teacher pool can be a subset that is enabled and runnable.

## 6. Definition of Done

This meeting requirement is implemented only when all of the following are true:

- E-step runs fixed teacher inference and records all candidate masks.
- Multi-candidate organs are selected by early LabelCritic.
- Single-candidate organs are marked as pseudo-label candidates, not true ground
  truth.
- Every selected output goes through ShapeKit or records a ShapeKit fallback.
- The pseudo-label dataset manifest is source-aware and quality-aware.
- VoxTell-style 3D prompt student training is the default M-step.
- Student target space is 373 exact organs, not VISTA3D 127.
- Round2+ can compare student predictions with first-round best pseudo-labels.
- Hard-case mining produces review lists for student failures.
- Documentation and reporting avoid true accuracy claims before JHU fine labels
  are available.
