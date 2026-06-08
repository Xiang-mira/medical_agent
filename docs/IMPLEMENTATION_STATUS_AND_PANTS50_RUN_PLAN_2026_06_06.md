# 373-Organ Auto Fine-Label + VoxTell Student Status and PanTS50 Run Plan

Date: 2026-06-07

This document is the current execution checkpoint for the 373-organ auto
fine-label and VoxTell-style student distillation goal. The formal engineering
target is 373 organs. Historical counts such as 383/384/377/358/127 must not be
used as formal target spaces.

## 1. Current Completion Matrix

| Requirement | Current status | Evidence | Remaining action |
| --- | --- | --- | --- |
| Formal target is exactly 373 organs | Complete | `outputs/audit_21_models/routing_373_audit.json` reports 373 targets, 373 unique, 0 blocking | Keep fail-fast target validation in all formal entrypoints |
| PanTS 50 case list is usable | Complete after repair; reference quality caveat found | `outputs/audit_21_models/case_list_50_validation_latest.json` reports 50 usable cases and 45 fully non-empty core-reference cases | Do not treat empty reference masks as low-quality teacher evidence |
| Teacher registry/Drive static alignment | Complete with caveats | `outputs/audit_21_models/drive_alignment_report.json` | Track DAPS/VISTA3D old-manifest caveats and MOOSE plans-name mismatch |
| Teacher dry-run command readiness | Complete | `outputs/audit_21_models/teacher_readiness_current.json` reports 22/22 ready for command | Continue real runnable checks as scale increases |
| E-step candidate selection path | Complete for 50-case / 10-organ pilot subset | `outputs/pants50_stage4_50case_combined_summary.json` reports 500/500 selected items | Expand from 10 organs and 3 models to full 373-organ/all-ready-teacher formal run |
| ShapeKit formal path | Complete for tiny and real subset | `current_system_tiny_shapekit_check` and `current_real_teacher_subset_check` show `shapekit_status=success` | Track fallback rate during scale-up |
| LabelCritic metadata path | Complete structurally | Real subset writes 5 VLM decision records with stub backend | Real VLM quality still needs conservative review/fallback |
| Label passport / grade / training weight | Complete for selected masks | Real subset has 5 passport files and manifest grade/weight fields | Monitor grade distribution during larger runs |
| Auto fine-label dashboard | Complete for current layouts | Dashboard emits 373 rows from `estep/annotation_versions` | Run after every scale stage |
| VoxTell wrapper contract | Complete for dry-run/contract, including 50-case pilot prompts | `outputs/pants50_stage4_50case_student_dryrun/PanTS_00000026/voxtell_student_plan.json` | Run real official reproduction before student quality claims |
| VoxTell fine-tuning | Contract complete for 50-case pilot manifest; meaningful training not complete | `outputs/pants50_stage4_50case_voxtell_train_dryrun/voxtell_prompt_train_result.json` validates 489 trainable items | Train only after A/B/C weighted manifest passes QC |
| Round2 student competition | Script-level path exists, not complete at scale | Dry-run checks only | Run after real student inference |
| Student failure mining | Complete structurally | Smoke/dry-run failure mining outputs exist | Run after real student predictions |
| Formal 50 CT x 373 run | Not complete | 50-case pilot covered 10 organs and selected model subset only | Run after resource planning for all 373 organs and all ready teachers |

## 2. Fixes Applied In This Execution Pass

- Repaired `data_manifest/case_list_50_tumor.csv` by filling 25 missing
  `annotation_folder` values from the local PanTS layout.
- Added `scripts/repair_case_list_50_annotations.py` so the repair is
  reproducible.
- Improved `scripts/validate_case_list_50.py` so it can derive local annotation
  folders by case id and distinguish blocking errors from warnings.
- Extended PanTS validation to flag empty core-organ reference masks. The 50
  cases remain usable for auto-label generation, but empty references are no
  longer counted as fully validated reference evidence.
- Hardened `multimodel_loop.py` so any requested non-target organ fails fast
  instead of silently proceeding outside the 373-organ formal target space.
- Hardened reliability scoring so `empty_reference_nonempty_prediction` does
  not incorrectly turn a non-empty teacher prediction into a low-Dice D-grade
  label.
- Improved `scripts/build_auto_fine_label_dashboard.py` so it can read common
  run layouts such as `<run>/estep/annotation_versions`.
- Added `scripts/build_combined_voxtell_manifest.py` to combine staged PanTS
  manifests into one deterministic VoxTell prompt-student training manifest.

## 3. Verification Commands Already Passed

```bash
python scripts/validate_case_list_50.py \
  --case-list data_manifest/case_list_50_tumor.csv \
  --output-json outputs/audit_21_models/case_list_50_validation_with_nonempty_core_check.json

python scripts/audit_373_organ_routing.py
python scripts/audit_21_model_drive_alignment.py
python scripts/audit_teacher_readiness.py \
  --models cads551,cads552,cads553,cads554,cads555,cads556,cads557,cads558,cads559,moose666,moose888,nnunet_private,saros_nnunet,atm,airrc,lvp,daps,epai_20250421,vsmtrans,vista3d,unest,totalsegmentator \
  --output outputs/audit_21_models/teacher_readiness_current.json

pytest -q agent-harness/tests/test_core.py

python scripts/run_teacher_plan_tiny_check.py \
  --num-cases 1 \
  --organs liver,spleen \
  --enable-shapekit \
  --output-dir outputs/audit_21_models/current_system_tiny_shapekit_check

python scripts/run_real_teacher_subset_check.py \
  --num-cases 1 \
  --models epai_20250421,cads551,mock_seg \
  --organs liver,pancreas,spleen,kidney_left,kidney_right \
  --critic-backend stub \
  --enable-shapekit \
  --output-dir outputs/audit_21_models/current_real_teacher_subset_check
```

## 4. Stage 1 Pilot Result

The first 5-case pilot completed successfully at
`outputs/pants50_stage1_5case_epai_cads551_mock`.

Observed result:

- Cases: 5.
- Organs per case: 10.
- Manifest items: 50.
- Updated case-organs: 50/50.
- LabelCritic stub records: 40.
- ShapeKit statuses: all `success`.
- Label passports: 50.
- Dashboard rows: 373.
- Selected source models: `cads551`, `epai_20250421`.
- Grade distribution before empty-reference scoring fix: A=10, C=38, D=2.

Quality findings:

- `PanTS_00000035/kidney_left` had an empty PanTS reference mask; the selected
  teacher prediction should be reviewed but should not be penalized as a true
  low-Dice failure. This was fixed and verified with
  `outputs/audit_21_models/empty_reference_regression_check_v2`.
- `PanTS_00000047/stomach` had low pseudo-consistency and volume-ratio review
  flags; keep it out of strong training until reviewed or improved by additional
  teachers.
- Stub LabelCritic intentionally returns uncertain decisions, so many items are
  conservative fallback selections. This is acceptable for structural scale-up
  but not enough for final quality claims.

## 5. Stage 2 Pilot Result

The 10-case pilot completed successfully at
`outputs/pants50_stage2_10case_epai_cads551_mock`.

Observed result:

- Cases: 10.
- Organs per case: 10.
- Manifest items: 100.
- Updated case-organs: 100/100.
- LabelCritic stub records: 79.
- ShapeKit statuses: all `success`.
- Label passports: 100.
- Dashboard rows: 373.
- Grade distribution: A=21, C=78, D=1.
- Empty-reference items: 3, now tracked without being converted into D-grade
  labels.
- Remaining D-grade item: `PanTS_00000047/stomach`, caused by genuinely low
  pseudo-consistency plus volume-ratio review flags.

VoxTell contract checks from Stage 2:

- Built `outputs/pants50_stage2_10case_epai_cads551_mock/voxtell_prompt_manifest.json`.
- Manifest has 100 items, 10 cases, and 0 missing CT images.
- `scripts/train_voxtell_prompt_student.py --dry-run` succeeded with
  freeze-encoder training plan.
- `scripts/run_student_infer_then_round2.py --dry-run` succeeded for one case
  and four prompts.

Latest validation note:

- `outputs/audit_21_models/case_list_50_validation_latest.json` reports 50
  usable cases and 45 fully non-empty core-reference cases.
- Empty core-reference masks currently occur in:
  `PanTS_00000035/kidney_left`, `PanTS_00000100/spleen`,
  `PanTS_00000100/kidney_left`, `PanTS_00000270/pancreas`,
  `PanTS_00000554/kidney_left`, and `PanTS_00000881/kidney_left`.

## 6. Stage 3 Pilot Result

The 30-case gate is represented by the first 10-case run plus the incremental
case 11-30 run.

- Summary file: `outputs/pants50_stage3_30case_combined_summary.json`.
- Combined cases: 30.
- Organs per case: 10.
- Combined manifest items: 300.
- ShapeKit statuses: all `success`.
- Source metadata missing: 0.
- Grade distribution: A=58, B=8, C=227, D=7.
- Empty-reference items: 5, tracked separately from true low-consistency
  failures.
- D-grade items are limited to genuine low pseudo-consistency or volume-ratio
  review cases and have `training_weight=0`.

Stage 3 passed the structural gate. Quality remains conservative because
LabelCritic used the stub backend, so fallback selections require review before
strong claims.

## 7. Stage 4 50-Case Pilot Result

The requested 50 PanTS cases have been run for the current 10-organ pilot
scope using `epai_20250421`, `cads551`, and `mock_seg` with ShapeKit enabled
and the stub LabelCritic backend.

- Summary file: `outputs/pants50_stage4_50case_combined_summary.json`.
- Combined cases: 50.
- Organs per case: 10.
- Combined manifest items: 500.
- Expected items for 10 organs: 500.
- Updated case-organs: 500/500.
- LabelCritic decisions: 390.
- ShapeKit statuses: all `success`.
- Source metadata missing: 0.
- Selected real teacher models: `cads551`, `epai_20250421`.
- Grade distribution: A=96, B=12, C=381, D=11.
- Empty-reference items: 6, tracked separately from true low-consistency
  failures.
- D-grade items: 11, all with `training_weight=0`.

Quality interpretation:

- The run proves the structural 50-case pipeline for the 10-organ pilot:
  routing, E-step candidate selection, ShapeKit post-processing, label
  passports, training manifest fields, dashboards, and VoxTell manifest
  conversion all completed.
- It does not prove expert accuracy. Reported Dice is pseudo-consistency among
  candidate/reference masks, not expert ground-truth DSC.
- The D-grade rows are concentrated in low pseudo-consistency or volume-ratio
  review cases, especially `postcava`, `stomach`, and isolated
  `PanTS_00000836/aorta` and `PanTS_00000836/pancreas`.
- Empty PanTS reference masks no longer force non-empty teacher predictions
  into D grade; they are flagged as
  `empty_reference_nonempty_prediction` and down-weighted for review.

50-case VoxTell contract checks:

- Combined manifest:
  `outputs/pants50_stage4_50case_combined_voxtell_manifest.json`.
- Manifest count: 500 items, 50 cases, 10 organs, 0 missing image, 0 missing
  mask.
- Trainable items: 489; 11 D-grade / zero-weight items are skipped by the
  trainer as intended.
- Training dry-run:
  `outputs/pants50_stage4_50case_voxtell_train_dryrun/voxtell_prompt_train_result.json`
  has `validation_status=ok`.
- Student inference dry-run:
  `outputs/pants50_stage4_50case_student_dryrun/PanTS_00000026/voxtell_student_plan.json`
  validates the 10 pilot prompts, VoxTell official output names, project
  standardized mask names, and the formal 373 target config.

Remaining limitation:

- This is the direct 50-case pilot requested for analysis, not the full formal
  50 CT x 373 organ x all-ready-teacher trial. The formal run should be
  scheduled only after the 10-organ pilot issues are reviewed and the
  all-teacher runtime budget is confirmed.

## 8. Stage Gates For PanTS Scale-Up

### Stage 1: 5 cases

- Cases: first 5 validated rows from `data_manifest/case_list_50_tumor.csv`.
- Organs: liver, pancreas, spleen, kidney_left, kidney_right, stomach,
  duodenum, colon, aorta, postcava.
- Initial models: `epai_20250421,cads551,mock_seg`.
- LabelCritic backend: `stub` first for deterministic structure; real VLM only
  after structural output passes.
- ShapeKit: enabled.
- Pass criteria:
  - E-step status success.
  - Training manifest non-empty.
  - Every item has source metadata.
  - Every final mask has a label passport.
  - Dashboard has exactly 373 rows.
  - Review queue records fallbacks/uncertain cases.
  - No report claims expert ground truth or true accuracy.

### Stage 2: 10 cases

- Expand real teacher subset beyond the 5-case models.
- Include representative abdominal, vessel, bone, lung/airway, and small-organ
  targets.
- Run VoxTell official reproduction and project wrapper real inference on a
  small prompt subset.
- Pass criteria: stable per-organ status, empty masks tracked, no blocking
  geometry mismatch.

### Stage 3: 30 cases

- Broaden organs toward the full 373 target in batches.
- Use all teachers that are proven real-runnable.
- Build A/B/C/D weighted training manifest and run VoxTell mini fine-tuning.
- Pass criteria: student predictions enter Round2 as candidates only, failure
  mining produces actionable review items.

### Stage 4A: 50-case pilot, 10 organs

- Status: complete.
- Scope: 50 cases x 10 organs using the selected model subset.
- Pass criteria:
  - 500/500 case-organ items emitted.
  - 0 missing source metadata.
  - 0 missing image/mask in the combined VoxTell manifest.
  - D-grade items receive zero training weight.
  - Empty-reference items are flagged separately from true low Dice.
  - No report claims expert accuracy.

### Stage 4B: full 50-case formal run

- Full formal trial: 50 CT x 373 organs with all ready teachers.
- Required outputs:
  - raw teacher predictions
  - candidate pools
  - selection metadata
  - final selected masks
  - label passports
  - training manifest
  - VoxTell training/inference outputs
  - Round2 selection manifest
  - dashboards
  - failure reports
  - review queue

## 9. Reporting Rules

- Never call teacher/student outputs expert ground truth.
- Never call pseudo-label Dice true accuracy.
- Report coverage, grade distribution, ShapeKit fallback rate, LabelCritic
  uncertain rate, empty mask rate, pseudo-consistency, runtime, and unresolved
  organs.
- `expert_verified` is reserved for future expert-reviewed labels only.
