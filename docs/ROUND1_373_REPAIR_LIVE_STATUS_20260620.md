# Round1 373 Repair Live Status — 2026-06-20

## Running experiment

- PID at verification: `435981`
- Output: `outputs/round1_373_hierarchical_repair_20260620/estep`
- Scope: 50 PanTS cases × 373 targets
- Source cache: `outputs/stage4b_round1_50cases_20260611/round1/estep`
- Mode: hierarchical ROI, old masks parent-cache-only, ShapeKit enabled,
  conservative weighted fusion, LabelCritic stub, real VLM absolute grade off.
- Verified after launch: GPU child-ROI process active; four ROI inputs created for
  case 1; no fresh `hierarchical_full` output was present.

Command:

```bash
python scripts/run_full_373_hierarchical_repair.py \
  --num-cases 50 \
  --output-dir outputs/round1_373_hierarchical_repair_20260620/estep \
  --old-estep outputs/stage4b_round1_50cases_20260611/round1/estep \
  --critic-backend stub --timeout-sec 1800 --device cuda
```

## Fixed before launch

- ABCD/zero-volume/ShapeKit/LabelCritic/fusion policies.
- Preseed resolver now prefers binary `segmentations/` over combined labels.
- Old Round1 child masks are quarantined with `preseeded_parent_only=True`.
- Cached backups are preferred over uncached primary major teachers.
- Completed zero-output old inference is not rerun on out-of-scan anatomy.
- Normalized taxonomy keys no longer drop `vertebrae_L2`/other display IDs.
- Whole kidney may use approved left+right kidney union.
- Current 373 routing validation: 373/373, zero blocking routes.
- Full test suite before launch: 122 passed.

## Required after the run finishes

1. Validate every `hierarchical_inference_plan.json`: all child inference scopes
   are `child_roi`; no fresh full-volume child inference; parent-missing children
   are blocked.
2. Validate all selected identities and ensure no old Stage4B child path appears
   in selected metadata or the training manifest.
3. Report A/B/C/D distribution, zero-volume, ShapeKit fallback, fusion rejection,
   and unresolved/out-of-scan counts.
4. Build `mstep/voxtell_prompt_student_manifest.json` from repaired
   `annotation_versions`; exclude positive D/zero-weight items (allowed negative
   prompt zero masks remain separate negative supervision).
5. Run M-step manifest dry-run/sanity gate before starting VoxTell training.
6. Real LabelCritic was unavailable at `localhost:8000`; if required later,
   start the service and replay only uncertain/conflicting selections rather
   than rerunning teacher inference.

## Progress update — 2026-06-21 02:52 UTC

- Main process remains alive (PID `435981`), GPU child-ROI inference active.
- Completed: 15/50 cases; case 16 (`PanTS_00000368`) in progress.
- Intermediate audit over 15 cases: 826/826 task records are `child_roi`;
  0 fresh full-volume major runs; 0 identity mismatch; 0 old child-mask leaks.
- Selected masks: 2,104 before prompt expansion.
- Regrade policy fixed so advisory-only fallback/low pseudo-consistency cannot
  create D; D remains reserved for structural hard failure.
- Partial M-step gate succeeded after regrading all 2,104 masks: 18,276 positive
  prompt-expanded items, 0 D/zero-weight positives, 0 identity failures, 0 old
  child leaks. Evidence:
  `outputs/round1_373_hierarchical_repair_20260620/mstep/repair_mstep_gate.json`.
- Full suite after these fixes: 124 passed.
- Finalizer command after 50/50 completion:

```bash
python scripts/finalize_373_repair_for_mstep.py
```

## Scope change — 2026-06-21 03:38 UTC

- Requested formal scope changed from 50 cases to the first 20 cases.
- The 50-case process was stopped safely at 16 completed cases.
- Resume handling was fixed for parent-cache-only hierarchical runs: legitimate
  selected-or-gap completion no longer requires new raw/full-volume outputs.
- The 20-case process resumed successfully: cases 1–16 were verified complete
  and skipped; case 17 is running. Remaining E-step estimate: roughly 2–3 hours.
- Final M-step gate command: `python scripts/finalize_373_repair_for_mstep.py
  --expected-cases 20`.
