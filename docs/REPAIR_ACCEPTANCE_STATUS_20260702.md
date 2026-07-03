# Repair acceptance status — 2026-07-02

## Result

The implementation contracts are repaired and the full test suite passes, but
the project is not yet eligible to claim final experimental acceptance. A fresh
LabelCritic replay is still required because the available 10-case E-step was
created by the old fallback/fusion policy.

## Verified implementation

- The formal E-step defaults to no candidate fusion and strict LabelCritic
  selection. Inconclusive, disabled, or invalid comparisons are withheld for
  review instead of falling back to Dice or family evidence.
- All 373 targets have complete CT appearance entries. Generated descriptions
  remain marked for medical review.
- The exact organ entry and original-candidate provenance/QC metadata are
  persisted and injected into the third-party LabelCritic prompt.
- Pairwise results are normalized to a strict JSON decision contract.
- Candidate IDs are stable, selected IDs are checked against the candidate
  list, and source/final mask SHA-256 hashes verify mask lineage.
- The full E-step manifest contains one row per case × 373. Missing candidates
  become either evidence-backed `absent_negative` or zero-weight
  `unresolved_review`; the training manifest is a separate eligible subset.
- Sampling logs include configured/actual ratios, positive/negative counts,
  foreground ratio, all-zero count, and negative-reason counts.
- The generic post-processing entry point now executes parent-ROI containment
  before component filtering and records ROI and before/after statistics.
- Metric artifacts distinguish expert GT, teacher, pseudo-label, and all-zero
  references.

## Evidence produced

- Tests: `249 passed`.
- Prompt bank: 373 targets, 373 entries, 365 distinct CT-appearance texts.
- Existing 10-case metadata replay:
  - expected/actual full targets: 3730/3730;
  - evidence-backed absent negatives: 961;
  - unresolved review targets: 1499.
- One-case real containment smoke: 738 masks processed with complete per-mask
  audit fields.

The replay audit intentionally fails the selection portion because old
selection records contain silent fallback decisions and lack the newly required
prompt/lineage fields. Metadata replay is not presented as a substitute for a
fresh VLM selection run.

## Remaining experimental work

1. Start the configured VLM endpoint and run a 2-case × 373 fresh E-step.
2. Require `scripts/audit_repair_plan.py` to pass the new full selection
   manifest before scaling to 10 and 20 cases.
3. Train a short student smoke run and audit `sampling_audit.json`, including
   all-zero loss behavior.
4. Run before/after post-processing evaluation with expert GT where available.
5. After these gates pass, run the complete 10/20-case experiments and publish
   commits. Metrics without expert GT must remain labeled as consistency
   metrics.

## Reproduction commands

```bash
python scripts/build_organ_ct_appearance_373.py
pytest -q agent-harness/tests

python scripts/rebuild_mstep_from_existing_estep.py \
  --source-estep outputs/formal_round1_final_20260627/round1/estep \
  --output-root outputs/repair_acceptance_replay_20260702

python scripts/apply_organ_type_postprocess.py \
  --input-root outputs/formal_round1_final_20260627/round1/student_predictions \
  --output-root outputs/repair_acceptance_replay_20260702/postprocess_smoke \
  --parent-root outputs/formal_round1_final_20260627/round1/estep/cases \
  --max-cases 1
```
