# Task 1 Rename Workflow

Task 1 is only for existing mask files whose source and target names refer to the same anatomy, same laterality, and same granularity. It performs no voxel edits, no merges, no splits, and no generated masks. Task 2 covers missing masks, coarse-to-fine targets, vessel segment refinement, tissue subclass generation, and all fixed 23 generate targets in `task2_generate_targets_23.csv`.

Canonical source is `configs/dataset_delivery/task1/organ_rename_mapping_373.csv`; boundary evidence is `task_boundary_classification.csv`; many-to-one aliases are explicit in `task1_alias_groups.csv`. The `main` branch is the only maintenance branch for this workflow. The standalone directory under `deliverables/task1_rename_refactor_source/` is generated with `tools/dataset_delivery/export_task1_standalone.py`; do not hand-maintain a second mapping.

Semantic taxonomy evidence for proposal review lives in `taxonomy_semantic_evidence.csv`. It intentionally separates medical concept equivalence from dataset label scope and from actual voxel equality. Token order and exact medical synonym rows can be confirmed at the semantic layer, but parent-child, aggregate-vs-sided, whole-vs-substructure, and ambiguous rows remain non-executable unless data dictionary evidence and exact voxel equality also support them.

Static validation:

```bash
python tools/dataset_delivery/validate_rename_mapping.py \
  --mapping configs/dataset_delivery/task1/organ_rename_mapping_373.csv \
  --taxonomy configs/student_3d_prompt_target_organs.json \
  --task2-targets configs/dataset_delivery/task1/task2_generate_targets_23.csv \
  --alias-groups configs/dataset_delivery/task1/task1_alias_groups.csv \
  --output-json reports/task1_mapping_validation.json \
  --output-md reports/task1_mapping_validation.md
```

HPC 100-case dry-run is read-only by default and writes only reports under `--output-root`. `would_rename` means exactly one confirmed source exists and target is absent. `already_normalized` means target exists and source is absent. `source_missing` means neither source nor target exists in that case. `conflict` means target exists beside source or multiple alias sources coexist, so nothing is chosen automatically.

Output-root apply is optional and must use `--apply --output-data-root <different-root>`; the source data root is refused when equal to the output root and original masks are not modified.

Task 1 finalization proposal audit is read-only and never applies renames:

```bash
python tools/dataset_delivery/propose_task1_finalization.py \
  --data-root /path/to/AbdomenAtlasPro \
  --case-manifest /path/to/cases_100_manifest.csv \
  --output-root /path/to/task1_proposal_audit
```

On HPC, set `CODE_ROOT`, `PYTHON`, `DATA_ROOT`, `CASE_MANIFEST`, and `OUT_ROOT`, then run:

```bash
bash scripts/run_task1_proposal_audit_hpc.sh
```

The proposal audit writes only under `OUT_ROOT`. It creates `proposed/` CSVs, semantic candidate reports, alias voxel equivalence rows, blocked cases, pending semantic review rows, and `proposal_summary.json/.md`. It records before/after metadata fingerprints for source NIfTI files and canonical config files; if any filename, size, mtime, or canonical config hash changes, the command fails.
