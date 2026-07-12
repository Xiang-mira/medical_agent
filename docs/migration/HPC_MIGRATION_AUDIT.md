# HPC Migration Audit

This audit records the selected restore set for moving the project to an HPC GPU
through OnDemand Shell or OnDemand VS Code Server. It intentionally avoids a
blind copy of the local 231G workspace.

## Destinations

- GitHub: 1163 tracked source/config/doc files.
- Hugging Face private repo: `Xiang-mira/MedIA-Agentic-AI-Private-HPC` with 39144 files, 29.49 GiB.
- Excluded: 139517 rows covering PanTS data, caches, public duplicate Qwen VL models, temporary files, and bad/smoke output categories.

## Required HF restore roots

The private HF repo is laid out so that downloading to `checkpoints/` restores
teacher assets at their expected local paths. It also contains
`student_models/em_round1_25case_full_mstep_lr3e-5_20260711/` and filtered
formal state under `outputs/`.

## Sensitive Data Guardrails

PanTS images/labels, PanTS tarballs, NIfTI volumes, probability arrays, runtime
caches, and VISTA3D/UCSF patient CSV metadata are excluded. Qwen2-VL and
Qwen2.5-VL are public upstream models and are documented for direct download
rather than mirrored into the private migration repo.

## Verification

Use `scripts/verify_hf_asset_manifest.py` against `hf_asset_manifest.tsv` after
downloading the HF private repo. Use the README HPC section for the full restore
and smoke-test sequence.
