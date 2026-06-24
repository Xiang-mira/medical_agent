# Windows-safe package notes

This delivery zip is built with a short top-level folder name and excludes files that are not needed to run the MedAI CLI workflow but can break Windows Explorer extraction because of long paths.

Excluded from the Windows-safe zip:

- `outputs/`, `.pytest_tmp*/`, and cache folders generated during validation.
- `checkpoints/` and private model weights. Use `scripts/install_epai_421_checkpoint.ps1` to install the local ePAI 2025-04-21 checkpoint after extraction.
- `third_party/ePAI-main/backup_model/`, which contains legacy backup configs with very long file names. The active ePAI 421 wrapper uses `third_party/ePAI-main/train/`.
- `docs/checkpoint_drive_export/sources/`, which is an expanded source/checkpoint export snapshot. The model registry and catalog summaries are kept.
- nnUNet integration-test and benchmarking example folders that are not used by the workflow.

The functional source files, wrappers, CLI entrypoints, registry configs, LabelCritic/ShapeKit wrappers, ePAI 421 label metadata, and workflow documentation are retained.
