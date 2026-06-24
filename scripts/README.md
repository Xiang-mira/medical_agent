# Script Index

## Primary execution

| Script | Purpose |
| --- | --- |
| `run_em_training.py` | formal multi-round EM workflow |
| `train_voxtell_prompt_student.py` | VoxTell prompt-student training |
| `check_gpu_resources.py` | GPU resource gate |

## Build and preparation

Scripts prefixed with `build_`, `generate_`, `prepare_`, `select_`, and
`split_` create configuration artifacts, manifests, prompt banks, datasets,
and model-specific inputs.

## Audit and verification

Scripts prefixed with `audit_`, `verify_`, `validate_`, and `evaluate_`
perform contract checks, routing audits, integrity validation, and evaluation.

## Inference adapters

`atlasnet_predict_and_split.py`, `nnunetv2_predict_and_split.py`,
`unest_predict_and_split.py`, and `vista3d_predict_and_split.py` normalize
model outputs to the repository mask contract.

## Experiments and repair

Scripts prefixed with `run_`, `replay_`, `rerun_`, `repair_`,
`finalize_`, and `rescore_` operate experiment, recovery, and rescoring
workflows.

## Data acquisition

`download_pants_dataset.sh`, `download_pants_dataset.ps1`,
`download_pants50_selective.py`, and `download_pants_mini.py` provide current
PanTS download paths. Historical download helpers are stored in `legacy/`.
