# Scheduler Audit

## Repository State

- Git root: `/home/teacher1/JHU-project1/medical_agent`
- Main medical CLI: `run_medai_cli.py` and `agent-harness/cli_anything/medai/medai_cli.py`
- Main EM entrypoint: `scripts/run_em_training.py`
- Student training entrypoint: `scripts/train_voxtell_prompt_student.py`
- Student inference entrypoint: `scripts/student_auto_segmentation_cli.py`
- LabelCritic profile support already exists in `agent-harness/cli_anything/medai/core/labelcritic_config.py`

## Key Findings

- The existing project already has a 373-target source config at `configs/student_3d_prompt_target_organs.json`.
- Existing validation logic treats 373 as the formal target space; pilot338 needs a separate target profile instead of pretending to be full 373.
- The repository did not contain the AbdomenAtlasPro pilot338 manifests or target config before this scheduler implementation.
- Existing LabelCritic profile logic already knows the official 72B model id and revision.
- Existing Student training has no validated DDP path, so the scheduler must enforce single-GPU training while allowing T4 or A100 as the selected GPU type.
- Existing SLURM coverage was limited to example scripts, not a reusable DAG/array/dependency scheduler.

## Safety Constraints Captured

- `IMAGE_ROOT` and `MASK_ROOT` are policy read-only.
- No training, inference, or LabelCritic task may receive GT or eval-reference manifests.
- Metrics is the only stage allowed to read eval-reference manifests.
- Teacher/Student inference must run as per-case job arrays.
- LabelCritic vLLM service and client must run in the same SLURM job on the same node.

## Local Audit Limitation

The local machine cannot see the `/projects/bodymaps/...` AbdomenAtlasPro roots or the fixed pilot manifests. Local preflight therefore validates repository contracts and warns that HPC preflight must verify those paths and manifests.
