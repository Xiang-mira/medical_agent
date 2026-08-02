# Task2 Test Report

Date: 2026-08-01

## Commands

- `python -m py_compile agent-harness/cli_anything/medai/core/registered_infer.py agent-harness/cli_anything/medai/core/multimodel_loop.py agent-harness/cli_anything/medai/medai_cli.py scripts/nnunetv2_predict_and_split.py scripts/unest_predict_and_split.py tools/dataset_delivery/delivery_lib.py tools/dataset_delivery/task2_audit.py`
  - passed: syntax/import compile completed
  - failed: 0

- `PYTHONPATH=agent-harness:. pytest -q tests/dataset_delivery`
  - passed: 29
  - failed: 0
  - skipped: 0
  - additional coverage: UNEST command rendering honors `MEDAI_UNEST_PYTHON` for the MONAI bundle subprocess.

- `PYTHONPATH=agent-harness:. pytest -q agent-harness/tests/test_hierarchical_identity.py::test_hierarchical_case_runs_major_then_child_roi agent-harness/tests/test_hierarchical_identity.py::test_roi_planner_blocks_child_when_parent_missing`
  - passed: 2
  - failed: 0
  - skipped: 0

- `PYTHONPATH=agent-harness:. python tools/dataset_delivery/task2_audit.py --output-dir reports`
  - passed: generated path inventory, 23-class scope, registry/taxonomy audit, kidney parent inventory placeholder, and ATM/AirRC/UNEST dry-run plans
  - failed: 0

- `PYTHONPATH=agent-harness:. python run_medai_cli.py --json infer --image /tmp/missing/ct.nii.gz --output /tmp/medai-task2-dryrun --model atm --registry configs/model_registry.yaml --dry-run`
  - passed: ATM command resolves to `/home/xhan74/envs/medical_agent/bin/python` and `/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict`
  - failed: 0

- `PYTHONPATH=agent-harness:. python run_medai_cli.py --json validate-373-target --organs airway_tree,airway_wall,lung_pulmonary_arteries,lung_pulmonary_veins,kidney_cortex,kidney_medulla,kidney_pelvicalyceal_system --allow-subset`
  - passed: requested task2 subset is inside the current 373 target config
  - failed: 0

- `rg -n '^kidney_cortex,' reports/task2_registry_taxonomy_audit.csv`
  - passed: kidney_cortex reports `body_region=abdomen and retroperitoneum` and `fov_region=abdomen and retroperitoneum`
  - failed: 0

- `git diff --check`
  - passed: no whitespace errors
  - failed: 0

## Not Run

- Single-case GPU smoke for ATM, AirRC, and UNEST was not run on this development machine because the fixed `/projects/bodymaps/...` data paths, `/home/xhan74/...` Python environments, and Slurm/T4 GPU runtime are not mounted here. `reports/task2_resolved_path_inventory.json` records these paths as missing on this machine.
