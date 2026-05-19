# Validation Report V2

Validated in the ChatGPT sandbox after integrating `drive_lightweight_sources.zip` and `PanTS-main.zip`.

## Commands run

```bash
python run_medai_cli.py --help
python run_medai_cli.py --json build-registry --checkpoint-map configs/class_checkpoint_map.xlsx --output configs/model_registry.yaml --parsed-csv configs/class_checkpoint_map.parsed.csv --checkpoint-root checkpoints
python run_medai_cli.py --json registry-candidates --organs pancreas,liver,aorta,pancreatic_duct
python run_medai_cli.py --json pants-download-50-plan
python run_medai_cli.py --json infer --model epai_20250421 --input data/fake/ct.nii.gz --output outputs/test_epai --dry-run
python run_medai_cli.py --json pants-select-50 --pants-root third_party/PanTS-main --output /tmp/case_list.csv
python -m compileall -q agent-harness/cli_anything/medai scripts
```

## Results

- CLI imports and command registration succeeded.
- `build-registry` parsed 385 anatomical structures and 16 model families.
- `registry-candidates` returned expected candidate models:
  - pancreas: `vsmtrans`, `cads`, `epai_20250421`, `moose3_0`, `mock_seg`
  - liver: `vsmtrans`, `cads`, `moose3_0`, `mock_seg`
  - aorta: `vista3d`, `cads`, `moose3_0`, `mock_seg`
  - pancreatic_duct: `epai_20250421`
- ePAI dry-run command renders to `scripts/nnunetv2_predict_and_split.py` with qchen76_2025_0421 / Dataset1017 model-folder mode.
- `pants-select-50` correctly warns that no real CT/label data are present in the uploaded PanTS repo. This is expected: the uploaded repo contains download scripts, not the actual dataset.

## Not run in sandbox

The real PanTS 50-case workflow was not run because the uploaded PanTS repo does not include CT NIfTI files or labels. The official PanTSMini image blocks are tens of GB each, and the total dataset is 300GB+. Run `scripts/download_pants50_selective.py` on Colab/Linux with enough storage, then run `pants-select-50` and `run-loop`.
