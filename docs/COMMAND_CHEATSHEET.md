# Command Cheatsheet

```bash
python run_medai_cli.py --json doctor
python run_medai_cli.py --json presets
python run_medai_cli.py --json check-image --image data/single_case/case_001/ct.nii.gz
python run_medai_cli.py --json infer --image ct.nii.gz --output-folder outputs/raw --backend totalseg --fast
python run_medai_cli.py --json adapt --segmentation-folder outputs/raw/case_001/segmentations
python run_medai_cli.py --json postprocess --input-folder outputs/raw --output-folder outputs/refined --shapekit-root third_party/ShapeKit-main
python run_medai_cli.py --json radthinking-check --patient-folder data/radthinking_demo/patient_001
python run_medai_cli.py --json trace-observation --ct-image ct.nii.gz --mask liver.nii.gz --organ liver
python run_medai_cli.py --json trace-temporal --previous-mask prev_liver.nii.gz --current-mask cur_liver.nii.gz --organ liver
python run_medai_cli.py --json trace-context --report report.txt --clinical clinical.json --organ liver
python run_medai_cli.py --json trace-build --ct-image ct.nii.gz --current-mask liver.nii.gz --organ liver
python run_medai_cli.py --json agent-loop --patient-folder data/radthinking_demo/patient_001 --output-folder outputs/agent_loop --backend custom --model-command "python third_party/mock_model/mock_seg_infer.py --image {image} --output {output}" --postprocess shapekit --expected-organs liver
```

## PanTS real data commands

Show official PanTS download and layout information:

```powershell
python run_medai_cli.py --json pants-info
```

After downloading PanTS/PanTSMini, verify whether actual CT and label files exist:

```powershell
python run_medai_cli.py --json pants-check --pants-root third_party\PanTS-main
```

Windows download helper, only if you have ~300GB+ free disk space:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_pants_dataset.ps1 -OutDir third_party\PanTS-main\data
```

Linux/Git Bash/WSL:

```bash
bash scripts/download_pants_dataset.sh third_party/PanTS-main/data
```

## Scientifically matched model note

TotalSegmentator is a free and robust anatomical segmentation backend. It is useful for verifying the CLI model-wrapper and ShapeKit post-processing path, but it is not a pancreatic tumor-specific model. For a PanTS tumor-segmentation experiment, use the custom backend with a PanTS/R-Super/MedFormer/nnU-Net inference command once the checkpoint and code are available.

## PanTS small-demo commands

查 PanTS 官方信息：

```powershell
python run_medai_cli.py --json pants-info
```

检查本地是否已有真实 PanTS 数据：

```powershell
python run_medai_cli.py --json pants-check --pants-root third_party\PanTS-main
```

定位某个已经下载的 PanTS case：

```powershell
python run_medai_cli.py --json pants-find-case `
  --pants-root third_party\PanTS-main `
  --case-id PanTS_00000001
```

把某个已经下载的 PanTS case 导入成 RadThinking-style patient folder：

```powershell
python run_medai_cli.py --json pants-import-case `
  --pants-root third_party\PanTS-main `
  --case-id PanTS_00000001 `
  --output-root data\pants_small
```

如果只有一个 CT 和 label folder，直接导入：

```powershell
python run_medai_cli.py --json pants-import-files `
  --ct D:\path\to\ct.nii.gz `
  --label-folder D:\path\to\segmentations `
  --output-root data\pants_small `
  --patient-id PanTS_00000001
```

导入后跑 agent-loop：

```powershell
python run_medai_cli.py --json agent-loop `
  --patient-folder data\pants_small\PanTS_00000001 `
  --output-folder outputs\pants_small_agent_loop `
  --backend totalseg `
  --postprocess shapekit `
  --shapekit-root third_party\ShapeKit-main `
  --fast `
  --organ pancreas `
  --expected-organs pancreas,liver,aorta,postcava
```

用 reference labels 做简单 Dice sanity check：

```powershell
python run_medai_cli.py --json pants-eval-case `
  --pred-folder outputs\pants_small_agent_loop\scan_outputs\PanTS_00000001\refined_predictions\PanTS_00000001_PanTS_00000001\segmentations `
  --reference-label-folder data\pants_small\PanTS_00000001\reference_labels\PanTS_00000001\segmentations `
  --organs pancreas,liver,aorta,postcava
```


## Final v4 notes

This package includes the v4 correction for the teacher meeting: the VLM/LabelCritic projection now defaults to LabelCritic `ProjectDatasetFlex_single.py` + `projection.py` rather than a naive average/slice projection. See `docs/LABELCRITIC_PROJECTION_UPDATE.md`, `docs/TASK_COMPLETION_MATRIX_V4.md`, and `docs/FINAL_DELIVERABLE_V4_STATUS.md`.
