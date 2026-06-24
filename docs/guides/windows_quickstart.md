# Windows 傻瓜级教程

## 1. 解压

把 zip 解压，例如：

```text
D:\medai_agent_loop_cli_anything_final
```

## 2. 创建环境

```powershell
conda create -n medai-agent python=3.10 -y
conda activate medai-agent
cd D:\medai_agent_loop_cli_anything_final
pip install -r agent-harness\requirements.txt
pip install -e agent-harness
```

如果 TotalSegmentator 安装慢，可以先不用它，直接跑 mock demo。

## 3. 检查环境

```powershell
python run_medai_cli.py --json doctor
```

看到 ShapeKit 的 `has_main_py: true` 即说明老师给的 ShapeKit 源码路径正确。

## 4. 生成测试病人

```powershell
python scripts\create_radthinking_demo.py
```

它会生成假的 CT、report、clinical.json、pathology.json、demo mask。仅用于测试命令，不是临床数据。

## 5. 检查 RadThinking-style patient folder

```powershell
python run_medai_cli.py --json radthinking-check --patient-folder data\radthinking_demo\patient_001
```

## 6. 跑完整 agent-loop demo

```powershell
python run_medai_cli.py --json agent-loop `
  --patient-folder data\radthinking_demo\patient_001 `
  --output-folder outputs\agent_loop_mock `
  --backend custom `
  --model-command "python third_party\mock_model\mock_seg_infer.py --image {image} --output {output}" `
  --postprocess shapekit `
  --shapekit-root third_party\ShapeKit-main `
  --organ liver `
  --expected-organs liver
```

## 7. 看结果

```text
outputs\agent_loop_mock\agent_state.json
outputs\agent_loop_mock\final_summary.json
outputs\agent_loop_mock\patient_traces.json
outputs\agent_loop_mock\review_queue.jsonl
```

## 8. 换成真实 TotalSegmentator

```powershell
python run_medai_cli.py --json agent-loop `
  --patient-folder data\your_patient_folder `
  --output-folder outputs\agent_loop_real `
  --backend totalseg `
  --postprocess shapekit `
  --shapekit-root third_party\ShapeKit-main `
  --fast `
  --organ liver `
  --expected-organs liver,pancreas,aorta,postcava
```

## Important Windows Fix / Smoke Test

If the mock/custom backend reports `infer_status: failed`, update to this checked package. The fixed version runs custom command templates using Windows-safe command execution instead of `shlex.split`, which can break backslash paths in PowerShell.

Fast smoke test with one scan:

```powershell
python scripts\create_radthinking_demo.py
python run_medai_cli.py --json agent-loop `
  --patient-folder data\radthinking_demo\patient_001 `
  --output-folder outputs\agent_loop_mock_one_scan `
  --backend custom `
  --model-command "python third_party\mock_model\mock_seg_infer.py --image {image} --output {output}" `
  --postprocess shapekit `
  --shapekit-root third_party\ShapeKit-main `
  --organ liver `
  --expected-organs liver `
  --max-scans 1
```

Expected result: `infer_status` should be `success`, and the output folder should contain `agent_state.json`, `patient_traces.json`, `review_queue.jsonl`, and `final_summary.json`.

For a faster check without ShapeKit, use `--postprocess none`.


## Final v4 notes

This package includes the v4 correction for the teacher meeting: the VLM/LabelCritic projection now defaults to LabelCritic `ProjectDatasetFlex_single.py` + `projection.py` rather than a naive average/slice projection. See `docs/LABELCRITIC_PROJECTION_UPDATE.md`, `docs/TASK_COMPLETION_MATRIX_V4.md`, and `docs/FINAL_DELIVERABLE_V4_STATUS.md`.
