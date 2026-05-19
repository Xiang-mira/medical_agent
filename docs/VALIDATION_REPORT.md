# Validation Report

## Static checks

- Python package imports: `agent-harness/tests/test_imports.py`
- Core module logic: `agent-harness/tests/test_core.py` (24 tests — label verifier, annotation manager, VLM parsing, RadThinking negation, QC checker, projection builder, EM loop)
- CLI entrypoint: `run_medai_cli.py` and console script `medai-cli`
- All major commands emit JSON

Run full test suite:

```powershell
cd agent-harness
python -m pytest tests/ -v
# Expected: 24 passed
```

## Runtime smoke test

```powershell
python run_medai_cli.py --json doctor
python scripts\create_radthinking_demo.py
python run_medai_cli.py --json radthinking-check --patient-folder data\radthinking_demo\patient_001
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

## VLM Label Expert smoke test (requires Ollama)

```powershell
# Start Ollama in background first:
# Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

python run_medai_cli.py --json vlm-label-expert `
  --ct-image data\pants_real\PanTS_00000023\scans\PanTS_00000023\ct.nii.gz `
  --annotation-a data\pants_real\PanTS_00000023\reference_labels\PanTS_00000023\segmentations\pancreas.nii.gz `
  --annotation-b outputs\pants_batch_totalseg\scan_outputs\PanTS_00000023\raw_predictions\PanTS_00000023_PanTS_00000023\segmentations\pancreas.nii.gz `
  --organ pancreas `
  --output-folder outputs\vlm_smoke_test `
  --vlm-backend ollama `
  --vlm-model qwen2.5vl:7b `
  --dsc-vlm-threshold 0.8
```

## Security note: shell=True in custom inference

`run_custom_inference()` in `totalseg_runner.py` calls `subprocess.run(..., shell=True)`.

**Why:** The `--model-command` option accepts a template string like
`"python third_party\mock_model\mock_seg_infer.py --image {image} --output {output}"`.
On Windows, `shlex.split()` uses POSIX escaping and mishandles backslash paths,
causing silent failures. `shell=True` resolves this reliably.

**Risk scope:** The command string comes from the CLI caller (you), not from any
external input, network request, or user-submitted data. This is a local research
tool — no web server, no untrusted input path. The risk of shell injection is
limited to whoever runs the script, which is the same person who controls the
command template.

**Mitigation:** If this CLI is ever wrapped in a web API or accepts model-command
strings from untrusted sources, replace `shell=True` with explicit argument list
parsing and validate all path segments before execution.

## Known limitations

- GPU retraining (M-step) is a stub — logged but not executed; no GPU available.
- TotalSegmentator is an anatomical segmentation model, not a pancreatic tumor model. Use `--backend custom --model-command "..."` to swap in a PanTS-specific model.
- RadThinking trace uses rule-based report parsing. LLM-based extraction can be added later.
- This project does not perform clinical diagnosis.

## Changelog

| Date | Fix |
|------|-----|
| 2026-05-08 | Custom backend switched to `shell=True` to fix Windows path handling |
| 2026-05-09 | Batch output changed to per-case directories (prevents overwrite) |
| 2026-05-09 | TotalSegmentator timeout added (`timeout_sec` param) |
| 2026-05-09 | QC checker temporal logic: `if/if` → `if/elif`, added SHRINKING to medium severity |
| 2026-05-09 | VLM Label Expert implemented (Ollama qwen2.5vl:7b, projection builder, DSC routing) |
| 2026-05-09 | AnnotationManager versioned storage added |
| 2026-05-09 | mini EM loop (`em_loop.py` + `em-loop` CLI) added |
| 2026-05-09 | `max_iterations` now drives a real refinement loop in `agent_controller.py` |
| 2026-05-10 | RadThinking negation-aware suspicious term detection fixed |
| 2026-05-10 | 21 unit tests added covering all core modules |
| 2026-05-10 | `agent_loop_trace` and `radthinking_style_reasoning` fields added to agent-loop JSON output |
| 2026-05-10 | README.md updated: removed stale "VLM未实现" language, added VLM/EM Loop/AnnotationManager |
| 2026-05-10 | SKILL.md updated: added all new commands (vlm-label-expert, em-loop, label-verify, etc.) |
| 2026-05-10 | `docs/MATERIAL_SUMMARY_CN.md` created: 5-material study summary (teacher Priority 1 deliverable) |
| 2026-05-10 | **Bug fix** `apply_update` winner=A now uses explicit candidate_a instead of always falling back to raw |
| 2026-05-10 | **Bug fix** VLM keyword fallback tightened: ambiguous "candidate a" → uncertain (medical conservative) |
| 2026-05-10 | **Bug fix** Dead text-only fallback branch removed from vlm_label_expert |
| 2026-05-10 | **Bug fix** VLM now sends up to 2 projection views (axial + coronal), was only 1 |
| 2026-05-10 | **Bug fix** projection_builder checks mask/CT shape mismatch, emits shape_affine_warnings |
| 2026-05-10 | **Feature** Projection PNGs now have "Candidate A / Candidate B" yellow text labels |
| 2026-05-10 | **Feature** em-loop now supports --postprocess shapekit (inference → ShapeKit → E-step) |
| 2026-05-10 | **Feature** agent-loop now supports --enable-vlm-refinement / --vlm-backend-refinement / --vlm-model-refinement |
| 2026-05-10 | **Feature** M-step now records data_annealing_plan per round in rounds_metrics.json |
| 2026-05-10 | **Docs** dsc_vlm_threshold 0.5 vs 0.8 distinction documented in MEETING_DEMO_REPORT_CN.md |
| 2026-05-10 | 24 unit tests passing (added disjoint-mask VLM routing + strict keyword fallback tests) |
