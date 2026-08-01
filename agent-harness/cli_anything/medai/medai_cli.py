from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
import click

from .core.adapter import normalize_totalseg_to_shapekit
from .core.agent_controller import run_agent_loop
from .core.case_sample_builder import build_case_samples, build_convergence_table
from .core.data_checker import check_case_folder, check_ct_image, check_environment
from .core.json_utils import to_jsonable
from .core.itksnap_helper import generate_itksnap_commands
from .core.label_verifier import verify_case
from .core.label_merger import build_runtime_alias_report, merge_case_segmentations
from .core.labelcritic_wrapper import run_labelcritic_compare
from .core.paths import resolve_path
from .core.model_registry import candidate_models_for_organs, load_registry, write_registry, model_inventory, recommend_primary_models_for_organs
from .core.mstep_runner import run_mstep_nnunet_training, run_model_specific_mstep_update
from .core.multimodel_loop import run_multimodel_annotation_loop
from .core.organ_router import route_organs
from .core.registered_infer import run_registered_model
from .core.pants_utils import pants_download_info, check_pants_dataset, find_pants_case, import_pants_case, import_pants_files, evaluate_segmentation_folder
from .core.presets import SHAPEKIT_ABDOMEN_ROI, SHAPEKIT_EXPECTED_ORGANS
from .core.projection_builder import build_projection
from .core.radthinking import build_patient_traces_from_outputs, build_reasoning_trace, check_radthinking_patient, compare_temporal_masks, discover_radthinking_scans, extract_observation, parse_report_sections
from .core.reasoning_trace import create_radthinking_trace_template
from .core.report_supervision import verify_tumor_with_report, batch_verify_with_reports
from .core.shapekit_runner import run_shapekit
from .core.summary import summarize_segmentation_folder, write_run_summary
from .core.totalseg_runner import run_custom_inference, run_totalsegmentator
from .core.target_space import validate_formal_373_target_space
from .core.vlm_label_expert import run_vlm_label_expert


def emit(data: dict) -> None:
    click.echo(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))


def fail(data: dict, code: int = 1) -> None:
    emit(data); raise SystemExit(code)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_organ_option(organs: str, target_config: str = "configs/student_3d_prompt_target_organs.json") -> list[str]:
    """Parse organ CLI option.

    `student_373` is the current teacher-approved mainline target. Keeping this
    as the run-loop default avoids accidentally running only the old 10-organ
    smoke subset when users invoke the generic CLI.
    """
    token = (organs or "").strip()
    if token.lower() in {"student_373", "373", "all_373", "prompt_targets"}:
        config_path = resolve_path(target_config)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        targets = [str(x) for x in data.get("target_organs", [])]
        if not targets:
            raise click.ClickException(f"No target_organs found in {config_path}")
        return targets
    return [x.strip() for x in token.replace(";", ",").split(",") if x.strip()]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "use_json", is_flag=True, default=False, help="Output machine-readable JSON. This CLI always emits JSON for agent compatibility.")
def cli(use_json: bool):
    """medai-cli: CLI-Anything style medical AI toolchain and lightweight agent loop."""


@cli.command("doctor")
@click.option("--shapekit-root", default="third_party/ShapeKit-main", show_default=True)
def doctor_cmd(shapekit_root: str):
    result = check_environment(resolve_path(shapekit_root))
    result["teacher_task_alignment"] = {"ai_model_backend": "registry-driven multi-model segmentation backend: TotalSegmentator/ePAI/VISTA3D/private templates/mock", "postprocess_tool": "ShapeKit as core anatomy-aware post-processing stage", "cli_standard": "CLI-Anything-style command interface with JSON output", "data_format": "PanTS/ShapeKit-like case folder: case_id/segmentations/*.nii.gz", "reasoning_layer": "RadThinking-style longitudinal trace"}
    emit(result)


@cli.command("presets")
def presets_cmd():
    emit({"roi_preset": "shapekit_abdomen", "totalsegmentator_roi_subset": SHAPEKIT_ABDOMEN_ROI, "shapekit_expected_organs": SHAPEKIT_EXPECTED_ORGANS})




@cli.command("build-registry")
@click.option("--checkpoint-map", required=True, help="Path to class_checkpoint_map.xlsx.")
@click.option("--output", "output_yaml", default="configs/model_registry.yaml", show_default=True, help="Output registry YAML path.")
@click.option("--parsed-csv", default="configs/class_checkpoint_map.parsed.csv", show_default=True, help="Output parsed CSV path.")
@click.option("--checkpoint-root", default="checkpoints", show_default=True, help="Local folder where private/public checkpoint folders are placed.")
def build_registry_cmd(checkpoint_map, output_yaml, parsed_csv, checkpoint_root):
    """Build a registry-driven organ→candidate-model map from class_checkpoint_map.xlsx."""
    emit(write_registry(resolve_path(checkpoint_map), resolve_path(output_yaml), resolve_path(parsed_csv) if parsed_csv else None, checkpoint_root=checkpoint_root))


@cli.command("registry-candidates")
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--organs", required=True, help="Comma-separated organs, e.g. pancreas,liver,aorta")
@click.option("--include-mock/--no-include-mock", default=False, show_default=True, help="Include mock_seg smoke-test backend in candidate lists.")
def registry_candidates_cmd(registry_path, organs, include_mock):
    """Show candidate model keys for target organs from model_registry.yaml."""
    organ_list = [x.strip() for x in organs.replace(";", ",").split(",") if x.strip()]
    reg = load_registry(resolve_path(registry_path))
    emit({"stage": "registry_candidates", "status": "success", "registry": str(resolve_path(registry_path)), "organs": organ_list, "candidates": candidate_models_for_organs(reg, organ_list, include_mock=include_mock)})


@cli.command("pants-info")
def pants_info_cmd():
    """Show official PanTS download/layout information without downloading data."""
    emit(pants_download_info())


@cli.command("pants-check")
@click.option("--pants-root", required=True, help="Path to PanTS repo root or PanTS/data folder.")
@click.option("--max-cases", default=10, type=int, show_default=True)
def pants_check_cmd(pants_root: str, max_cases: int):
    """Check whether real PanTS images/labels/reports have been downloaded."""
    emit(check_pants_dataset(resolve_path(pants_root), max_cases=max_cases))






@cli.command("pants-select-50")
@click.option("--pants-root", required=True, help="Path to downloaded PanTS repo root or data folder.")
@click.option("--output", "output_csv", default="data_manifest/case_list_50_tumor.csv", show_default=True)
@click.option("--split", default="train", type=click.Choice(["train", "test", "auto"]), show_default=True)
@click.option("--num-cases", default=50, type=int, show_default=True)
@click.option("--required-organs", default="pancreas,pancreatic_lesion,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava", show_default=True)
@click.option("--allow-empty-lesion", is_flag=True, default=False, help="Use only for smoke tests; formal selection requires non-empty pancreatic_lesion.")
def pants_select_50_cmd(pants_root: str, output_csv: str, split: str, num_cases: int, required_organs: str, allow_empty_lesion: bool):
    """Select 50 validated tumor-annotation PanTS cases; not random."""
    import subprocess, sys, json as _json
    script = resolve_path("scripts/select_pants50_cases.py")
    cmd = [sys.executable, str(script), "--pants-root", str(resolve_path(pants_root)), "--output", str(resolve_path(output_csv)), "--split", split, "--num-cases", str(num_cases), "--required-organs", required_organs]
    if allow_empty_lesion:
        cmd.append("--allow-empty-lesion")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        data = _json.loads(proc.stdout.strip() or "{}")
    except Exception:
        data = {"status": "failed", "stdout": proc.stdout, "stderr": proc.stderr}
    data["return_code"] = proc.returncode
    data["command"] = " ".join(cmd)
    if proc.stderr:
        data["stderr_tail"] = proc.stderr[-4000:]
    emit(data)


@cli.command("pants-download-50-plan")
@click.option("--case-list", default=None, help="Optional selected case-list CSV. If omitted, uses PanTS_00000001..00000050 plan.")
@click.option("--pants-root", default="third_party/PanTS-main", show_default=True)
def pants_download_50_plan_cmd(case_list: str | None, pants_root: str):
    """Print safe commands for selective PanTS50 download/extraction on Colab/Linux."""
    cmd = ["python", "scripts/download_pants50_selective.py", "--pants-root", pants_root, "--metadata", "--download-images", "--download-labels", "--yes"]
    if case_list:
        cmd.extend(["--case-list", case_list])
    emit({
        "stage": "pants_download_50_plan",
        "status": "info",
        "cannot_run_inside_this_chat": "The dataset blocks are too large for this sandbox; run this on Colab/Linux with enough storage.",
        "command": " ".join(cmd),
        "warning": "PanTSMini image blocks are ~28-34GB each and labels are large. The script extracts only selected cases but still downloads the required archive blocks.",
        "after_download": "python run_medai_cli.py --json pants-select-50 --pants-root third_party/PanTS-main --output data_manifest/case_list_50_tumor.csv"
    })


@cli.command("pants-find-case")
@click.option("--pants-root", required=True, help="Path to PanTS repo root or PanTS/data folder.")
@click.option("--case-id", required=True, help="PanTS case id, e.g., PanTS_00000001")
@click.option("--split", default="auto", type=click.Choice(["auto", "train", "test"]), show_default=True)
def pants_find_case_cmd(pants_root: str, case_id: str, split: str):
    """Locate a downloaded PanTS case without importing it."""
    emit(find_pants_case(resolve_path(pants_root), case_id, split))


@cli.command("pants-import-case")
@click.option("--pants-root", required=True, help="Path to PanTS repo root or PanTS/data folder.")
@click.option("--case-id", required=True, help="PanTS case id, e.g., PanTS_00000001")
@click.option("--output-root", required=True, help="Folder where the imported BDMAP/PanTS-style case folder will be created.")
@click.option("--split", default="auto", type=click.Choice(["auto", "train", "test"]), show_default=True)
@click.option("--patient-id", default=None, help="Optional patient folder name. Defaults to case id.")
@click.option("--scan-id", default=None, help="Optional scan id. Defaults to case id.")
@click.option("--copy-labels/--no-copy-labels", default=True, show_default=True)
def pants_import_case_cmd(pants_root: str, case_id: str, output_root: str, split: str, patient_id: str | None, scan_id: str | None, copy_labels: bool):
    """Import one downloaded PanTS case into the standard image.nii.gz + segmentations layout."""
    emit(import_pants_case(resolve_path(pants_root), case_id, resolve_path(output_root), split=split, patient_id=patient_id, scan_id=scan_id, copy_labels=copy_labels))


@cli.command("pants-import-files")
@click.option("--ct", "ct_path", required=True, help="Path to one real ct.nii.gz file.")
@click.option("--label-folder", default=None, help="Optional folder containing reference masks, e.g., segmentations/*.nii.gz")
@click.option("--output-root", required=True, help="Folder where the imported BDMAP/PanTS-style case folder will be created.")
@click.option("--patient-id", required=True, help="Patient/case folder name to create.")
@click.option("--scan-id", default=None, help="Optional scan id. Defaults to patient id.")
@click.option("--report", "report_path", default=None, help="Optional report.txt or source report path.")
@click.option("--copy-labels/--no-copy-labels", default=True, show_default=True)
def pants_import_files_cmd(ct_path: str, label_folder: str | None, output_root: str, patient_id: str, scan_id: str | None, report_path: str | None, copy_labels: bool):
    """Import any one PanTS-like CT case into the standard image.nii.gz + segmentations layout."""
    emit(import_pants_files(resolve_path(ct_path), resolve_path(label_folder) if label_folder else None, resolve_path(output_root), patient_id=patient_id, scan_id=scan_id, report_path=resolve_path(report_path) if report_path else None, copy_labels=copy_labels))


@cli.command("pants-eval-case")
@click.option("--pred-folder", required=True, help="Folder containing predicted masks, usually .../segmentations")
@click.option("--reference-label-folder", required=True, help="Folder containing reference masks from imported PanTS case")
@click.option("--organs", default=None, help="Comma-separated mask names/organs, e.g. pancreas,pancreatic_lesion,liver. Defaults to all reference masks.")
def pants_eval_case_cmd(pred_folder: str, reference_label_folder: str, organs: str | None):
    """Compute simple binary Dice for one case. This is a sanity check, not full PanTS benchmark."""
    organ_list = [x.strip() for x in organs.replace(';', ',').split(',') if x.strip()] if organs else None
    emit(evaluate_segmentation_folder(resolve_path(pred_folder), resolve_path(reference_label_folder), organ_list))


@cli.command("check-image")
@click.option("--image", required=True)
def check_image_cmd(image: str): emit(check_ct_image(resolve_path(image)))


@cli.command("check-folder")
@click.option("--input-folder", required=True)
def check_folder_cmd(input_folder: str): emit(check_case_folder(resolve_path(input_folder)))




@cli.command("model-inventory")
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--include-mock/--no-include-mock", default=False, show_default=True)
def model_inventory_cmd(registry_path, include_mock):
    """List all teacher-provided/referenced model families and M-step trainability."""
    reg = load_registry(resolve_path(registry_path))
    emit(model_inventory(reg, include_mock=include_mock))


@cli.command("route-models")
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--organs", required=True, help="Comma-separated target organs/tasks.")
@click.option("--target-config", default="configs/student_3d_prompt_target_organs.json", show_default=True, help="Target config used when --organs=student_373.")
def route_models_cmd(registry_path, organs, target_config):
    """Recommend primary E-step model, auxiliary models, and M-step target per organ."""
    organ_list = parse_organ_option(organs, target_config=target_config)
    reg = load_registry(resolve_path(registry_path))
    emit(recommend_primary_models_for_organs(reg, organ_list))


@cli.command("validate-373-target")
@click.option("--target-config", default="configs/student_3d_prompt_target_organs.json", show_default=True)
@click.option("--organs", default="student_373", show_default=True, help="Optional requested organ set to validate against the formal target.")
@click.option("--require-full-target/--allow-subset", default=True, show_default=True)
def validate_373_target_cmd(target_config, organs, require_full_target):
    """Validate the current formal 373-organ target source of truth."""
    organ_list = parse_organ_option(organs, target_config=target_config) if organs else None
    result = validate_formal_373_target_space(
        resolve_path(target_config),
        requested_organs=organ_list,
        require_full_target=require_full_target,
    )
    emit(result)


@cli.command("infer")
@click.option("--image", "image", default=None)
@click.option("--input", "image_alias", default=None, help="Alias for --image; kept for teacher-facing command examples.")
@click.option("--output-folder", "output_folder", default=None)
@click.option("--output", "output_alias", default=None, help="Alias for --output-folder; kept for teacher-facing command examples.")
@click.option("--case-id", default=None)
@click.option("--model", "model_key", default=None, help="Registry model key, e.g. atlasnet, totalsegmentator, epai_20250421, vista3d, mock_seg.")
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--backend", default="totalseg", type=click.Choice(["totalseg", "custom"]), show_default=True)
@click.option("--model-command", default=None)
@click.option("--fast/--no-fast", default=True, show_default=True)
@click.option("--task", default=None)
@click.option("--roi-preset", default="shapekit_abdomen", show_default=True)
@click.option("--roi-subset", default=None)
@click.option("--device", default=None)
@click.option("--adapt/--no-adapt", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def infer_cmd(image, image_alias, output_folder, output_alias, case_id, model_key, registry_path, backend, model_command, fast, task, roi_preset, roi_subset, device, adapt, dry_run):
    if image_alias: image = image_alias
    if output_alias: output_folder = output_alias
    if not image or not output_folder:
        fail({"status": "failed", "reason": "infer requires --image/--input and --output-folder/--output"})
    if model_key:
        infer_result = run_registered_model(resolve_path(image), resolve_path(output_folder), model_key, registry_path=resolve_path(registry_path), case_id=case_id, dry_run=dry_run, fast=fast, device=device)
    elif backend == "totalseg":
        infer_result = run_totalsegmentator(resolve_path(image), resolve_path(output_folder), case_id, fast, task, roi_preset, roi_subset, device, dry_run=dry_run)
    else:
        if not model_command: fail({"status": "failed", "reason": "--model-command is required for custom backend"})
        infer_result = run_custom_inference(resolve_path(image), resolve_path(output_folder), model_command, case_id, dry_run)
    adapter_result = normalize_totalseg_to_shapekit(infer_result["segmentation_output"]) if adapt and infer_result.get("status") == "success" and infer_result.get("segmentation_output") and not dry_run else None
    emit({"pipeline_stage": "infer", "infer": infer_result, "adapter": adapter_result})


@cli.command("segment-all")
@click.option("--image", "image", default=None)
@click.option("--input", "image_alias", default=None, help="Alias for --image.")
@click.option("--output-folder", "output_folder", default=None)
@click.option("--output", "output_alias", default=None, help="Alias for --output-folder.")
@click.option("--case-id", default=None)
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--organs", default=None, help="Comma-separated global organs. Defaults to configs/student_3d_prompt_target_organs.json (current 373 exact targets).")
@click.option("--device", default=None)
@click.option("--fast/--no-fast", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--global-label-space", default="configs/global_label_space.json", show_default=True)
@click.option("--alias-config", default="configs/model_label_aliases.json", show_default=True)
@click.option("--models", default="", help="Optional comma-separated model keys to restrict segment-all to a small validation subset.")
@click.option("--extra-models", default="", help="Optional comma-separated model keys to add to every requested organ for smoke/debug checks.")
def segment_all_cmd(image, image_alias, output_folder, output_alias, case_id, registry_path, organs, device, fast, dry_run, global_label_space, alias_config, models, extra_models):
    if image_alias:
        image = image_alias
    if output_alias:
        output_folder = output_alias
    if not image or not output_folder:
        fail({"status": "failed", "reason": "segment-all requires --image/--input and --output-folder/--output"})

    image_path = resolve_path(image)
    output_root = resolve_path(output_folder)
    resolved_case_id = case_id or (image_path.parent.name or image_path.stem)
    case_root = output_root / resolved_case_id
    (case_root / "per_model").mkdir(parents=True, exist_ok=True)

    organ_list = parse_organ_option(organs) if organs else None
    route_result = route_organs(organ_list)
    registry = load_registry(resolve_path(registry_path))
    model_filter = {x.strip() for x in models.replace(";", ",").split(",") if x.strip()}
    extra_model_keys = [x.strip() for x in extra_models.replace(";", ",").split(",") if x.strip()]
    if extra_model_keys:
        existing_model_keys = {
            item.get("model_key")
            for item in route_result.get("selected_models", [])
            if item.get("model_key")
        }
        for model_key in extra_model_keys:
            if model_key not in existing_model_keys:
                route_result.setdefault("selected_models", []).append({
                    "model_key": model_key,
                    "subtask": None,
                    "manual_extra_model": True,
                })
                route_result.setdefault("selected_model_keys", []).append(model_key)
                existing_model_keys.add(model_key)
        for organ in route_result.get("requested_organs", []):
            candidates = route_result.setdefault("ranked_candidates", {}).setdefault(organ, [])
            candidate_keys = {item.get("model_key") for item in candidates}
            for model_key in extra_model_keys:
                if model_key not in candidate_keys:
                    candidates.append({
                        "token": "manual_extra_model",
                        "model_key": model_key,
                        "subtask": None,
                        "enabled": True,
                        "reason": None,
                        "note": "Added through segment-all --extra-models for smoke/debug validation.",
                    })
    if model_filter:
        route_result["selected_models"] = [
            item for item in route_result.get("selected_models", [])
            if item.get("model_key") in model_filter
        ]
        route_result["selected_model_keys"] = [
            item.get("model_key")
            for item in route_result.get("selected_models", [])
            if item.get("model_key")
        ]
        for organ, candidates in (route_result.get("ranked_candidates", {}) or {}).items():
            route_result["ranked_candidates"][organ] = [
                item for item in candidates
                if item.get("model_key") in model_filter
            ]

    grouped_models: dict[str, dict[str, Any]] = {}
    for item in route_result.get("selected_models", []):
        model_key = item["model_key"]
        grouped = grouped_models.setdefault(model_key, {"model_key": model_key, "subtasks": []})
        if item.get("subtask"):
            grouped["subtasks"].append(item.get("subtask"))

    model_runs = []
    for model_key, grouped in grouped_models.items():
        (case_root / "per_model" / model_key).mkdir(parents=True, exist_ok=True)
        infer_result = run_registered_model(
            image_path,
            output_root,
            model_key,
            registry_path=resolve_path(registry_path),
            case_id=resolved_case_id,
            dry_run=dry_run,
            fast=fast,
            device=device,
            extra_context={
                "subtasks": grouped.get("subtasks", []),
                "requested_organs": route_result.get("requested_organs", []),
                "case_output_override": str(case_root),
                "segmentation_output_override": str(case_root / "per_model" / model_key / "segmentations"),
            },
        )
        model_runs.append({
            "model_key": model_key,
            "subtasks": grouped.get("subtasks", []),
            "infer": infer_result,
        })

    alias_report_path = None
    merge_result = None
    if not dry_run:
        alias_report_path = build_runtime_alias_report(case_root, route_result, registry, alias_config_path=alias_config)
        merge_result = merge_case_segmentations(
            case_root,
            route_result,
            registry,
            global_label_space_path=global_label_space,
            alias_config_path=alias_config,
        )

    emit({
        "pipeline_stage": "segment_all",
        "status": "dry_run" if dry_run else "success",
        "case_id": resolved_case_id,
        "case_root": str(case_root),
        "routing": route_result,
        "model_runs": model_runs,
        "runtime_alias_report": alias_report_path,
        "merge": merge_result,
    })


@cli.command("adapt")
@click.option("--segmentation-folder", required=True)
def adapt_cmd(segmentation_folder: str): emit(normalize_totalseg_to_shapekit(resolve_path(segmentation_folder)))


@cli.command("postprocess")
@click.option("--input-folder", "input_folder", default=None)
@click.option("--input", "input_alias", default=None, help="Alias for --input-folder.")
@click.option("--output-folder", "output_folder", default=None)
@click.option("--output", "output_alias", default=None, help="Alias for --output-folder.")
@click.option("--shapekit-root", default="third_party/ShapeKit-main", show_default=True)
@click.option("--log-folder", default=None)
@click.option("--cpu-count", default=2, show_default=True, type=int)
@click.option("--continue-prediction", is_flag=True, default=False)
@click.option("--auto-config/--no-auto-config", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def postprocess_cmd(input_folder, input_alias, output_folder, output_alias, shapekit_root, log_folder, cpu_count, continue_prediction, auto_config, dry_run):
    if input_alias: input_folder = input_alias
    if output_alias: output_folder = output_alias
    if not input_folder or not output_folder:
        fail({"status": "failed", "reason": "postprocess requires --input-folder/--input and --output-folder/--output"})
    out = resolve_path(output_folder); logs = resolve_path(log_folder) if log_folder else out / "logs"
    emit(run_shapekit(resolve_path(shapekit_root), resolve_path(input_folder), out, logs, cpu_count, continue_prediction, dry_run=dry_run, auto_config=auto_config))


@cli.command("run")
@click.option("--image", required=True)
@click.option("--output-folder", required=True)
@click.option("--case-id", default=None)
@click.option("--backend", default="totalseg", type=click.Choice(["totalseg", "custom"]), show_default=True)
@click.option("--model-command", default=None)
@click.option("--postprocess", default="shapekit", type=click.Choice(["none", "shapekit"]), show_default=True)
@click.option("--shapekit-root", default="third_party/ShapeKit-main", show_default=True)
@click.option("--fast/--no-fast", default=True, show_default=True)
@click.option("--task", default=None)
@click.option("--roi-preset", default="shapekit_abdomen", show_default=True)
@click.option("--roi-subset", default=None)
@click.option("--device", default=None)
@click.option("--cpu-count", default=2, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def run_cmd(image, output_folder, case_id, backend, model_command, postprocess, shapekit_root, fast, task, roi_preset, roi_subset, device, cpu_count, dry_run):
    out = resolve_path(output_folder); raw = out / "raw_predictions"; refined = out / "refined_predictions"; logs = out / "logs"
    if backend == "totalseg": infer_result = run_totalsegmentator(resolve_path(image), raw, case_id, fast, task, roi_preset, roi_subset, device, dry_run=dry_run)
    else:
        if not model_command: fail({"status": "failed", "reason": "--model-command is required for custom backend"})
        infer_result = run_custom_inference(resolve_path(image), raw, model_command, case_id, dry_run)
    adapter_result = normalize_totalseg_to_shapekit(infer_result["segmentation_output"]) if infer_result.get("status") == "success" and not dry_run else None
    post_result = None; summary = None
    if postprocess == "shapekit" and infer_result.get("status") == "success":
        post_result = run_shapekit(resolve_path(shapekit_root), raw, refined, logs, cpu_count, dry_run=dry_run)
        if not dry_run: summary = summarize_segmentation_folder(refined if post_result.get("status") == "success" else raw)
    elif infer_result.get("status") == "success" and not dry_run: summary = summarize_segmentation_folder(raw)
    result = {"pipeline": "AI infer + adapter + optional ShapeKit", "teacher_alignment": {"AI_model_infer": "TotalSegmentator/custom", "ShapeKit": "post-processing, not model", "CLI_Anything": "single JSON CLI", "PanTS": "case_id/segmentations layout"}, "infer": infer_result, "adapter": adapter_result, "postprocess": post_result, "final_summary": summary}
    if not dry_run: result["summary_json"] = write_run_summary(out, result)
    emit(result)


@cli.command("summary")
@click.option("--folder", required=True)
def summary_cmd(folder: str): emit(summarize_segmentation_folder(resolve_path(folder)))


@cli.command("radthinking-check")
@click.option("--patient-folder", required=True)
def radthinking_check_cmd(patient_folder: str): emit(check_radthinking_patient(resolve_path(patient_folder)))


@cli.command("radthinking-run-patient")
@click.option("--patient-folder", required=True)
@click.option("--output-folder", required=True)
@click.option("--backend", default="totalseg", type=click.Choice(["totalseg", "custom"]), show_default=True)
@click.option("--model-command", default=None)
@click.option("--postprocess", default="shapekit", type=click.Choice(["none", "shapekit"]), show_default=True)
@click.option("--shapekit-root", default="third_party/ShapeKit-main", show_default=True)
@click.option("--fast/--no-fast", default=True, show_default=True)
@click.option("--task", default=None)
@click.option("--roi-preset", default="shapekit_abdomen", show_default=True)
@click.option("--roi-subset", default=None)
@click.option("--device", default=None)
@click.option("--cpu-count", default=2, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def radthinking_run_patient_cmd(patient_folder, output_folder, backend, model_command, postprocess, shapekit_root, fast, task, roi_preset, roi_subset, device, cpu_count, dry_run):
    patient_root = resolve_path(patient_folder); out_root = resolve_path(output_folder); scans = discover_radthinking_scans(patient_root); results = []
    for scan in scans:
        scan_out = out_root / scan.scan_id; raw = scan_out / "raw_predictions"; refined = scan_out / "refined_predictions"; logs = scan_out / "logs"; case_id = f"{patient_root.name}_{scan.scan_id}"
        if not scan.ct_image: results.append({"scan_id": scan.scan_id, "status": "skipped", "reason": "ct image missing"}); continue
        if backend == "totalseg": infer_result = run_totalsegmentator(scan.ct_image, raw, case_id, fast, task, roi_preset, roi_subset, device, dry_run=dry_run)
        else:
            if not model_command: fail({"status": "failed", "reason": "--model-command is required for custom backend"})
            infer_result = run_custom_inference(scan.ct_image, raw, model_command, case_id, dry_run)
        adapter = normalize_totalseg_to_shapekit(infer_result["segmentation_output"]) if infer_result.get("status") == "success" and not dry_run else None
        post = run_shapekit(resolve_path(shapekit_root), raw, refined, logs, cpu_count, dry_run=dry_run) if postprocess == "shapekit" and infer_result.get("status") == "success" else None
        results.append({"scan_id": scan.scan_id, "case_id": case_id, "ct_image": str(scan.ct_image), "infer": infer_result, "adapter": adapter, "postprocess": post})
    result = {"pipeline": "RadThinking patient batch: per-scan AI inference + optional ShapeKit", "status": "success" if results else "warning", "patient_folder": str(patient_root), "output_folder": str(out_root), "num_scans": len(scans), "num_attempted": len(results), "not_agent_loop": "deterministic patient workflow", "results": results}
    if not dry_run: result["summary_json"] = write_run_summary(out_root, result)
    emit(result)


@cli.command("trace-template")
@click.option("--case-id", required=True)
@click.option("--ct-image", default=None)
@click.option("--segmentation-folder", default=None)
@click.option("--report-path", default=None)
@click.option("--prior-case-id", default=None)
def trace_template_cmd(case_id, ct_image, segmentation_folder, report_path, prior_case_id): emit(create_radthinking_trace_template(case_id, str(resolve_path(ct_image)) if ct_image else None, str(resolve_path(segmentation_folder)) if segmentation_folder else None, str(resolve_path(report_path)) if report_path else None, prior_case_id))


@cli.command("trace-observation")
@click.option("--ct-image", required=True)
@click.option("--mask", "mask_path", required=True)
@click.option("--organ", default=None)
def trace_observation_cmd(ct_image, mask_path, organ): emit(extract_observation(resolve_path(ct_image), resolve_path(mask_path), organ))


@cli.command("trace-temporal")
@click.option("--previous-mask", default=None)
@click.option("--current-mask", default=None)
@click.option("--organ", default=None)
def trace_temporal_cmd(previous_mask, current_mask, organ): emit(compare_temporal_masks(resolve_path(previous_mask) if previous_mask else None, resolve_path(current_mask) if current_mask else None, organ))


@cli.command("trace-context")
@click.option("--report", "report_path", default=None)
@click.option("--clinical", "clinical_path", default=None)
@click.option("--organ", default=None)
def trace_context_cmd(report_path, clinical_path, organ): emit(parse_report_sections(resolve_path(report_path) if report_path else None, resolve_path(clinical_path) if clinical_path else None, organ))


@cli.command("trace-build")
@click.option("--patient-folder", default=None)
@click.option("--scan-id", default=None)
@click.option("--ct-image", default=None)
@click.option("--current-mask", default=None)
@click.option("--previous-mask", default=None)
@click.option("--organ", default=None)
@click.option("--report", "report_path", default=None)
@click.option("--clinical", "clinical_path", default=None)
@click.option("--pathology", "pathology_path", default=None)
@click.option("--output-json", default=None)
def trace_build_cmd(patient_folder, scan_id, ct_image, current_mask, previous_mask, organ, report_path, clinical_path, pathology_path, output_json): emit(build_reasoning_trace(resolve_path(patient_folder) if patient_folder else None, scan_id, resolve_path(ct_image) if ct_image else None, resolve_path(current_mask) if current_mask else None, resolve_path(previous_mask) if previous_mask else None, organ, resolve_path(report_path) if report_path else None, resolve_path(clinical_path) if clinical_path else None, resolve_path(pathology_path) if pathology_path else None, resolve_path(output_json) if output_json else None))


@cli.command("trace-patient")
@click.option("--patient-folder", required=True)
@click.option("--output-folder", required=True)
@click.option("--organ", required=True)
@click.option("--output-json", default=None)
def trace_patient_cmd(patient_folder, output_folder, organ, output_json): emit(build_patient_traces_from_outputs(resolve_path(patient_folder), resolve_path(output_folder), organ, resolve_path(output_json) if output_json else None))


@cli.command("agent-loop")
@click.option("--patient-folder", required=True)
@click.option("--output-folder", required=True)
@click.option("--backend", default="totalseg", type=click.Choice(["totalseg", "custom"]), show_default=True)
@click.option("--model-command", default=None)
@click.option("--postprocess", default="shapekit", type=click.Choice(["none", "shapekit"]), show_default=True)
@click.option("--shapekit-root", default="third_party/ShapeKit-main", show_default=True)
@click.option("--fast/--no-fast", default=True, show_default=True)
@click.option("--task", default=None)
@click.option("--roi-preset", default="shapekit_abdomen", show_default=True)
@click.option("--roi-subset", default=None)
@click.option("--device", default=None)
@click.option("--cpu-count", default=2, type=int, show_default=True)
@click.option("--organ", default="liver", show_default=True)
@click.option("--expected-organs", default="liver,pancreas,aorta,postcava", show_default=True)
@click.option("--enable-trace/--no-enable-trace", default=True, show_default=True)
@click.option("--enable-qc/--no-enable-qc", default=True, show_default=True)
@click.option("--retry-failed/--no-retry-failed", default=True, show_default=True)
@click.option("--max-iterations", default=1, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--max-scans", default=None, type=int, help="Optional smoke-test limit for processing only the first N scans.")
@click.option("--shapekit-timeout-sec", default=120, type=int, show_default=True, help="Maximum seconds allowed for each ShapeKit call before fallback/review.")
@click.option("--enable-vlm-refinement", is_flag=True, default=False, help="Enable real VLM calls during iterative refinement (max_iterations>=2). Default: stub.")
@click.option("--vlm-backend-refinement", default="stub", type=click.Choice(["ollama", "vllm", "stub"]), show_default=True)
@click.option("--vlm-model-refinement", default="qwen2.5vl:7b", show_default=True)
def agent_loop_cmd(patient_folder, output_folder, backend, model_command, postprocess, shapekit_root, fast, task, roi_preset, roi_subset, device, cpu_count, organ, expected_organs, enable_trace, enable_qc, retry_failed, max_iterations, dry_run, max_scans, shapekit_timeout_sec, enable_vlm_refinement, vlm_backend_refinement, vlm_model_refinement):
    organs = [x.strip() for x in expected_organs.replace(";", ",").split(",") if x.strip()]
    if backend == "custom" and not model_command: fail({"status": "failed", "reason": "--model-command is required for custom backend"})
    emit(run_agent_loop(resolve_path(patient_folder), resolve_path(output_folder), backend, model_command, postprocess, resolve_path(shapekit_root), fast, task, roi_preset, roi_subset, device, cpu_count, organ, organs, enable_trace, enable_qc, retry_failed, max_iterations, dry_run, max_scans, shapekit_timeout_sec, enable_vlm_refinement, vlm_backend_refinement, vlm_model_refinement))


@cli.command("label-verify")
@click.option("--annotation-folder", required=True, help="Folder with prior/current pseudo labels. Not treated as expert ground truth.")
@click.option("--prediction-folder", required=True, help="Folder with model predictions.")
@click.option("--organs", default="pancreas,liver,spleen,kidney_left,kidney_right,aorta", show_default=True)
@click.option("--dsc-replace-threshold", default=0.0, type=float, show_default=True, help="DSC at or below this → auto_replace_candidate.")
@click.option("--dsc-vlm-threshold", default=0.5, type=float, show_default=True, help="DSC below this → send_to_vlm_label_expert.")
@click.option("--dsc-accept-threshold", default=0.8, type=float, show_default=True, help="DSC at/above this → accept; 0.5-0.8 → uncertain.")
def label_verify_cmd(annotation_folder, prediction_folder, organs, dsc_replace_threshold, dsc_vlm_threshold, dsc_accept_threshold):
    """Compare pseudo references vs model predictions using DSC consistency thresholds."""
    organ_list = [x.strip() for x in organs.replace(";", ",").split(",") if x.strip()]
    emit(verify_case(resolve_path(annotation_folder), resolve_path(prediction_folder),
                     organ_list, dsc_replace_threshold, dsc_vlm_threshold, dsc_accept_threshold))


@cli.command("projection-build")
@click.option("--ct-image", required=True, help="Path to CT image (.nii.gz).")
@click.option("--annotation-a", default=None, help="Candidate A mask (.nii.gz).")
@click.option("--annotation-b", default=None, help="Candidate B mask (.nii.gz).")
@click.option("--organ", required=True, help="Organ name, e.g. pancreas.")
@click.option("--output-folder", required=True, help="Where to save projection PNGs.")
@click.option("--views", default="axial,coronal", show_default=True, help="Used only by --projection-backend slice/auto fallback.")
@click.option("--projection-backend", default="labelcritic", type=click.Choice(["labelcritic", "auto", "slice"]), show_default=True, help="Use LabelCritic 3D-to-2D projection by default; slice is only a lightweight fallback.")
@click.option("--labelcritic-root", default="third_party/LabelCritic-main", show_default=True)
@click.option("--axis", default=1, type=int, show_default=True, help="LabelCritic projection axis.")
@click.option("--device", default="cpu", show_default=True, help="LabelCritic projection device.")
@click.option("--num-processes", default=2, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--strict-alignment/--no-strict-alignment", default=False, show_default=True, help="Fail if mask/CT affine or shape mismatch detected for slice fallback.")
def projection_build_cmd(ct_image, annotation_a, annotation_b, organ, output_folder, views, projection_backend, labelcritic_root, axis, device, num_processes, dry_run, strict_alignment):
    """Build 2D CT+mask projection PNGs for manual inspection or VLM input.

    By default this uses LabelCritic's ProjectDatasetFlex_single.py/projection.py
    rather than a naive average projection.
    """
    view_list = [v.strip() for v in views.split(",") if v.strip()]
    emit(build_projection(
        resolve_path(ct_image),
        resolve_path(annotation_a) if annotation_a else None,
        resolve_path(annotation_b) if annotation_b else None,
        resolve_path(output_folder),
        organ=organ,
        views=view_list,
        strict_alignment=strict_alignment,
        projection_backend=projection_backend,
        labelcritic_root=resolve_path(labelcritic_root),
        axis=axis,
        device=device,
        num_processes=num_processes,
        dry_run=dry_run,
    ))


@cli.command("vlm-label-expert")
@click.option("--ct-image", required=True, help="Path to CT image (.nii.gz).")
@click.option("--annotation-a", default=None, help="Current annotation mask (.nii.gz).")
@click.option("--annotation-b", default=None, help="Model prediction mask (.nii.gz).")
@click.option("--organ", required=True, help="Organ name, e.g. pancreas.")
@click.option("--output-folder", required=True, help="Where to save projections and results.")
@click.option("--vlm-backend", default="ollama", type=click.Choice(["ollama", "vllm", "stub"]), show_default=True)
@click.option("--vlm-model", default="qwen2.5vl:7b", show_default=True)
@click.option("--case-id", default=None)
@click.option("--dsc-replace-threshold", default=0.0, type=float, show_default=True)
@click.option("--dsc-vlm-threshold", default=0.5, type=float, show_default=True)
@click.option("--strict-alignment/--no-strict-alignment", default=False, show_default=True, help="Fail if mask/CT affine or shape mismatch detected.")
@click.option("--vllm-base-url", default=os.getenv("LABELCRITIC_API_BASE", "http://localhost:8000/v1"), show_default=True, help="OpenAI-compatible VLM endpoint for --vlm-backend vllm.")
def vlm_label_expert_cmd(ct_image, annotation_a, annotation_b, organ, output_folder,
                          vlm_backend, vlm_model, case_id, dsc_replace_threshold, dsc_vlm_threshold,
                          strict_alignment, vllm_base_url):
    """VLM Label Expert: project 3D masks to 2D, compare with VLM, output annotation decision."""
    emit(run_vlm_label_expert(
        resolve_path(ct_image),
        resolve_path(annotation_a) if annotation_a else None,
        resolve_path(annotation_b) if annotation_b else None,
        organ, resolve_path(output_folder),
        vlm_backend, vlm_model, case_id,
        dsc_replace_threshold, dsc_vlm_threshold,
        strict_alignment=strict_alignment,
        vllm_base_url=vllm_base_url,
    ))


@cli.command("verify")
@click.option("--prediction", "prediction_folder", required=True, help="Folder containing predicted masks.")
@click.option("--reference", "annotation_folder", required=True, help="Folder containing prior/current pseudo-reference masks.")
@click.option("--organs", default="pancreas,liver,spleen,kidney_left,kidney_right,aorta", show_default=True)
@click.option("--dsc-replace-threshold", default=0.0, type=float, show_default=True)
@click.option("--dsc-vlm-threshold", default=0.5, type=float, show_default=True)
@click.option("--dsc-accept-threshold", default=0.8, type=float, show_default=True)
@click.option("--output", "output_path", default=None, help="Optional JSON output file.")
def verify_cmd(prediction_folder, annotation_folder, organs, dsc_replace_threshold, dsc_vlm_threshold, dsc_accept_threshold, output_path):
    """Teacher-facing alias for label-verify: pseudo-consistency DICE/DSC gate."""
    organ_list = [x.strip() for x in organs.replace(";", ",").split(",") if x.strip()]
    result = verify_case(resolve_path(annotation_folder), resolve_path(prediction_folder), organ_list, dsc_replace_threshold, dsc_vlm_threshold, dsc_accept_threshold)
    if output_path:
        from .core.json_utils import write_json
        result["saved_to"] = write_json(resolve_path(output_path), result)
    emit(result)


@cli.command("critic")
@click.option("--ct", "ct_image", required=True, help="Path to CT image (.nii.gz).")
@click.option("--mask-a", required=True, help="Candidate A mask file or folder.")
@click.option("--mask-b", required=True, help="Candidate B mask file or folder.")
@click.option("--organ", required=True)
@click.option("--output", "output_json", required=True, help="Decision JSON path.")
@click.option("--labelcritic-root", default="third_party/LabelCritic-main", show_default=True)
@click.option("--backend", default="labelcritic", type=click.Choice(["stub", "labelcritic"]), show_default=True)
@click.option("--base-url", default="http://localhost", show_default=True, help="LabelCritic/vLLM host WITHOUT /v1; port is supplied by --port.")
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--no-dice-check", is_flag=True, default=False, help="Diagnostic only: force VLM comparison even when projections are similar.")
@click.option("--no-dual-confirmation", is_flag=True, default=False, help="Diagnostic only: disable dual confirmation prompt.")
@click.option("--simple-prompt-ablation", is_flag=True, default=False, help="Diagnostic only: use simpler prompt wording.")
@click.option("--conservative-dual", is_flag=True, default=False, help="Diagnostic only: require stricter dual agreement.")
@click.option("--skip-organ-presence-gate", is_flag=True, default=False, help="Diagnostic only: bypass organ-presence gate.")
@click.option("--strict-choice-prompt", is_flag=True, default=False, help="Diagnostic only: force overlay 1/overlay 2/tie answer format.")
def critic_cmd(ct_image, mask_a, mask_b, organ, output_json, labelcritic_root, backend, base_url, port, dry_run,
               no_dice_check, no_dual_confirmation, simple_prompt_ablation, conservative_dual,
               skip_organ_presence_gate, strict_choice_prompt):
    """LabelCritic wrapper: compare two candidate masks and write a decision JSON."""
    emit(run_labelcritic_compare(
        resolve_path(ct_image), resolve_path(mask_a), resolve_path(mask_b),
        organ, resolve_path(output_json), resolve_path(labelcritic_root),
        backend=backend, base_url=base_url, port=port, dry_run=dry_run,
        no_dice_check=no_dice_check,
        no_dual_confirmation=no_dual_confirmation,
        simple_prompt_ablation=simple_prompt_ablation,
        conservative_dual=conservative_dual,
        skip_organ_presence_gate=skip_organ_presence_gate,
        strict_choice_prompt=strict_choice_prompt,
    ))


@cli.command("run-loop")
@click.option("--case-list", required=True, help="CSV with case_id,ct_path,annotation_folder[,report_path,clinical_path,pathology_path].")
@click.option("--models", default="epai_20250421,vsmtrans", show_default=True, help="Comma-separated registry model keys.")
@click.option("--organs", default="student_373", show_default=True, help="Comma-separated organs, or student_373 to load the current 373 exact targets.")
@click.option("--target-config", default="configs/student_3d_prompt_target_organs.json", show_default=True, help="Target config used when --organs=student_373.")
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--output", "output_folder", required=True)
@click.option("--checkpoint-map-models/--no-checkpoint-map-models", default=False, show_default=True, help="Also include all organ-specific candidates from registry.")
@click.option("--shapekit-root", default="third_party/ShapeKit-main", show_default=True)
@click.option("--enable-shapekit/--no-enable-shapekit", default=True, show_default=True)
@click.option("--debug-allow-no-shapekit", is_flag=True, default=False, help="Allow disabling ShapeKit for smoke/debug runs only.")
@click.option("--enable-critic/--no-enable-critic", default=True, show_default=True)
@click.option("--critic-backend", default="labelcritic", type=click.Choice(["stub", "labelcritic"]), show_default=True)
@click.option("--critic-base-url", default=os.getenv("LABELCRITIC_BASE_URL", "http://localhost"), show_default=True, help="LabelCritic/vLLM host WITHOUT /v1; port is supplied by --critic-port.")
@click.option("--critic-port", default=int(os.getenv("LABELCRITIC_PORT", "8000")), type=int, show_default=True)
@click.option("--critic-vlm-model", default=os.getenv("LABELCRITIC_MODEL_ID"), help="Explicit LabelCritic VLM id. Formal runs use Qwen/Qwen2-VL-72B-Instruct-AWQ; if omitted, the first model from /v1/models is used.")
@click.option("--labelcritic-no-dice-check", is_flag=True, default=False, help="Diagnostic only: force VLM comparison even when projections are similar.")
@click.option("--labelcritic-no-dual-confirmation", is_flag=True, default=False, help="Diagnostic only: disable dual confirmation prompt.")
@click.option("--labelcritic-simple-prompt-ablation", is_flag=True, default=False, help="Diagnostic only: use simpler prompt wording.")
@click.option("--labelcritic-conservative-dual", is_flag=True, default=False, help="Diagnostic only: require stricter dual agreement.")
@click.option("--labelcritic-skip-organ-presence-gate", is_flag=True, default=False, help="Diagnostic only: bypass organ-presence gate.")
@click.option("--labelcritic-strict-choice-prompt", is_flag=True, default=False, help="Diagnostic only: force overlay 1/overlay 2/tie answer format.")
@click.option("--vlm-threshold", default=0.5, type=float, show_default=True)
@click.option("--accept-threshold", default=0.8, type=float, show_default=True)
@click.option("--device", default=None)
@click.option("--timeout-sec", default=1800, type=int, show_default=True)
@click.option("--perf-tracker-path", default=None, help="Path to organ_model_performance.json for explore/exploit switching.")
@click.option("--teacher-inference-mode", type=click.Choice(["full_volume", "hierarchical_roi"]), default="hierarchical_roi", show_default=True)
@click.option("--roi-margin-mm", type=float, default=20.0, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--strict-delivery-targets", is_flag=True, default=False, help="Fail strict delivery runs when requested teachers/outputs are not actually produced.")
@click.option("--log-file", default=os.getenv("MEDAI_LOG_FILE"), help="Optional fixed log file; run-loop progress and final JSON are tee'd here.")
def run_loop_cmd(case_list, models, organs, target_config, registry_path, output_folder, checkpoint_map_models, shapekit_root, enable_shapekit, debug_allow_no_shapekit, enable_critic, critic_backend, critic_base_url, critic_port, critic_vlm_model, labelcritic_no_dice_check, labelcritic_no_dual_confirmation, labelcritic_simple_prompt_ablation, labelcritic_conservative_dual, labelcritic_skip_organ_presence_gate, labelcritic_strict_choice_prompt, vlm_threshold, accept_threshold, device, timeout_sec, perf_tracker_path, teacher_inference_mode, roi_margin_mm, dry_run, strict_delivery_targets, log_file):
    """End-to-end multi-model annotation refinement loop for the 50-case debug set."""
    if not enable_shapekit and not dry_run and not debug_allow_no_shapekit:
        fail({
            "status": "failed",
            "reason": (
                "Formal non-dry-run E-step requires ShapeKit. Use --dry-run or "
                "--debug-allow-no-shapekit only for smoke/debug checks."
            ),
            "teacher_meeting_requirement": "All selected outputs pass through ShapeKit in formal runs.",
        })
    model_list = [x.strip() for x in models.replace(";", ",").split(",") if x.strip()]
    organ_list = parse_organ_option(organs, target_config=target_config)
    def _run() -> dict[str, Any]:
        return run_multimodel_annotation_loop(
            resolve_path(case_list), resolve_path(output_folder), model_list,
            organ_list, resolve_path(registry_path), checkpoint_map_models,
            resolve_path(shapekit_root), enable_shapekit, enable_critic,
            critic_backend, critic_base_url, critic_port, vlm_threshold,
            accept_threshold, dry_run, timeout_sec, device,
            perf_tracker_path=resolve_path(perf_tracker_path) if perf_tracker_path else None,
            labelcritic_options={
                "no_dice_check": labelcritic_no_dice_check,
                "no_dual_confirmation": labelcritic_no_dual_confirmation,
                "simple_prompt_ablation": labelcritic_simple_prompt_ablation,
                "conservative_dual": labelcritic_conservative_dual,
                "skip_organ_presence_gate": labelcritic_skip_organ_presence_gate,
                "strict_choice_prompt": labelcritic_strict_choice_prompt,
            },
            teacher_inference_mode=teacher_inference_mode,
            roi_margin_mm=roi_margin_mm,
            vlm_model=critic_vlm_model,
            strict_delivery_targets=strict_delivery_targets,
        )

    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[medai run-loop log start]\n")
            with redirect_stdout(_Tee(sys.stdout, handle)), redirect_stderr(_Tee(sys.stderr, handle)):
                result = _run()
            handle.write(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False) + "\n")
        emit(result)
    else:
        emit(_run())


@cli.command("mstep-train")
@click.option("--training-manifest", required=True, help="training_manifest.json or .csv from run-loop output.")
@click.option("--output-folder", required=True)
@click.option("--ct-source-root", default=None, help="Root folder with CT images for dataset preparation.")
@click.option("--dataset-id", default=999, type=int, show_default=True)
@click.option("--max-epochs", default=5, type=int, show_default=True)
@click.option("--pretrained-weights", default=None, help="Path to pretrained checkpoint for continual training.")
@click.option("--dry-run", is_flag=True, default=False)
def mstep_train_cmd(training_manifest, output_folder, ct_source_root, dataset_id, max_epochs, pretrained_weights, dry_run):
    """M-step: prepare nnUNet v2 dataset and launch fine-tuning from updated annotations."""
    emit(run_mstep_nnunet_training(
        resolve_path(training_manifest), resolve_path(output_folder),
        dataset_id=dataset_id, ct_source_root=resolve_path(ct_source_root) if ct_source_root else None,
        max_epochs=max_epochs, pretrained_weights=pretrained_weights, dry_run=dry_run,
    ))




@cli.command("mstep-update")
@click.option("--training-manifest", required=True, help="training_manifest.json or .csv from run-loop output.")
@click.option("--output-folder", required=True)
@click.option("--target-model", required=True, help="Selected E-step primary model to update, e.g. epai_20250421 or cads.")
@click.option("--registry", "registry_path", default="configs/model_registry.yaml", show_default=True)
@click.option("--ct-source-root", default=None, help="Root folder with CT images for dataset preparation.")
@click.option("--dataset-id", default=None, type=int, help="Optional override for nnUNet dataset id.")
@click.option("--max-epochs", default=5, type=int, show_default=True)
@click.option("--pretrained-weights", default=None, help="Specific .pth checkpoint file for true fine-tuning initialization.")
@click.option("--timeout-sec", default=21600, type=int, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def mstep_update_cmd(training_manifest, output_folder, target_model, registry_path, ct_source_root, dataset_id, max_epochs, pretrained_weights, timeout_sec, dry_run):
    """Selected-model-aware M-step: update the chosen E-step model family when trainable."""
    emit(run_model_specific_mstep_update(
        training_manifest=resolve_path(training_manifest),
        output_folder=resolve_path(output_folder),
        target_model=target_model,
        registry_path=resolve_path(registry_path),
        ct_source_root=resolve_path(ct_source_root) if ct_source_root else None,
        dataset_id=dataset_id,
        max_epochs=max_epochs,
        pretrained_weights=pretrained_weights,
        dry_run=dry_run,
        timeout_sec=timeout_sec,
    ))


@cli.command("report-supervision")
@click.option("--report", "report_path", required=True, help="Path to radiology report (.txt/.pdf).")
@click.option("--tumor-mask", required=True, help="Path to AI-predicted tumor mask (.nii.gz).")
@click.option("--organ", default="pancreas", show_default=True)
@click.option("--clinical", "clinical_path", default=None)
def report_supervision_cmd(report_path, tumor_mask, organ, clinical_path):
    """Compare AI tumor mask against radiology report (replaces ROC analysis)."""
    emit(verify_tumor_with_report(
        resolve_path(report_path), resolve_path(tumor_mask), organ,
        resolve_path(clinical_path) if clinical_path else None,
    ))




@cli.command("report-supervision-batch")
@click.option("--case-list", required=True, help="CSV with case_id,annotation_folder,report_path[,clinical_path].")
@click.option("--output-jsonl", required=True, help="Where to write report-supervision decisions.")
@click.option("--organ", default="pancreas", show_default=True)
@click.option("--tumor-mask-name", default="pancreatic_lesion", show_default=True)
def report_supervision_batch_cmd(case_list, output_jsonl, organ, tumor_mask_name):
    """Batch report-supervision across the selected PanTS 50-case list."""
    import csv
    rows = []
    with open(resolve_path(case_list), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any((v or "").strip() for v in row.values()):
                rows.append(row)
    emit(batch_verify_with_reports(rows, resolve_path(output_jsonl), organ=organ, tumor_mask_name=tumor_mask_name))

@cli.command("build-samples")
@click.option("--run-output", required=True, help="run-loop output folder.")
@click.option("--output-jsonl", default=None, help="Output JSONL file for per-case samples.")
@click.option("--organs", default="pancreas,liver,spleen,kidney_left,kidney_right,colon,duodenum,stomach,aorta,postcava", show_default=True)
def build_samples_cmd(run_output, output_jsonl, organs):
    """Build per-case aggregated dataset samples in the teacher-expected format."""
    organ_list = [x.strip() for x in organs.replace(";", ",").split(",") if x.strip()]
    emit(build_case_samples(
        resolve_path(run_output), organ_list,
        resolve_path(output_jsonl) if output_jsonl else None,
    ))


@cli.command("convergence-table")
@click.option("--round-csvs", required=True, help="Comma-separated paths to round_metrics.csv files from successive loops.")
def convergence_table_cmd(round_csvs):
    """Build the teacher-expected loop convergence table showing DICE improvement across rounds."""
    paths = [resolve_path(p.strip()) for p in round_csvs.split(",") if p.strip()]
    emit(build_convergence_table(paths))


@cli.command("itksnap-review")
@click.option("--review-queue", required=True, help="Path to review_queue.jsonl from run-loop.")
@click.option("--ct-root", default=None, help="Root folder for CT images.")
@click.option("--annotation-root", default=None, help="Root folder for annotation versions.")
@click.option("--output-script", default=None, help="Output .sh script with ITK-SNAP commands.")
@click.option("--max-cases", default=10, type=int, show_default=True)
def itksnap_review_cmd(review_queue, ct_root, annotation_root, output_script, max_cases):
    """Generate ITK-SNAP commands for manual review of uncertain cases."""
    emit(generate_itksnap_commands(
        resolve_path(review_queue),
        resolve_path(ct_root) if ct_root else None,
        resolve_path(annotation_root) if annotation_root else None,
        resolve_path(output_script) if output_script else None,
        max_cases,
    ))


@cli.command("vista3d-segment")
@click.option("--ct-image", required=True, help="CT NIfTI 文件路径")
@click.option("--prompts", required=True, help="逗号分隔的器官名或自然语言提示，如 'pancreas,liver' 或 'segment pancreas,segment liver'")
@click.option("--output-folder", required=True, help="输出 mask 目录")
@click.option("--vista3d-root", default="checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master", show_default=True)
@click.option("--model-path", default=None, help="VISTA3D checkpoint 路径，默认使用 vista3d_root/models/model.pt")
@click.option("--device", default="cuda", show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def vista3d_segment_cmd(ct_image, prompts, output_folder, vista3d_root, model_path, device, dry_run):
    """Legacy/reference VISTA3D teacher-style prompt segmentation.

    This is not the current 373-organ 3D prompt student mainline. Use
    voxtell-student-segment for the teacher-approved student path.
    """
    from .core.vista3d_student import VISTA3DStudent
    student = VISTA3DStudent(
        vista3d_root=resolve_path(vista3d_root),
        model_path=resolve_path(model_path) if model_path else None,
        device=device,
    )
    result = student.segment(
        ct_image=resolve_path(ct_image),
        prompts=[p.strip() for p in prompts.split(",")],
        output_dir=resolve_path(output_folder),
        dry_run=dry_run,
    )
    result["legacy_warning"] = (
        "VISTA3D is kept as a teacher/reference/legacy component and does not "
        "define the current 373-organ student target space."
    )
    emit(result)


@cli.command("vista3d-finetune")
@click.option("--pseudo-label-dir", required=True, help="伪标签目录（每个 case 一个子目录）")
@click.option("--ct-dir", required=True, help="CT 文件根目录")
@click.option("--target-organs", required=True, help="逗号分隔的目标器官名")
@click.option("--output-folder", required=True, help="微调后 checkpoint 输出目录")
@click.option("--vista3d-root", default="checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master", show_default=True)
@click.option("--model-path", default=None)
@click.option("--learning-rate", default=5e-5, show_default=True, type=float)
@click.option("--max-epochs", default=50, show_default=True, type=int)
@click.option("--freeze-backbone", is_flag=True, default=True, help="冻结 SwinUNETR backbone，只更新 point_head")
@click.option("--global-consolidation", is_flag=True, default=False, help="全局整合模式：小学习率部分解冻 backbone")
@click.option("--device", default="cuda", show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def vista3d_finetune_cmd(pseudo_label_dir, ct_dir, target_organs, output_folder,
                          vista3d_root, model_path, learning_rate, max_epochs,
                          freeze_backbone, global_consolidation, device, dry_run):
    """Legacy/reference VISTA3D continual fine-tuning helper.

    This command is kept for historical reproduction only. It is not the
    current M-step for the teacher-approved 373-organ 3D prompt student.
    """
    from .core.vista3d_student import VISTA3DStudent
    student = VISTA3DStudent(
        vista3d_root=resolve_path(vista3d_root),
        model_path=resolve_path(model_path) if model_path else None,
        device=device,
    )
    result = student.continual_finetune(
        pseudo_label_dir=resolve_path(pseudo_label_dir),
        ct_dir=resolve_path(ct_dir),
        target_organs=[o.strip() for o in target_organs.split(",")],
        output_dir=resolve_path(output_folder),
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        freeze_backbone=freeze_backbone,
        global_consolidation=global_consolidation,
        dry_run=dry_run,
    )
    result["legacy_warning"] = (
        "VISTA3D fine-tuning is a legacy/reference path. The current mainline "
        "uses VoxTell-style 3D prompt student training over 373 exact organs."
    )
    emit(result)


@cli.command("voxtell-student-segment")
@click.option("--ct-image", required=True, help="3D CT NIfTI path.")
@click.option("--output-folder", required=True, help="Output directory for per-prompt 3D masks.")
@click.option("--model-dir", required=True, help="VoxTell model directory containing plans.json and fold_0/checkpoint_final.pth.")
@click.option("--target-config", default="configs/student_3d_prompt_target_organs.json", show_default=True)
@click.option("--prompts", default=None, help="Optional comma-separated global organ names. Defaults to all configured 373 target organs.")
@click.option("--text-encoding-model", default=None, help="Qwen embedding model name or local path. Defaults to local checkpoints/Qwen/Qwen3-Embedding-4B if present.")
@click.option("--device", default="cuda", show_default=True)
@click.option("--gpu", default=0, type=int, show_default=True)
@click.option("--timeout-sec", default=1800, type=int, show_default=True)
@click.option("--prompt-batch-size", default=16, type=int, show_default=True, help="Run prompts in batches to avoid loading all 373 prompts at once.")
@click.option("--backend", default="official_python_api", type=click.Choice(["official_python_api", "official_cli"]), show_default=True, help="Use the official VoxTell Python API or official CLI through the project adapter.")
@click.option("--dry-run", is_flag=True, default=False)
def voxtell_student_segment_cmd(ct_image, output_folder, model_dir, target_config, prompts, text_encoding_model, device, gpu, timeout_sec, prompt_batch_size, backend, dry_run):
    """New 3D prompt-based student inference.

    This route keeps the CT as a 3D volume and uses free-text organ prompts. It
    does not use VISTA3D 127-class label IDs.
    """
    from .core.voxtell_student import VoxTellStudent
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()] if prompts else None
    student = VoxTellStudent(
        model_dir=resolve_path(model_dir),
        device=device,
        gpu=gpu,
        target_config=resolve_path(target_config),
        text_encoding_model=resolve_path(text_encoding_model) if text_encoding_model else None,
        backend=backend,
    )
    emit(student.segment(
        ct_image=resolve_path(ct_image),
        output_dir=resolve_path(output_folder),
        prompts=prompt_list,
        dry_run=dry_run,
        timeout_sec=timeout_sec,
        prompt_batch_size=prompt_batch_size,
    ))


@cli.command("voxtell-student-manifest")
@click.option("--cases-root", required=True, help="Root containing merged case folders with segmentations/*.nii.gz.")
@click.option("--output-manifest", required=True, help="Output JSON manifest path.")
@click.option("--model-dir", required=True, help="VoxTell model directory; not loaded for manifest building.")
@click.option("--target-config", default="configs/student_3d_prompt_target_organs.json", show_default=True)
@click.option("--case-list", default=None, help="Optional CSV with case_id,ct_path to populate image paths.")
@click.option("--require-images/--allow-missing-images", default=False, show_default=True)
def voxtell_student_manifest_cmd(cases_root, output_manifest, model_dir, target_config, case_list, require_images):
    """Build prompt/mask examples for the 3D prompt student M-step."""
    from .core.voxtell_student import VoxTellStudent
    student = VoxTellStudent(
        model_dir=resolve_path(model_dir),
        target_config=resolve_path(target_config),
    )
    emit(student.build_training_manifest(
        cases_root=resolve_path(cases_root),
        output_manifest=resolve_path(output_manifest),
        case_list=resolve_path(case_list) if case_list else None,
        require_images=require_images,
    ))


@cli.command("auto-fine-label-dashboard")
@click.option("--run-output", required=True, help="run-loop output folder containing annotation_versions/.")
@click.option("--target-config", default="configs/student_3d_prompt_target_organs.json", show_default=True)
@click.option("--student-summary", default="", help="Optional student_inference_summary.json.")
@click.option("--failure-json", default="", help="Optional student_failure_cases.json.")
@click.option("--output-dir", default="", help="Default: <run-output>/dashboards.")
def auto_fine_label_dashboard_cmd(run_output, target_config, student_summary, failure_json, output_dir):
    """Build 373-row organ/student auto fine-label dashboards."""
    import subprocess, sys, json as _json
    script = resolve_path("scripts/build_auto_fine_label_dashboard.py")
    cmd = [
        sys.executable,
        str(script),
        "--run-output",
        str(resolve_path(run_output)),
        "--target-config",
        str(resolve_path(target_config)),
    ]
    if student_summary:
        cmd.extend(["--student-summary", str(resolve_path(student_summary))])
    if failure_json:
        cmd.extend(["--failure-json", str(resolve_path(failure_json))])
    if output_dir:
        cmd.extend(["--output-dir", str(resolve_path(output_dir))])
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        data = _json.loads(proc.stdout.strip() or "{}")
    except Exception:
        data = {"status": "failed", "stdout": proc.stdout}
    data["return_code"] = proc.returncode
    if proc.stderr:
        data["stderr_tail"] = proc.stderr[-4000:]
    emit(data)


@cli.command("build-teacher-map")
@click.option("--output", default="configs/teacher_branch_map.yaml", show_default=True, help="输出 YAML 路径")
def build_teacher_map_cmd(output):
    """从 xlsx 分析结果生成 teacher_branch_map.yaml。

    该文件定义每个器官的 VISTA3D label_id、最优 teacher 模型和 M-step 更新策略。
    """
    from .core.teacher_branch_map import build_teacher_branch_map
    result = build_teacher_branch_map(resolve_path(output))
    emit(result)


def main(): cli()
if __name__ == "__main__": main()
