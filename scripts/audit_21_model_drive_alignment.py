#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGET_MODELS = [
    ("cads551", "Dataset551_Totalseg251"),
    ("cads552", "Dataset552_Totalseg252"),
    ("cads553", "Dataset553_Totalseg253"),
    ("cads554", "Dataset554_Totalseg254"),
    ("cads555", "Dataset555_Totalseg255"),
    ("cads556", "Dataset556_GC256"),
    ("cads557", "Dataset557_Brain257"),
    ("cads558", "Dataset558_OAR258"),
    ("cads559", "Dataset559_Saros259"),
    ("moose666", "Dataset666_Peripheral-Bones"),
    ("moose888", "Dataset888_Cardiac"),
    ("nnunet_private", "Dataset224_AbdomenAtlas1.1"),
    ("epai_20250421", "Dataset1339_ePAI"),
    ("saros_nnunet", "Dataset1345_SAROS"),
    ("daps", "Dataset1347_DAPS"),
    ("atm", "Dataset1370_ATM"),
    ("airrc", "Dataset1380_AirRC"),
    ("lvp", "Dataset1381_LVP"),
    ("unest", "UNEST"),
    ("vista3d", "VISTA3D"),
    ("vsmtrans", "VSmTrans"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_size_mb(path: Path) -> float | None:
    if not path.exists() or not path.is_file():
        return None
    return round(path.stat().st_size / 1024 / 1024, 3)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_drive_manifest() -> dict[str, dict[str, str]]:
    manifest_path = ROOT / "docs/checkpoint_drive_export/drive_checkpoints_manifest.csv"
    if not manifest_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["relative_path"]] = row
    return rows


def read_live_drive_listing() -> dict[str, dict[str, str]]:
    listing_path = ROOT / "outputs/audit_21_models/live_drive_listing_gdown.json"
    if not listing_path.exists():
        return {}
    data = load_json(listing_path)
    rows: dict[str, dict[str, str]] = {}
    for row in data.get("files", []):
        path = row.get("path")
        if path:
            rows[path] = row
    return rows


def dataset_root_for(entry: dict[str, Any], key: str) -> Path | None:
    dataset_json = entry.get("dataset_json_path")
    if dataset_json:
        return (ROOT / dataset_json).resolve().parent
    checkpoint_path = entry.get("checkpoint_path")
    if checkpoint_path and key in {"unest", "vista3d"}:
        return (ROOT / checkpoint_path).resolve()
    if key == "vsmtrans":
        return ROOT / "checkpoints/VSmTrans/VSmTrans/nnUNet_results/Dataset001_BDMAP/nnUNetTrainer__nnUNetPlans__3d_fullres"
    return None


def expected_files_for(key: str, entry: dict[str, Any], dataset_root: Path | None) -> list[dict[str, Any]]:
    if key == "unest":
        root = dataset_root or ROOT / "checkpoints/UNEST/UNEST/renalStructures_UNEST_segmentation"
        return [
            {"path": root / "run_UNEST.sh", "required": True},
            {"path": root / "configs/inference.json", "required": True},
            {"path": root / "configs/metadata.json", "required": True},
            {"path": root / "configs/logging.conf", "required": True},
            {"path": root / "models/model.pt", "required": True},
        ]
    if key == "vista3d":
        root = dataset_root or ROOT / "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master"
        return [
            {"path": root / "run.sh", "required": True},
            {"path": root / "README.md", "required": True},
            {"path": root / "configs/inference.json", "required": True},
            {"path": root / "configs/batch_inference.json", "required": True},
            {"path": root / "models/model.pt", "required": True},
        ]
    if key == "vsmtrans":
        root = dataset_root or ROOT / "checkpoints/VSmTrans/VSmTrans/nnUNet_results/Dataset001_BDMAP/nnUNetTrainer__nnUNetPlans__3d_fullres"
        return [
            {"path": root / "dataset.json", "required": True},
            {"path": root / "dataset_fingerprint.json", "required": True},
            {"path": root / "plans.json", "required": True},
            {"path": root / "fold_0/checkpoint_final.pth", "required": True},
            {"path": ROOT / "checkpoints/VSmTrans/VSmTrans/README.md", "required": True},
        ]
    if not dataset_root:
        return []
    checkpoint = entry.get("checkpoint_name", "checkpoint_final")
    return [
        {"path": dataset_root / "dataset.json", "required": True},
        {"path": dataset_root / "dataset_fingerprint.json", "required": True},
        {"path": dataset_root / "plans.json", "required": True},
        {"path": dataset_root / "fold_all/debug.json", "required": False},
        {"path": dataset_root / "fold_all" / f"{checkpoint}.pth", "required": True},
    ]


def drive_rel_candidates(local_path: Path) -> list[str]:
    rp = rel(local_path)
    out = []
    prefix_map = (
        ("checkpoints/CADS_series/CADS_series/", "CADS_series/"),
        ("checkpoints/MOOSE_series/MOOSE_series/", "MOOSE_series/"),
        ("checkpoints/nnUNet_private/nnUNet_private/", "nnUNet_private/"),
        ("checkpoints/UNEST/UNEST/", "UNEST/"),
        ("checkpoints/VISTA3D-Inference-Pipeline-master/", "VISTA3D-Inference-Pipeline-master/"),
        ("checkpoints/VISTA3D/", "VISTA3D/"),
        ("checkpoints/VSmTrans/VSmTrans/", "VSmTrans/"),
    )
    for prefix, drive_prefix in prefix_map:
        if rp.startswith(prefix):
            out.append(drive_prefix + rp[len(prefix):])
            out.append(rp[len(prefix):])
    if rp.startswith("checkpoints/"):
        out.append(rp[len("checkpoints/"):])
    return out


def file_record(
    path: Path,
    drive_manifest: dict[str, dict[str, str]],
    live_drive_listing: dict[str, dict[str, str]],
    required: bool = True,
) -> dict[str, Any]:
    candidates = drive_rel_candidates(path)
    drive_row = next((drive_manifest[c] for c in candidates if c in drive_manifest), None)
    live_drive_row = next((live_drive_listing[c] for c in candidates if c in live_drive_listing), None)
    size = local_size_mb(path)
    size_match = None
    if drive_row and size is not None:
        try:
            size_match = abs(size - float(drive_row["size_mb"])) <= max(0.01, float(drive_row["size_mb"]) * 0.01)
        except Exception:
            size_match = None
    return {
        "path": rel(path),
        "required": required,
        "exists": path.exists(),
        "size_mb": size,
        "drive_manifest_path": drive_row.get("relative_path") if drive_row else None,
        "drive_manifest_size_mb": float(drive_row["size_mb"]) if drive_row else None,
        "size_matches_drive_manifest": size_match,
        "live_drive_path": live_drive_row.get("path") if live_drive_row else None,
        "live_drive_file_id": live_drive_row.get("id") if live_drive_row else None,
    }


def live_drive_expected_paths_for(key: str, entry: dict[str, Any], files: list[dict[str, Any]]) -> list[str]:
    if key == "vista3d":
        return ["VISTA3D/model_bundle.pt"]
    out: list[str] = []
    for f in files:
        if not f.get("required"):
            continue
        path = ROOT / f["path"]
        candidates = drive_rel_candidates(path)
        if candidates:
            out.append(candidates[0])
    return out


def command_alignment(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    template = entry.get("command_template", "")
    recipe = entry.get("recipe")
    if key.startswith("cads"):
        expected = ["nnunetv2_predict_and_split.py", "--folds {folds}"]
        expected_params = {"trainer": "nnUNetTrainerNoMirroring", "plans": "nnUNetResEncUNetLPlans", "folds": "all"}
        source = "run_CADS.sh"
    elif key.startswith("moose"):
        expected = ["nnunetv2_predict_and_split.py", "--folds {folds}"]
        expected_params = {"trainer": "nnUNetTrainerNoMirroring", "plans": "nnUNetResEncUNetLPlans", "folds": "all"}
        source = "run_MOOSE.sh"
    elif recipe == "type1":
        expected = ["nnunetv2_predict_and_split.py", "--folds {folds}"]
        expected_params = {"trainer": "nnUNetTrainer", "plans": "nnUNetResEncUNetLPlans", "folds": "all"}
        source = "run_type1.sh"
    elif recipe == "type2":
        expected = ["nnunetv2_predict_and_split.py", "--folds {folds}"]
        expected_params = {"trainer": "nnUNetTrainer", "plans": "nnUNetPlans", "folds": "all"}
        source = "run_type2.sh"
    elif key == "unest":
        expected = ["unest_predict_and_split.py", "--unest-root"]
        expected_params = {}
        source = "run_UNEST.sh / MONAI bundle"
    elif key == "vista3d":
        expected = ["vista3d_predict_and_split.py", "--vista-root", "--label-map"]
        expected_params = {}
        source = "VISTA3D run.sh / MONAI bundle"
    elif key == "vsmtrans":
        expected = ["nnunetv2_predict_and_split.py", "--folds {folds}"]
        expected_params = {"trainer": "nnUNetTrainer", "plans": "nnUNetPlans", "folds": "0"}
        source = "VSmTrans README nnUNet layout"
    else:
        expected = []
        expected_params = {}
        source = "unknown"
    parameter_alignment = {
        name: {"expected": expected_value, "actual": entry.get(name), "matches": entry.get(name) == expected_value}
        for name, expected_value in expected_params.items()
    }
    dataset_json_path = str(entry.get("dataset_json_path") or "")
    actual_trainer = entry.get("trainer")
    actual_plans = entry.get("plans")
    actual_configuration = entry.get("configuration", "3d_fullres")
    expected_folder_fragment = (
        f"{actual_trainer}__{actual_plans}__{actual_configuration}"
        if actual_trainer and actual_plans
        else None
    )
    return {
        "source_entrypoint": source,
        "has_independent_command_template": bool(template),
        "expected_fragments_present": {fragment: fragment in template for fragment in expected},
        "expected_parameter_alignment": parameter_alignment,
        "all_expected_parameters_match": all(item["matches"] for item in parameter_alignment.values()),
        "checkpoint_folder_fragment": expected_folder_fragment,
        "parameters_match_checkpoint_folder": (
            expected_folder_fragment in dataset_json_path
            if expected_folder_fragment
            else None
        ),
        "command_template": template,
    }


def audit_models() -> dict[str, Any]:
    registry = yaml.safe_load((ROOT / "configs/model_registry.yaml").read_text(encoding="utf-8"))
    models = registry.get("models", {})
    drive_manifest = read_drive_manifest()
    live_drive_listing = read_live_drive_listing()
    reports = {}
    for key, drive_name in TARGET_MODELS:
        entry = models.get(key)
        if not entry:
            reports[key] = {"present_in_registry": False, "drive_name": drive_name}
            continue
        droot = dataset_root_for(entry, key)
        files = [
            file_record(item["path"], drive_manifest, live_drive_listing, item["required"])
            for item in expected_files_for(key, entry, droot)
        ]
        missing = [f["path"] for f in files if f["required"] and not f["exists"]]
        unmatched_drive_manifest = [
            f["path"] for f in files
            if f["required"] and f["exists"] and not f["drive_manifest_path"]
        ]
        drive_mismatches = [
            f["path"] for f in files
            if f["drive_manifest_path"] and f["size_matches_drive_manifest"] is False
        ]
        live_expected_paths = live_drive_expected_paths_for(key, entry, files)
        live_missing = [
            path for path in live_expected_paths
            if live_drive_listing and path not in live_drive_listing
        ]
        reports[key] = {
            "drive_name": drive_name,
            "present_in_registry": True,
            "enabled": bool(entry.get("enabled")),
            "recipe": entry.get("recipe"),
            "distribution_route": entry.get("distribution_route"),
            "private_checkpoint": bool(entry.get("private_checkpoint")),
            "provenance_note": entry.get("provenance_note"),
            "license_note": entry.get("license_note"),
            "dataset_id": entry.get("dataset_id"),
            "checkpoint_name": entry.get("checkpoint_name"),
            "dataset_root": rel(droot) if droot else None,
            "files": files,
            "missing_required_files": missing,
            "required_files_unmatched_to_drive_manifest": unmatched_drive_manifest,
            "drive_size_mismatches": drive_mismatches,
            "live_drive_expected_paths": live_expected_paths,
            "live_drive_missing_paths": live_missing,
            "live_drive_presence_verified": bool(live_drive_listing) and not live_missing,
            "command_alignment": command_alignment(key, entry),
        }
    return {
        "target_model_count": len(TARGET_MODELS),
        "registry_present_count": sum(1 for r in reports.values() if r.get("present_in_registry")),
        "enabled_count": sum(1 for r in reports.values() if r.get("enabled")),
        "models_with_missing_required_files": {
            k: v["missing_required_files"] for k, v in reports.items() if v.get("missing_required_files")
        },
        "models_with_required_files_unmatched_to_drive_manifest": {
            k: v["required_files_unmatched_to_drive_manifest"]
            for k, v in reports.items()
            if v.get("required_files_unmatched_to_drive_manifest")
        },
        "models_with_drive_size_mismatches": {
            k: v["drive_size_mismatches"] for k, v in reports.items() if v.get("drive_size_mismatches")
        },
        "live_drive_listing_available": bool(live_drive_listing),
        "models_with_live_drive_missing_paths": {
            k: v["live_drive_missing_paths"] for k, v in reports.items() if v.get("live_drive_missing_paths")
        },
        "models_with_command_parameter_mismatches": {
            k: v["command_alignment"]["expected_parameter_alignment"]
            for k, v in reports.items()
            if v.get("command_alignment", {}).get("expected_parameter_alignment")
            and not v.get("command_alignment", {}).get("all_expected_parameters_match")
        },
        "models": reports,
    }


def audit_routing() -> dict[str, Any]:
    routing = load_json(ROOT / "configs/organ_routing_from_xlsx.json")
    token_doc = load_json(ROOT / "configs/routing_token_to_model.json")
    global_space = load_json(ROOT / "configs/global_label_space.json")
    registry = yaml.safe_load((ROOT / "configs/model_registry.yaml").read_text(encoding="utf-8"))
    alias_config = load_json(ROOT / "configs/model_label_aliases.json")
    tokens = token_doc.get("tokens", {})
    enabled_registry = {
        key for key, entry in (registry.get("models", {}) or {}).items()
        if entry.get("enabled") is not False
    }

    best_model_by_organ: dict[str, dict[str, Any]] = {}
    unresolved = []
    policy_skipped = []
    all_organs = sorted(routing.get("organ_to_models", {}).keys())
    for organ in all_organs:
        candidates = []
        for token in routing["organ_to_models"].get(organ, []):
            entry = tokens.get(token)
            override = ((token_doc.get("organ_token_overrides", {}) or {}).get(organ, {}) or {}).get(token)
            if isinstance(override, dict):
                entry = override
            if token == "CADS" and entry and entry.get("model") is None:
                chosen = (token_doc.get("bare_cads_organ_overrides", {}) or {}).get(organ)
                entry = {"model": chosen, "enabled": bool(chosen), "subtask": None, "note": "bare CADS override"}
            if not entry:
                candidates.append({"token": token, "model_key": None, "enabled": False, "reason": "token not mapped"})
                continue
            model = entry.get("model")
            enabled = bool(entry.get("enabled")) and bool(model) and model in enabled_registry
            candidates.append({
                "token": token,
                "model_key": model,
                "subtask": entry.get("subtask"),
                "enabled": enabled,
                "reason": entry.get("reason"),
            })
        best = next((c for c in candidates if c["enabled"]), None)
        if best:
            best_model_by_organ[organ] = best
            model_aliases = (alias_config.get("models", {}) or {}).get(best["model_key"], {})
            skip_reason = (model_aliases.get("skip_global_organs", {}) or {}).get(organ)
            if skip_reason:
                policy_skipped.append({
                    "organ": organ,
                    "model_key": best["model_key"],
                    "reason": str(skip_reason),
                    "policy": "skip_unresolvable_coarse_label",
                })
        else:
            unresolved.append({"organ": organ, "candidates": candidates})

    organ_to_id = global_space.get("organ_to_id", {})
    missing_global_ids = [organ for organ in all_organs if organ not in organ_to_id]
    return {
        "xlsx_total_organs": routing.get("total_organs", len(all_organs)),
        "routing_organs": len(all_organs),
        "global_label_space_organs": len(organ_to_id),
        "organs_with_best_enabled_model": len(best_model_by_organ),
        "organs_without_best_enabled_model": len(unresolved),
        "policy_skipped_organs": policy_skipped,
        "policy_skipped_organ_count": len(policy_skipped),
        "static_exact_merge_candidate_organs": len(best_model_by_organ) - len(policy_skipped),
        "accepted_target_coverage": 381,
        "meets_381_target": len(best_model_by_organ) >= 381,
        "unresolved_organs": unresolved,
        "missing_global_label_ids": missing_global_ids,
        "best_model_by_organ": best_model_by_organ,
    }


def main() -> int:
    out_dir = ROOT / "outputs/audit_21_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_drive_url": "https://drive.google.com/drive/folders/1H11EMT83SyteAnh5DwaK5kkJpsr3UEbx",
        "safety": "Drive is treated read-only; all writes are local audit outputs.",
        "model_audit": audit_models(),
        "routing_audit": audit_routing(),
    }
    (out_dir / "drive_alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# 21 Model Drive Alignment Audit",
        "",
        f"- Source Drive: {report['source_drive_url']}",
        "- Drive policy: read/download only; no Drive writes.",
        f"- Target models: {report['model_audit']['target_model_count']}",
        f"- Registry present/enabled: {report['model_audit']['registry_present_count']}/{report['model_audit']['enabled_count']}",
        f"- Missing required local files: {len(report['model_audit']['models_with_missing_required_files'])}",
        f"- Required files unmatched to Drive manifest: {len(report['model_audit']['models_with_required_files_unmatched_to_drive_manifest'])}",
        f"- Drive manifest size mismatches: {len(report['model_audit']['models_with_drive_size_mismatches'])}",
        f"- Live Drive listing available: {report['model_audit']['live_drive_listing_available']}",
        f"- Live Drive missing model paths: {len(report['model_audit']['models_with_live_drive_missing_paths'])}",
        f"- Command parameter mismatches vs Drive run scripts/README: {len(report['model_audit']['models_with_command_parameter_mismatches'])}",
        f"- Organs with enabled best model: {report['routing_audit']['organs_with_best_enabled_model']}",
        f"- Policy-skipped unresolvable coarse-label organs: {report['routing_audit']['policy_skipped_organ_count']}",
        f"- Static exact-merge candidate organs: {report['routing_audit']['static_exact_merge_candidate_organs']}",
        f"- Organs without enabled best model: {report['routing_audit']['organs_without_best_enabled_model']}",
        f"- Meets accepted 381-organ target: {report['routing_audit']['meets_381_target']}",
        "",
        "## Remaining Known Gaps",
        "",
    ]
    cads_private = {
        key: data for key, data in report["model_audit"]["models"].items()
        if key.startswith("cads")
        and data.get("distribution_route") == "private_drive_nnunet_checkpoint"
    }
    if cads_private:
        md_lines.append("CADS private checkpoint / TotalSegmentator naming clarification:")
        md_lines.append(
            "- `cads551`-`cads559` are local Google-Drive CADS nnUNet checkpoints "
            "run with `run_CADS.sh` / `nnUNetv2_predict` parameters. Dataset names "
            "such as `Dataset551_Totalseg251` are treated as CADS/private label-set "
            "names, not as the official public TotalSegmentator package."
        )
        md_lines.append(
            "- Official TotalSegmentator usage is isolated to the separate "
            "`totalsegmentator` registry entry, which uses the installed official "
            "CLI and applies academic-license subtask filtering."
        )
        md_lines.append("")
    policy_skipped = report["routing_audit"]["policy_skipped_organs"]
    unmatched_manifest = report["model_audit"]["models_with_required_files_unmatched_to_drive_manifest"]
    live_missing = report["model_audit"]["models_with_live_drive_missing_paths"]
    command_mismatches = report["model_audit"]["models_with_command_parameter_mismatches"]
    if live_missing:
        md_lines.append("Required model paths missing from the live Drive listing:")
        for key, paths in live_missing.items():
            joined = ", ".join(f"`{p}`" for p in paths)
            md_lines.append(f"- `{key}`: {joined}")
        md_lines.append("")
    if command_mismatches:
        md_lines.append("Command parameter mismatches vs Drive run scripts/README:")
        for key, params in command_mismatches.items():
            model_report = report["model_audit"]["models"][key]
            bad = [
                f"{name}: expected `{value['expected']}`, actual `{value['actual']}`"
                for name, value in params.items()
                if not value.get("matches")
            ]
            folder_note = ""
            if model_report.get("command_alignment", {}).get("parameters_match_checkpoint_folder"):
                folder_note = (
                    "; current parameters match checkpoint folder "
                    f"`{model_report['command_alignment'].get('checkpoint_folder_fragment')}`"
                )
            md_lines.append(f"- `{key}`: " + "; ".join(bad) + folder_note)
        md_lines.append("")
    if unmatched_manifest:
        md_lines.append("Required local files not matched to the cached Drive size manifest:")
        for key, paths in unmatched_manifest.items():
            md_lines.append(f"- `{key}`: {len(paths)} required files need manifest mapping review")
        md_lines.append("")
    if policy_skipped:
        md_lines.append("Policy-skipped SAROS coarse-label organs:")
        for item in policy_skipped:
            md_lines.append(f"- `{item['organ']}`: `{item['model_key']}` -> {item['reason']}")
        md_lines.append("")
    unresolved = report["routing_audit"]["unresolved_organs"]
    if unresolved:
        md_lines.append("No enabled routed model:")
        for item in unresolved:
            reasons = "; ".join(
                f"{c.get('token')} -> {c.get('reason') or c.get('model_key')}"
                for c in item.get("candidates", [])
            )
            md_lines.append(f"- `{item['organ']}`: {reasons}")
    else:
        md_lines.append("- None for routing coverage.")
    md_lines += [
        "",
        "## Notes",
        "",
        "- DAPS was refreshed from Drive with `plans.json`, `dataset_fingerprint.json`, and `fold_all/debug.json`; it uses `checkpoint_best.pth`.",
        "- MOOSE registry uses `nnUNetPlans` because the local/live-Drive checkpoint folders are named `nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres`, even though `run_MOOSE.sh` shows `nnUNetResEncUNetLPlans`; changing to the script value would not match the provided folder layout.",
        "- TotalSegmentator is handled through the official CLI contract, not through the CADS 551-559 private nnUNet weights.",
        "- VISTA3D is handled through a MONAI bundle wrapper aligned with the official pipeline stack; the project wrapper provides single-case CLI and per-model output contract.",
        "- SAROS aggregate labels for arm/head/leg/adipose subcomponents are intentionally policy-skipped; they are not hard-aliased because that would invent unsupported label splits.",
        "",
        "## Per-Model Summary",
        "",
        "| Model key | Drive name | Enabled | Recipe | Checkpoint | Required files missing |",
        "|---|---|---:|---|---|---|",
    ]
    for key, data in report["model_audit"]["models"].items():
        missing = ", ".join(f"`{p}`" for p in data.get("missing_required_files", [])) or "None"
        md_lines.append(
            f"| `{key}` | {data.get('drive_name')} | {data.get('enabled')} | "
            f"{data.get('recipe')} | {data.get('checkpoint_name')} | {missing} |"
        )
    (out_dir / "drive_alignment_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "report": rel(out_dir / "drive_alignment_report.json"),
        "markdown_report": rel(out_dir / "drive_alignment_report.md"),
        "target_model_count": report["model_audit"]["target_model_count"],
        "registry_present_count": report["model_audit"]["registry_present_count"],
        "enabled_count": report["model_audit"]["enabled_count"],
        "models_with_missing_required_files": report["model_audit"]["models_with_missing_required_files"],
        "models_with_required_files_unmatched_to_drive_manifest": report["model_audit"]["models_with_required_files_unmatched_to_drive_manifest"],
        "models_with_drive_size_mismatches": report["model_audit"]["models_with_drive_size_mismatches"],
        "live_drive_listing_available": report["model_audit"]["live_drive_listing_available"],
        "models_with_live_drive_missing_paths": report["model_audit"]["models_with_live_drive_missing_paths"],
        "models_with_command_parameter_mismatches": report["model_audit"]["models_with_command_parameter_mismatches"],
        "organs_with_best_enabled_model": report["routing_audit"]["organs_with_best_enabled_model"],
        "policy_skipped_organ_count": report["routing_audit"]["policy_skipped_organ_count"],
        "static_exact_merge_candidate_organs": report["routing_audit"]["static_exact_merge_candidate_organs"],
        "organs_without_best_enabled_model": report["routing_audit"]["organs_without_best_enabled_model"],
        "meets_381_target": report["routing_audit"]["meets_381_target"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
