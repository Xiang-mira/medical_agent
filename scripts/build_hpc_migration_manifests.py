#!/usr/bin/env python3
"""Build auditable manifests for the DISCOVERY/HPC migration.

The manifests intentionally separate GitHub code, private Hugging Face assets,
and excluded local data/caches. They are used before upload and again after a
fresh download to prove that the selected restore set is complete.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = ROOT / "docs" / "migration"
HF_REPO_ID = "Xiang-mira/MedIA-Agentic-AI-Private-HPC"

HF_DIRS = [
    ("checkpoints/CADS_series", "CADS_series", "teacher checkpoints"),
    ("checkpoints/MOOSE_series", "MOOSE_series", "teacher checkpoints"),
    ("checkpoints/nnUNet_private", "nnUNet_private", "private nnUNet teacher checkpoints"),
    ("checkpoints/VSmTrans", "VSmTrans", "VSmTrans teacher checkpoint and runtime files"),
    ("checkpoints/UNEST", "UNEST", "UNEST teacher checkpoint"),
    ("checkpoints/ATLAS-Net", "ATLAS-Net", "ATLAS-Net checkpoint and postprocess utilities"),
    ("checkpoints/VoxTell/voxtell_v1.1", "VoxTell/voxtell_v1.1", "official VoxTell v1.1 assets"),
    (
        "checkpoints/VoxTell/embeddings/voxtell_v1.1",
        "VoxTell/embeddings/voxtell_v1.1",
        "official VoxTell text embeddings",
    ),
    ("checkpoints/LabelCritic-main", "LabelCritic-main", "LabelCritic assets mirrored with checkpoints"),
    ("checkpoints/Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-4B", "Qwen3 embedding model for prompt encoding"),
]

HF_FILES = [
    ("checkpoints/class_checkpoint_map.xlsx", "class_checkpoint_map.xlsx", "teacher checkpoint workbook"),
]

VISTA3D_REQUIRED = [
    ("checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master/models", "models"),
    (
        "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master/label_mappings",
        "label_mappings",
    ),
    ("checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master/configs", "configs"),
]
VISTA3D_REQUIRED_FILES = [
    (
        "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master/docs/labels.json",
        "docs/labels.json",
    ),
]

STUDENT_SOURCE = ROOT / "outputs/em_round1_25case_pseudo_label_20260709/round1/mstep_full_stability_C_ampoff_lr3e-5_2000step_20260711"
STUDENT_DEST = "student_models/em_round1_25case_full_mstep_lr3e-5_20260711"
STUDENT_FILES = [
    ("voxtell_finetuned_model/fold_0/checkpoint_final.pth", "formal student checkpoint"),
    ("voxtell_finetuned_model/plans.json", "formal student nnUNet-style plan"),
    ("model_finetune.pth", "formal student training state"),
    ("prompt_embeddings.pt", "formal student prompt embeddings"),
    ("voxtell_prompt_train_result.json", "formal student train result"),
    ("training_provenance.json", "formal student provenance"),
    ("loss_history.json", "formal student loss history"),
    ("sampling_audit.json", "formal student sampling audit"),
    ("organ_exposure_audit.json", "formal student organ exposure audit"),
    ("training_stability_diagnosis.json", "formal student stability gate"),
    ("nonfinite_gradient_audit.json", "formal student gradient audit"),
]

OUTPUT_STATE_DIRS = [
    "outputs/em_round1_25case_pseudo_label_20260709",
    "outputs/em_round2_plus10_20260705",
    "outputs/formal_round1_final_20260627",
    "outputs/round2_estep_mstep_readiness_20260705",
    "outputs/round2_plus10_case_plan_20260705",
    "outputs/labelcritic_373_repair_20260703",
]

OUTPUT_INCLUDE_EXTS = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".log", ".md"}
OUTPUT_EXCLUDE_PARTS = {"student_predictions", "raw_predictions", "standard_dataset", "annotation_versions"}
OUTPUT_EXCLUDE_TOKENS = ("smoke", "dryrun", "bad", "_debug", "_tmp", "_verify")

EXCLUDED_PATHS = [
    ("third_party/PanTS-main/tars", "PanTS tar archives are not migrated; new HPC data will be used"),
    ("third_party/PanTS-main/data", "PanTS extracted data is not migrated; new HPC data will be used"),
    ("data/PanTS", "PanTS image/label data is not migrated"),
    ("outputs/archived_bad_round2_round3_20260705", "known bad historical output archive"),
    ("checkpoints/._____temp", "temporary checkpoint directory"),
    ("checkpoints/.lock", "checkpoint lock/runtime file"),
    ("checkpoints/Qwen/Qwen2-VL-7B-Instruct", "public model; README records direct HF download instead"),
    ("checkpoints/Qwen/Qwen2.5-VL-7B-Instruct", "public model; README records direct HF download instead"),
]

SENSITIVE_GIT_PATTERNS = [
    "third_party/VISTA3D-Inference-Pipeline-master/UCSF_metadata(in).csv",
    "third_party/ePAI-main/**/input_csv/*.csv",
    "third_party/ePAI-main/**/input_csv_*/*.csv",
    "third_party/ePAI-main/reader_study/*.csv",
]


@dataclass(frozen=True)
class ManifestRow:
    destination: str
    repo_path: str
    source_path: str
    size_bytes: int
    sha256: str
    reason: str


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return "missing"
    return digest.hexdigest()


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and d != ".cache")
        for name in sorted(files):
            if "__pycache__" in name:
                continue
            yield Path(root) / name


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def write_tsv(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["destination", "repo_path", "source_path", "size_bytes", "sha256", "reason"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_upload_plan(path: Path, upload_roots: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "local_path", "path_in_repo", "reason"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(upload_roots)


def github_rows(hash_files: bool) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for item in git_ls_files():
        path = ROOT / item
        if not path.is_file():
            continue
        if matches_any(item, SENSITIVE_GIT_PATTERNS):
            continue
        rows.append(
            ManifestRow(
                destination="github",
                repo_path=item,
                source_path=item,
                size_bytes=file_size(path),
                sha256=sha256_file(path) if hash_files else "",
                reason="tracked source/config/doc/lightweight metadata",
            )
        )
    return rows


def add_hf_path(rows: list[ManifestRow], src: Path, dest_prefix: str, reason: str, hash_files: bool) -> None:
    for path in iter_files(src):
        if any(part == ".cache" for part in path.parts) or "__pycache__" in path.parts:
            continue
        rel_src = rel(path)
        rel_under = path.relative_to(src).as_posix() if src.is_dir() else path.name
        repo_path = f"{dest_prefix}/{rel_under}" if src.is_dir() else dest_prefix
        rows.append(
            ManifestRow(
                destination="hf_private",
                repo_path=repo_path,
                source_path=rel_src,
                size_bytes=file_size(path),
                sha256=sha256_file(path) if hash_files else "",
                reason=reason,
            )
        )


def hf_rows(hash_files: bool) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    rows: list[ManifestRow] = []
    upload_roots: list[dict[str, str]] = []
    for src_rel, dest, reason in HF_DIRS:
        src = ROOT / src_rel
        add_hf_path(rows, src, dest, reason, hash_files)
        upload_roots.append({"kind": "folder", "local_path": src_rel, "path_in_repo": dest, "reason": reason})
    for src_rel, dest, reason in HF_FILES:
        add_hf_path(rows, ROOT / src_rel, dest, reason, hash_files)
        upload_roots.append({"kind": "file", "local_path": src_rel, "path_in_repo": dest, "reason": reason})
    vista_base = "VISTA3D-Inference-Pipeline-master"
    for src_rel, dest_suffix in VISTA3D_REQUIRED:
        add_hf_path(rows, ROOT / src_rel, f"{vista_base}/{dest_suffix}", "VISTA3D required runtime subset", hash_files)
        upload_roots.append(
            {
                "kind": "folder",
                "local_path": src_rel,
                "path_in_repo": f"{vista_base}/{dest_suffix}",
                "reason": "VISTA3D required runtime subset",
            }
        )
    for src_rel, dest_suffix in VISTA3D_REQUIRED_FILES:
        add_hf_path(rows, ROOT / src_rel, f"{vista_base}/{dest_suffix}", "VISTA3D labels metadata", hash_files)
        upload_roots.append(
            {
                "kind": "file",
                "local_path": src_rel,
                "path_in_repo": f"{vista_base}/{dest_suffix}",
                "reason": "VISTA3D labels metadata",
            }
        )
    for rel_file, reason in STUDENT_FILES:
        src = STUDENT_SOURCE / rel_file
        add_hf_path(rows, src, f"{STUDENT_DEST}/{rel_file}", reason, hash_files)
        upload_roots.append(
            {
                "kind": "file",
                "local_path": rel(src),
                "path_in_repo": f"{STUDENT_DEST}/{rel_file}",
                "reason": reason,
            }
        )
    for root_rel in OUTPUT_STATE_DIRS:
        root = ROOT / root_rel
        for path in iter_files(root):
            rel_path = rel(path)
            lowered = rel_path.lower()
            if any(part in OUTPUT_EXCLUDE_PARTS for part in path.parts):
                continue
            if any(token in lowered for token in OUTPUT_EXCLUDE_TOKENS):
                continue
            suffix = "".join(path.suffixes)
            if suffix == ".nii.gz" or path.suffix.lower() not in OUTPUT_INCLUDE_EXTS:
                continue
            rows.append(
                ManifestRow(
                    destination="hf_private",
                    repo_path=rel_path,
                    source_path=rel_path,
                    size_bytes=file_size(path),
                    sha256=sha256_file(path) if hash_files else "",
                    reason="lightweight formal experiment state",
                )
            )
        upload_roots.append(
            {
                "kind": "formal_outputs",
                "local_path": root_rel,
                "path_in_repo": root_rel,
                "reason": "filtered lightweight formal experiment state",
            }
        )
    return rows, upload_roots


def excluded_rows(hash_files: bool) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path_rel, reason in EXCLUDED_PATHS:
        path = ROOT / path_rel
        if not path.exists():
            rows.append(ManifestRow("excluded", path_rel, path_rel, 0, "", f"missing locally; {reason}"))
            continue
        if path.is_file():
            files = [path]
        else:
            files = list(iter_files(path))
        if not files:
            rows.append(ManifestRow("excluded", path_rel, path_rel, 0, "", reason))
            continue
        for file_path in files:
            rows.append(
                ManifestRow(
                    destination="excluded",
                    repo_path=rel(file_path),
                    source_path=rel(file_path),
                    size_bytes=file_size(file_path),
                    sha256=sha256_file(file_path) if hash_files else "",
                    reason=reason,
                )
            )
    return rows


def write_json_manifest(path: Path, github: list[ManifestRow], hf: list[ManifestRow], excluded: list[ManifestRow]) -> None:
    payload = {
        "schema_version": "hpc-migration-1.0",
        "hf_repo_id": HF_REPO_ID,
        "policy": {
            "no_pants_data": True,
            "no_raw_medical_images": True,
            "no_vista3d_ucsf_csv": True,
            "qwen2_vl_public_download_only": True,
            "qwen25_vl_public_download_only": True,
        },
        "counts": {
            "github_files": len(github),
            "hf_private_files": len(hf),
            "excluded_rows": len(excluded),
            "hf_private_bytes": sum(row.size_bytes for row in hf),
        },
        "restore_roots": ["checkpoints", "student_models", "outputs"],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_audit_md(path: Path, github: list[ManifestRow], hf: list[ManifestRow], excluded: list[ManifestRow]) -> None:
    def gib(size: int) -> str:
        return f"{size / (1024 ** 3):.2f} GiB"

    text = f"""# HPC Migration Audit

This audit records the selected restore set for moving the project to an HPC GPU
through OnDemand Shell or OnDemand VS Code Server. It intentionally avoids a
blind copy of the local 231G workspace.

## Destinations

- GitHub: {len(github)} tracked source/config/doc files.
- Hugging Face private repo: `{HF_REPO_ID}` with {len(hf)} files, {gib(sum(row.size_bytes for row in hf))}.
- Excluded: {len(excluded)} rows covering PanTS data, caches, public duplicate Qwen VL models, temporary files, and bad/smoke output categories.

## Required HF restore roots

The private HF repo is laid out so that downloading to `checkpoints/` restores
teacher assets at their expected local paths. It also contains
`student_models/em_round1_25case_full_mstep_lr3e-5_20260711/` and filtered
formal state under `outputs/`.

## Sensitive Data Guardrails

PanTS images/labels, PanTS tarballs, NIfTI volumes, probability arrays, runtime
caches, and VISTA3D/UCSF patient CSV metadata are excluded. Qwen2-VL and
Qwen2.5-VL are public upstream models and are documented for direct download
rather than mirrored into the private migration repo.

## Verification

Use `scripts/verify_hf_asset_manifest.py` against `hf_asset_manifest.tsv` after
downloading the HF private repo. Use the README HPC section for the full restore
and smoke-test sequence.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-hash", action="store_true", help="Skip sha256 computation for a quick dry run.")
    args = ap.parse_args()
    hash_files = not args.no_hash

    MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
    github = github_rows(hash_files=hash_files)
    hf, upload_roots = hf_rows(hash_files=hash_files)
    excluded = excluded_rows(hash_files=hash_files)

    write_tsv(MIGRATION_DIR / "github_file_manifest.tsv", github)
    write_tsv(MIGRATION_DIR / "hf_asset_manifest.tsv", hf)
    write_tsv(MIGRATION_DIR / "excluded_manifest.tsv", excluded)
    write_upload_plan(MIGRATION_DIR / "hf_upload_plan.tsv", upload_roots)
    write_json_manifest(MIGRATION_DIR / "asset_manifest.json", github, hf, excluded)
    write_audit_md(MIGRATION_DIR / "HPC_MIGRATION_AUDIT.md", github, hf, excluded)

    print(
        json.dumps(
            {
                "github_files": len(github),
                "hf_private_files": len(hf),
                "hf_private_bytes": sum(row.size_bytes for row in hf),
                "excluded_rows": len(excluded),
                "migration_dir": str(MIGRATION_DIR),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
