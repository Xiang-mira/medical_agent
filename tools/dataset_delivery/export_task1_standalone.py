#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import sha256_file, validate_rename_mapping, write_json  # noqa: E402


GENERATED_RELATIVE_PATHS = [
    "README.md",
    "export_manifest.json",
    "configs/organ_rename_mapping_373.csv",
    "configs/student_3d_prompt_target_organs.json",
    "configs/task2_generate_targets_23.csv",
    "configs/task1_alias_groups.csv",
    "configs/task_boundary_classification.csv",
    "configs/non_rename_decisions.csv",
    "tools/__init__.py",
    "tools/dataset_delivery/__init__.py",
    "tools/dataset_delivery/delivery_lib.py",
    "tools/dataset_delivery/rename_anatomical_labels.py",
    "tools/dataset_delivery/validate_rename_mapping.py",
]

OBSOLETE_RELATIVE_PATHS = [
    "non_rename_decisions.csv",
    "resolved_unmatched_targets.txt",
    "unresolved_unmatched_targets.txt",
]


def copy_file(src: Path, dst: Path) -> dict[str, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "target": str(dst), "sha256": sha256_file(dst)}


def clear_generated_files(output_dir: Path) -> None:
    for rel in GENERATED_RELATIVE_PATHS + OBSOLETE_RELATIVE_PATHS:
        path = output_dir / rel
        if path.exists() and path.is_file():
            path.unlink()


def export_standalone(
    *,
    mapping: Path,
    taxonomy: Path,
    task2_targets: Path,
    alias_groups: Path,
    boundary_classification: Path,
    non_rename_decisions: Path,
    output_dir: Path,
) -> dict:
    validate_rename_mapping(
        mapping,
        taxonomy,
        task2_targets=task2_targets,
        alias_groups=alias_groups,
        boundary_classification=boundary_classification,
    )
    clear_generated_files(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = [
        copy_file(mapping, output_dir / "configs" / "organ_rename_mapping_373.csv"),
        copy_file(taxonomy, output_dir / "configs" / "student_3d_prompt_target_organs.json"),
        copy_file(task2_targets, output_dir / "configs" / "task2_generate_targets_23.csv"),
        copy_file(alias_groups, output_dir / "configs" / "task1_alias_groups.csv"),
        copy_file(boundary_classification, output_dir / "configs" / "task_boundary_classification.csv"),
        copy_file(non_rename_decisions, output_dir / "configs" / "non_rename_decisions.csv"),
        copy_file(repo_root / "tools" / "dataset_delivery" / "delivery_lib.py", output_dir / "tools" / "dataset_delivery" / "delivery_lib.py"),
        copy_file(repo_root / "tools" / "dataset_delivery" / "rename_anatomical_labels.py", output_dir / "tools" / "dataset_delivery" / "rename_anatomical_labels.py"),
        copy_file(repo_root / "tools" / "dataset_delivery" / "validate_rename_mapping.py", output_dir / "tools" / "dataset_delivery" / "validate_rename_mapping.py"),
    ]
    for init in (output_dir / "tools" / "__init__.py", output_dir / "tools" / "dataset_delivery" / "__init__.py"):
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("", encoding="utf-8")
        copied.append({"source": "generated_empty_init", "target": str(init), "sha256": sha256_file(init)})
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join([
            "# Task 1 Rename Standalone Export",
            "",
            "This directory is generated from canonical source in `configs/dataset_delivery/task1/` and `tools/dataset_delivery/`.",
            "Do not hand-edit files here; regenerate with `tools/dataset_delivery/export_task1_standalone.py`.",
            "",
            "Task 1 only renames existing masks when anatomy, laterality, and granularity are identical.",
            "Task 2 targets are listed separately and are never executed as Task 1 renames.",
            "",
            "Validate this export from the repository root:",
            "",
            "```bash",
            "python tools/dataset_delivery/validate_rename_mapping.py \\",
            "  --mapping configs/dataset_delivery/task1/organ_rename_mapping_373.csv \\",
            "  --taxonomy configs/student_3d_prompt_target_organs.json \\",
            "  --task2-targets configs/dataset_delivery/task1/task2_generate_targets_23.csv \\",
            "  --alias-groups configs/dataset_delivery/task1/task1_alias_groups.csv",
            "```",
            "",
        ]),
        encoding="utf-8",
    )
    copied.append({"source": "generated_readme", "target": str(readme), "sha256": sha256_file(readme)})
    manifest = {"status": "success", "output_dir": str(output_dir), "files": copied}
    write_json(output_dir / "export_manifest.json", manifest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Export Task 1 standalone package from canonical source.")
    p.add_argument("--mapping", default=Path("configs/dataset_delivery/task1/organ_rename_mapping_373.csv"), type=Path)
    p.add_argument("--taxonomy", default=Path("configs/student_3d_prompt_target_organs.json"), type=Path)
    p.add_argument("--task2-targets", default=Path("configs/dataset_delivery/task1/task2_generate_targets_23.csv"), type=Path)
    p.add_argument("--alias-groups", default=Path("configs/dataset_delivery/task1/task1_alias_groups.csv"), type=Path)
    p.add_argument("--boundary-classification", default=Path("configs/dataset_delivery/task1/task_boundary_classification.csv"), type=Path)
    p.add_argument("--non-rename-decisions", default=Path("configs/dataset_delivery/task1/non_rename_decisions.csv"), type=Path)
    p.add_argument("--output-dir", default=Path("deliverables/task1_rename_refactor_source"), type=Path)
    args = p.parse_args()
    result = export_standalone(
        mapping=args.mapping,
        taxonomy=args.taxonomy,
        task2_targets=args.task2_targets,
        alias_groups=args.alias_groups,
        boundary_classification=args.boundary_classification,
        non_rename_decisions=args.non_rename_decisions,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
