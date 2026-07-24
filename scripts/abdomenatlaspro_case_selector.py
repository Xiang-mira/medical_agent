#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.case_selection import (
    build_case_selection_slurm_plan,
    case_selection_fingerprint,
    clean_cache,
    discover_inventory,
    load_case_selection_config,
    merge_metric_shards,
    run_deep_audit_shard,
    run_header_scan_shard,
    run_prefilter_stage,
    run_case_selection,
    run_select_stage,
    scan_ct_header,
    selection_status,
    submit_case_selection_slurm_plan,
    write_inventory_stage,
)
from scheduler.config import resolve_path
from scheduler.utils import SchedulerError


def emit(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _default_output(train: int, test: int, seed: int) -> Path:
    return Path("runs") / "case_selection" / f"abdomenatlaspro_train{train}_test{test}_seed{seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/abdomenatlaspro_case_selector.py")
    parser.add_argument("--config", default="configs/abdomenatlaspro_case_selection.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("inventory", "scan-headers", "merge-header-metrics", "prefilter", "deep-audit", "merge-deep-metrics", "select", "validate", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--train-cases", type=int, default=2)
        p.add_argument("--test-cases", type=int, default=2)
        p.add_argument("--seed", type=int, default=20260724)
        p.add_argument("--output-dir")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--max-inventory-cases", type=int, default=None)
        p.add_argument("--shard-index", type=int, default=None)
        p.add_argument("--num-shards", type=int, default=None)
    p = sub.add_parser("run")
    p.add_argument("--train-cases", type=int, required=True)
    p.add_argument("--test-cases", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", choices=["local", "slurm"], default="local")
    p.add_argument("--output-dir")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reuse-cache", action="store_true", default=True)
    p.add_argument("--no-reuse-cache", action="store_false", dest="reuse_cache")
    p.add_argument("--force-inventory", action="store_true")
    p.add_argument("--force-header-scan", action="store_true")
    p.add_argument("--force-deep-audit", action="store_true")
    p.add_argument("--max-inventory-cases", type=int, default=None)
    p = sub.add_parser("wizard")
    p.add_argument("--max-inventory-cases", type=int, default=None)
    p = sub.add_parser("status")
    p.add_argument("--selection-dir", required=True)
    p = sub.add_parser("clean-cache")
    p.add_argument("--selection-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_case_selection_config(args.config)
        if args.cmd == "wizard":
            train = int(input("Train cases: ").strip())
            test = int(input("Test cases: ").strip())
            seed = int(input("Seed [20260724]: ").strip() or "20260724")
            max_inventory_cases = args.max_inventory_cases
            out = _default_output(train, test, seed)
            confirm = input(f"Freeze split in {out}? Type yes to continue: ").strip().lower()
            if confirm != "yes":
                emit({"status": "cancelled"})
                return 0
            emit(run_case_selection(train, test, seed, out, config_path=args.config, max_inventory_cases=max_inventory_cases))
            return 0
        if args.cmd == "status":
            emit(selection_status(Path(args.selection_dir)))
            return 0
        if args.cmd == "clean-cache":
            emit(clean_cache(Path(args.selection_dir)))
            return 0
        if args.cmd == "inventory":
            image_root = resolve_path(cfg.paths.get("image_root"))
            mask_root = resolve_path(cfg.paths.get("mask_root"))
            assert image_root is not None and mask_root is not None
            if args.output_dir:
                emit(write_inventory_stage(args.train_cases, args.test_cases, args.seed, Path(args.output_dir), config_path=args.config, max_inventory_cases=args.max_inventory_cases))
                return 0
            emit({"status": "success", "cases": discover_inventory(image_root, mask_root, max_inventory_cases=args.max_inventory_cases)[:20]})
            return 0
        if args.cmd == "scan-headers":
            if args.output_dir and args.shard_index is not None and args.num_shards is not None:
                emit(run_header_scan_shard(args.train_cases, args.test_cases, args.seed, Path(args.output_dir), config_path=args.config, shard_index=args.shard_index, num_shards=args.num_shards, max_inventory_cases=args.max_inventory_cases))
                return 0
            image_root = resolve_path(cfg.paths.get("image_root"))
            mask_root = resolve_path(cfg.paths.get("mask_root"))
            assert image_root is not None and mask_root is not None
            rows = discover_inventory(image_root, mask_root, max_inventory_cases=args.max_inventory_cases)[: int(args.train_cases) + int(args.test_cases)]
            emit({"status": "success", "headers": [scan_ct_header(Path(r["ct_path"])) for r in rows]})
            return 0
        out = Path(args.output_dir) if args.output_dir else _default_output(args.train_cases, args.test_cases, args.seed)
        fingerprint = case_selection_fingerprint(cfg, args.train_cases, args.test_cases, args.seed, args.max_inventory_cases)
        if args.cmd == "merge-header-metrics":
            emit(merge_metric_shards("header", out, expected_shards=args.num_shards or 20, expected_fingerprint=fingerprint))
            return 0
        if args.cmd == "prefilter":
            emit(run_prefilter_stage(args.train_cases, args.test_cases, args.seed, out, config_path=args.config, max_inventory_cases=args.max_inventory_cases))
            return 0
        if args.cmd == "deep-audit" and args.shard_index is not None and args.num_shards is not None:
            emit(run_deep_audit_shard(args.train_cases, args.test_cases, args.seed, out, config_path=args.config, shard_index=args.shard_index, num_shards=args.num_shards, max_inventory_cases=args.max_inventory_cases))
            return 0
        if args.cmd == "merge-deep-metrics":
            emit(merge_metric_shards("deep_audit", out, expected_shards=args.num_shards or 10, expected_fingerprint=fingerprint))
            return 0
        if args.cmd == "select":
            emit(run_select_stage(args.train_cases, args.test_cases, args.seed, out, config_path=args.config, max_inventory_cases=args.max_inventory_cases))
            return 0
        if args.cmd == "validate":
            emit(selection_status(out))
            return 0
        if args.cmd == "run" and args.backend == "slurm":
            plan = build_case_selection_slurm_plan(args.train_cases, args.test_cases, args.seed, out, config_path=args.config, dry_run=args.dry_run, max_inventory_cases=args.max_inventory_cases)
            if args.dry_run:
                emit(plan)
                return 0
            emit(submit_case_selection_slurm_plan(plan))
            return 0
        emit(run_case_selection(args.train_cases, args.test_cases, args.seed, out, config_path=args.config, dry_run=args.dry_run, cache_options={
            "reuse_cache": getattr(args, "reuse_cache", True),
            "force_inventory": getattr(args, "force_inventory", False),
            "force_header_scan": getattr(args, "force_header_scan", False),
            "force_deep_audit": getattr(args, "force_deep_audit", False),
        }, max_inventory_cases=args.max_inventory_cases))
    except SchedulerError as exc:
        emit({"status": "failed", "error": str(exc)})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
