#!/usr/bin/env python3
"""Download/extract the minimal PanTS blocks needed for a 50-case debug run.

Disk-space reality check (IMPORTANT for 50GB data disk, e.g. AutoDL):
  Phase 1 — images:
    download image tar block (~34 GB) + extract 50 cases (~10 GB) → peak ~44 GB
    delete tar immediately after extraction → ~10 GB remaining
  Phase 2 — labels:
    download label tar (~15 GB) + already-extracted images (~10 GB) → peak ~25 GB
    delete tar immediately after extraction → ~15 GB final
  Grand total after both phases: ~15 GB on disk.

  A 50 GB data disk is sufficient IF phases are run sequentially (default).
  Never run --download-images and --download-labels simultaneously on a 50 GB disk
  without first verifying that the image tar has already been deleted.

HuggingFace mirror for China servers (AutoDL, etc.):
  Set HF_ENDPOINT=https://hf-mirror.com before running, or the script auto-detects
  it from the environment. Example:
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_pants50_selective.py ...
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Respect HF_ENDPOINT for China mirrors (e.g. https://hf-mirror.com on AutoDL).
_HF_ROOT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_BASE = f"{_HF_ROOT}/datasets/BodyMaps/PanTSMini/resolve/main"
LABEL_URL = "http://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"
DEFAULT_FIRST50 = [f"PanTS_{i:08d}" for i in range(1, 51)]

# Minimum free space (bytes) required before each phase.
_MIN_FREE_IMAGES_GB = 45   # 34 GB tar + 10 GB extracted, plus headroom
_MIN_FREE_LABELS_GB = 20   # 15 GB label tar + some headroom


def _free_gb(path: Path) -> float:
    """Return free disk space in GB for the filesystem containing path."""
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def _check_disk(path: Path, required_gb: float, phase: str) -> None:
    """Print a warning (not an error) if free space may be insufficient."""
    try:
        free = _free_gb(path)
        print(f"[disk] Free space on {path}: {free:.1f} GB (need ~{required_gb} GB for {phase})", flush=True)
        if free < required_gb:
            print(
                f"WARNING: Only {free:.1f} GB free but {phase} needs ~{required_gb} GB. "
                "Proceeding anyway — make sure previous tars were deleted. "
                "If you run out of space, re-run this phase after freeing up space.",
                flush=True,
            )
    except Exception:
        pass  # disk_usage can fail on unusual mount points; don't abort


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def read_case_ids(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_FIRST50
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["case_id"] for r in rows if r.get("case_id")]
    return ids or DEFAULT_FIRST50


def block_for_case(cid: str) -> tuple[str, str, str]:
    n = int(cid.split("_")[1])
    if n >= 9001:
        return "test", "00009001", "00009901"
    start = ((n - 1) // 1000) * 1000 + 1
    end = ((n - 1) // 1000 + 1) * 1000
    return "train", f"{start:08d}", f"{end:08d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pants-root", default="third_party/PanTS-main", help="PanTS repo root containing data/ folder")
    ap.add_argument("--case-list", default=None, help="Optional existing case list; if omitted, uses PanTS_00000001..00000050 to minimize blocks")
    ap.add_argument("--download-images", action="store_true")
    ap.add_argument("--download-labels", action="store_true")
    ap.add_argument("--metadata", action="store_true", help="Download metadata.xlsx")
    ap.add_argument("--yes", action="store_true", help="Confirm large downloads")
    args = ap.parse_args()

    root = Path(args.pants_root).resolve()
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    case_ids = read_case_ids(Path(args.case_list).resolve() if args.case_list else None)
    blocks = sorted(set(block_for_case(cid) for cid in case_ids))

    print("Selected cases:", case_ids[:10], "...", len(case_ids))
    print("Required image blocks:", blocks)
    print()
    print("Disk usage summary:")
    print("  Phase 1 (images): peak ~44 GB during tar download+extract → ~10 GB after tar deleted")
    print("  Phase 2 (labels): peak ~25 GB during label tar download+extract → ~15 GB final")
    print("  Run phases SEQUENTIALLY on a 50 GB disk — do NOT overlap.")
    print()
    if _HF_ROOT != "https://huggingface.co":
        print(f"[hf-mirror] Using HuggingFace mirror: {_HF_ROOT}")
    if (args.download_images or args.download_labels) and not args.yes:
        print("Refusing to start large downloads without --yes. Re-run with --yes after checking storage.")
        return 2

    if args.metadata:
        run(["wget", "--show-progress", "-O", "metadata.xlsx", f"{HF_BASE}/metadata.xlsx?download=true"], cwd=data)

    if args.download_images:
        # Phase 1: download each required tar block, extract the 50 case folders, delete tar immediately.
        _check_disk(data, _MIN_FREE_IMAGES_GB, "image tar download+extract")
        for split, start, end in blocks:
            if split == "train":
                fname = f"PanTSMini_ImageTr_{start}_{end}.tar.gz"
                outdir = data / "ImageTr"
            else:
                fname = "PanTSMini_ImageTe_00009001_00009901.tar.gz"
                outdir = data / "ImageTe"
            outdir.mkdir(parents=True, exist_ok=True)
            url = f"{HF_BASE}/{fname}?download=true"
            print(f"[phase-1] Downloading image block ~34 GB: {fname}", flush=True)
            run(["wget", "--show-progress", "-O", fname, url], cwd=data)
            members = []
            for cid in case_ids:
                b = block_for_case(cid)
                if b == (split, start, end):
                    members.append(f"{cid}/")
            # Extract selected case folders only, then delete tar to reclaim ~34 GB.
            cmd = ["tar", "-xzf", fname, "-C", str(outdir)] + members
            try:
                run(cmd, cwd=data)
            except subprocess.CalledProcessError:
                print("Selective extraction by folder name failed; extracting full block as fallback.")
                run(["tar", "-xzf", fname, "-C", str(outdir)], cwd=data)
            tar_path = data / fname
            print(f"[phase-1] Deleting image tar to reclaim disk space: {tar_path}", flush=True)
            os.remove(tar_path)
            _check_disk(data, 0, "after image tar deleted")
        print("[phase-1] Image extraction complete. Disk reclaimed. Proceed to --download-labels.", flush=True)

    if args.download_labels:
        # Phase 2: download label tar, extract selected cases, delete tar immediately.
        # Must run AFTER image tar is deleted so peak usage stays within 50 GB.
        _check_disk(data, _MIN_FREE_LABELS_GB, "label tar download+extract")
        label_all = data / "LabelAll_selected"
        label_all.mkdir(exist_ok=True)
        fname = "PanTSMini_Label.tar.gz"
        print(f"[phase-2] Downloading label archive (~15 GB): {LABEL_URL}", flush=True)
        run(["wget", "--show-progress", "-O", fname, LABEL_URL], cwd=data)
        members = [f"{cid}/" for cid in case_ids]
        try:
            run(["tar", "-xzf", fname, "-C", str(label_all)] + members, cwd=data)
        except subprocess.CalledProcessError:
            print("Selective label extraction failed; extracting full label archive as fallback.")
            run(["tar", "-xzf", fname, "-C", str(label_all)], cwd=data)
        (data / "LabelTr").mkdir(exist_ok=True)
        (data / "LabelTe").mkdir(exist_ok=True)
        for cid in case_ids:
            src = label_all / cid
            if src.exists():
                dstroot = data / ("LabelTe" if int(cid.split("_")[1]) >= 9001 else "LabelTr")
                dst = dstroot / cid
                if dst.exists():
                    continue
                src.rename(dst)
        tar_path = data / fname
        print(f"[phase-2] Deleting label tar to reclaim disk space: {tar_path}", flush=True)
        os.remove(tar_path)
        _check_disk(data, 0, "after label tar deleted")
        print("[phase-2] Label extraction complete. Final disk usage ~15 GB.", flush=True)

    print("Done. Next: run scripts/select_pants50_cases.py --pants-root", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
