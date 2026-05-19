"""
Stream-download a small subset of PanTSMini cases from Hugging Face.

Instead of downloading the full ~300 GB dataset, this script streams each
tar.gz bundle and extracts only the first N cases to disk, so peak disk
usage is just the size of N extracted cases (typically 5-10 GB for 50 cases).

Usage:
    python scripts/download_pants_mini.py --n-cases 50
    python scripts/download_pants_mini.py --n-cases 10 --images-only
    python scripts/download_pants_mini.py --n-cases 50 --bundle 2

Requirements:
    pip install requests tqdm

IMPORTANT: The HuggingFace dataset (BodyMaps/PanTSMini) uses the CC-BY-NC-SA
4.0 license. Visit https://huggingface.co/datasets/BodyMaps/PanTSMini and
accept the license before downloading if prompted.

Download time estimate (depends on your internet speed):
  - 100 Mbps connection: ~45 min per bundle (images only)
  - 50  Mbps connection: ~90 min per bundle
  The script streams through the full bundle but only writes N cases to disk.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests is not installed. Run:  pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

HF_BASE = "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main"
JHU_LABEL_URL = "http://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"
CASE_PREFIX = "PanTS_"


class _StreamWithProgress:
    """Thin wrapper around an HTTP response stream that prints progress."""

    def __init__(self, raw, total_bytes: int = 0, label: str = ""):
        self._raw = raw
        self._total = total_bytes
        self._read = 0
        self._label = label
        self._last_report_gb = -1.0
        if HAS_TQDM:
            self._bar = tqdm(
                total=total_bytes if total_bytes else None,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=label[:40], dynamic_ncols=True,
            )
        else:
            self._bar = None

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self._read += len(chunk)
        if self._bar:
            self._bar.update(len(chunk))
        else:
            gb = self._read / 1024 ** 3
            if gb - self._last_report_gb > 0.5:
                pct = f" ({100*self._read/self._total:.0f}%)" if self._total else ""
                print(f"  Streamed {gb:.2f} GB{pct} ...", flush=True)
                self._last_report_gb = gb
        return chunk

    def close(self):
        if self._bar:
            self._bar.close()


def _stream_tar_extract_first_n(url: str, output_dir: Path, n: int,
                                 hf_token: str | None = None) -> list[str]:
    """
    Stream a .tar.gz URL and extract only the first N top-level case directories.
    Returns the list of extracted case IDs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    print(f"\nConnecting to: {url}")
    resp = requests.get(url, stream=True, timeout=60, headers=headers)
    if resp.status_code == 401:
        print("ERROR: 401 Unauthorized. You may need to accept the dataset license on")
        print("  https://huggingface.co/datasets/BodyMaps/PanTSMini")
        print("and pass --hf-token <your_token>")
        sys.exit(1)
    resp.raise_for_status()

    total_bytes = int(resp.headers.get("content-length", 0))
    print(f"Bundle size reported by server: "
          f"{total_bytes/1024**3:.1f} GB (streaming through without saving to disk)")
    print(f"Will extract first {n} cases to: {output_dir}\n")

    stream = _StreamWithProgress(resp.raw, total_bytes, label="images")
    extracted_case_ids: list[str] = []
    seen: set[str] = set()

    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                if not member.name or member.name in (".", ""):
                    continue
                parts = Path(member.name).parts
                if not parts:
                    continue
                top = parts[0]
                if not top.startswith(CASE_PREFIX):
                    continue
                if top not in seen:
                    if len(seen) >= n:
                        break
                    seen.add(top)
                    print(f"  [{len(seen):>3}/{n}] Extracting {top}", flush=True)
                tf.extract(member, path=str(output_dir), set_attrs=False)
    except Exception as exc:
        print(f"\nWarning during extraction: {exc}")
    finally:
        stream.close()

    extracted_case_ids = sorted(seen)
    print(f"\nExtracted {len(extracted_case_ids)} cases to {output_dir}")
    return extracted_case_ids


def _stream_tar_extract_by_ids(url: str, output_dir: Path, case_ids: list[str],
                                hf_token: str | None = None) -> int:
    """
    Stream a .tar.gz URL and extract only members belonging to the given case_ids.
    Returns the count of extracted files.
    """
    if not case_ids:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    id_set = set(case_ids)

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    print(f"\nConnecting to: {url}")
    resp = requests.get(url, stream=True, timeout=60, headers=headers)
    resp.raise_for_status()

    total_bytes = int(resp.headers.get("content-length", 0))
    print(f"Label bundle size: {total_bytes/1024**3:.1f} GB")
    print(f"Streaming through to find labels for {len(id_set)} cases ...\n")

    stream = _StreamWithProgress(resp.raw, total_bytes, label="labels")
    extracted_count = 0
    found_ids: set[str] = set()

    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                if not member.name or member.name in (".", ""):
                    continue
                parts = Path(member.name).parts
                if not parts:
                    continue
                top = parts[0]
                if top in id_set:
                    if top not in found_ids:
                        found_ids.add(top)
                        print(f"  [{len(found_ids):>3}/{len(id_set)}] Labels for {top}", flush=True)
                    tf.extract(member, path=str(output_dir), set_attrs=False)
                    extracted_count += 1
    except Exception as exc:
        print(f"\nWarning during label extraction: {exc}")
    finally:
        stream.close()

    print(f"\nExtracted labels for {len(found_ids)} cases ({extracted_count} files)")
    return extracted_count


def _disk_usage_gb(folder: Path) -> float:
    if not folder.exists():
        return 0.0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024 ** 3


def main():
    parser = argparse.ArgumentParser(
        description="Stream-download a small subset of PanTSMini cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", default="third_party/PanTS-main/data",
                        help="Root data directory (default: third_party/PanTS-main/data)")
    parser.add_argument("--n-cases", type=int, default=50,
                        help="Number of training cases to extract (default: 50)")
    parser.add_argument("--bundle", type=int, default=1, choices=range(1, 10),
                        help="Which 1000-case image bundle to use, 1-9 (default: 1 = cases 1-1000)")
    parser.add_argument("--images-only", action="store_true",
                        help="Skip label download (useful if label server is slow)")
    parser.add_argument("--labels-only", action="store_true",
                        help="Only download labels for cases already in output-dir/ImageTr")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace access token (only needed if dataset is gated)")
    args = parser.parse_args()

    if args.n_cases < 1:
        parser.error("--n-cases must be at least 1")

    out = Path(args.output_dir).resolve()
    image_out = out / "ImageTr"
    label_out = out / "LabelTr"

    print("=" * 60)
    print(f"  PanTSMini partial download — {args.n_cases} cases")
    print("=" * 60)
    print(f"Output dir : {out}")
    print(f"Bundle     : {args.bundle} (cases {(args.bundle-1)*1000+1:,} – {args.bundle*1000:,})")
    print(f"Images only: {args.images_only}")
    print(f"Labels only: {args.labels_only}")
    print()

    case_ids: list[str] = []

    if not args.labels_only:
        i = args.bundle
        start = f"{(i-1)*1000+1:08d}"
        end = f"{i*1000:08d}"
        bundle_name = f"PanTSMini_ImageTr_{start}_{end}.tar.gz"
        url = f"{HF_BASE}/{bundle_name}?download=true"
        print(f"--- Downloading images ---")
        case_ids = _stream_tar_extract_first_n(url, image_out, args.n_cases, args.hf_token)
    else:
        # Collect case IDs from already-downloaded images
        if image_out.exists():
            case_ids = sorted(
                d.name for d in image_out.iterdir()
                if d.is_dir() and d.name.startswith(CASE_PREFIX)
            )[:args.n_cases]
            print(f"Found {len(case_ids)} existing case directories in {image_out}")
        else:
            print(f"ERROR: --labels-only specified but {image_out} does not exist.")
            sys.exit(1)

    if not args.images_only and case_ids:
        print(f"\n--- Downloading labels for {len(case_ids)} cases ---")
        _stream_tar_extract_by_ids(JHU_LABEL_URL, label_out, case_ids, args.hf_token)
    elif not case_ids:
        print("\nNo cases extracted — skipping labels.")

    # Summary
    img_gb = _disk_usage_gb(image_out)
    lbl_gb = _disk_usage_gb(label_out)
    print()
    print("=" * 60)
    print("  Download complete!")
    print("=" * 60)
    print(f"  Images  : {img_gb:.2f} GB in {image_out}")
    print(f"  Labels  : {lbl_gb:.2f} GB in {label_out}")
    print(f"  Total   : {img_gb + lbl_gb:.2f} GB")
    print()
    print("Next steps:")
    print(f"  python run_medai_cli.py --json pants-check --pants-root {out.parent}")
    print()
    if case_ids:
        sample = sorted(case_ids)[0]
        print(f"  # Import a case into RadThinking-style layout:")
        print(f"  python run_medai_cli.py --json pants-import-case \\")
        print(f"    --pants-root {out.parent} \\")
        print(f"    --case-id {sample} \\")
        print(f"    --output-root data\\pants_small")
    print()


if __name__ == "__main__":
    main()
