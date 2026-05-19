#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

def norm(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())).strip("_")

def load_labels(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("labels", data) if isinstance(data, dict) else {}
    labels: dict[str, int] = {}
    for k, v in raw.items():
        try:
            iv = int(v)
        except Exception:
            continue
        name = norm(k)
        if iv == 0 or name in {"background", "bg"}:
            continue
        labels[name] = iv
    return labels

def split(labelmap: Path, labels_json: Path, output: Path, organs: list[str] | None = None) -> dict:
    import nibabel as nib
    import numpy as np
    img = nib.load(str(labelmap))
    arr = np.asanyarray(img.dataobj)
    labels = load_labels(labels_json)
    if organs:
        wanted = {norm(x) for x in organs}
        labels = {k: v for k, v in labels.items() if norm(k) in wanted}
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, value in sorted(labels.items(), key=lambda x: x[1]):
        mask = (arr == value).astype("uint8")
        vox = int(mask.sum())
        if vox == 0:
            continue
        out = output / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out))
        written.append({"organ": name, "label_value": value, "voxels": vox, "path": str(out)})
    return {"status": "success", "combined_label": str(labelmap), "output": str(output), "num_written": len(written), "written_masks": written, "num_labels_available": len(labels)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelmap", required=True)
    ap.add_argument("--labels-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--organs", default=None)
    args = ap.parse_args()
    organs = [x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()] if args.organs else None
    print(json.dumps(split(Path(args.labelmap).resolve(), Path(args.labels_json).resolve(), Path(args.output).resolve(), organs), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
