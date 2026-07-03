#!/usr/bin/env python3
"""Read-only audit of official VoxTell source, checkpoint, plans and prompt bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "ec517b79a19aa59b25789c878d808790326e9651"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", type=Path, default=ROOT / "third_party/VoxTell")
    ap.add_argument("--model", type=Path, default=ROOT / "checkpoints/VoxTell/voxtell_v1.1")
    ap.add_argument("--bank", type=Path, default=ROOT / "checkpoints/VoxTell/embeddings/voxtell_v1.1/text_embeddings.npz")
    ap.add_argument("--labels", type=Path, default=ROOT / "checkpoints/VoxTell/embeddings/voxtell_v1.1/labels.json")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.vendor), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(args.vendor), "status", "--short"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    ).stdout.splitlines()
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels.exists() else []
    bank_shape = None
    bank_dtype = None
    bank_labels_match = False
    if args.bank.exists():
        with np.load(args.bank, allow_pickle=False) as bank:
            bank_shape = list(bank["embeddings"].shape)
            bank_dtype = str(bank["embeddings"].dtype)
            bank_labels_match = list(bank["labels"]) == labels
    checkpoint = args.model / "fold_0/checkpoint_final.pth"
    checkpoint_keys = []
    tensor_groups = {}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = state.get("network_weights", {})
        checkpoint_keys = list(state)
        for key in weights:
            group = key.split(".", 1)[0]
            tensor_groups[group] = tensor_groups.get(group, 0) + 1
    errors = []
    if commit != EXPECTED_COMMIT:
        errors.append("vendor_commit_mismatch")
    if status:
        errors.append("vendor_dirty")
    if len(labels) != 14194 or bank_shape != [14194, 2560] or not bank_labels_match:
        errors.append("official_prompt_bank_contract_failed")
    if checkpoint_keys != ["network_weights"]:
        errors.append("checkpoint_contract_changed")
    for required in ("encoder", "decoder", "transformer_decoder", "project_text_embed"):
        if required not in tensor_groups:
            errors.append(f"checkpoint_missing_{required}")
    payload = {
        "stage": "voxtell_official_asset_audit",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "read_only": True,
        "external_uploads_performed": False,
        "vendor": {"commit": commit, "expected_commit": EXPECTED_COMMIT, "dirty_files": status},
        "model": {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "plans_sha256": sha256(args.model / "plans.json"), "checkpoint_top_keys": checkpoint_keys,
            "tensor_groups": tensor_groups,
        },
        "prompt_bank": {
            "license": "cc-by-nc-sa-4.0",
            "labels": len(labels), "embedding_shape": bank_shape, "embedding_dtype": bank_dtype,
            "labels_match_npz": bank_labels_match, "labels_sha256": sha256(args.labels),
            "bank_sha256": sha256(args.bank),
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
