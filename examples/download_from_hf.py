#!/usr/bin/env python3
from __future__ import annotations
import argparse
from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description='Download MedIA model assets from Hugging Face.')
    parser.add_argument('--repo-id', default='Xiang-mira/MedIA-Agentic-AI')
    parser.add_argument('--local-dir', default='checkpoints/MedIA-Agentic-AI')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    path = snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=args.local_dir)
    print(path)
    print('Set HF_ASSET_ROOT=' + args.local_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
