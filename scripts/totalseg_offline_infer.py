#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path


class OfflineNetworkBlocked(RuntimeError):
    pass


def _install_no_network_guard() -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _is_local(address: object) -> bool:
        if not isinstance(address, tuple) or not address:
            return True
        host = str(address[0])
        return host in {"127.0.0.1", "::1", "localhost"} or host.startswith("127.")

    def guarded_connect(self: socket.socket, address: object) -> None:
        if not _is_local(address):
            raise OfflineNetworkBlocked(f"TOTALSEG_OFFLINE_NETWORK_BLOCKED:{address!r}")
        return original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        if not _is_local(address):
            raise OfflineNetworkBlocked(f"TOTALSEG_OFFLINE_NETWORK_BLOCKED:{address!r}")
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TotalSegmentator in explicit offline mode with network calls blocked.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--roi-subset", nargs="*")
    parser.add_argument("--statistics", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if str(os.getenv("MEDAI_TOTALSEG_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        _install_no_network_guard()

    from totalsegmentator import config as ts_config
    from totalsegmentator import python_api as ts_api
    from totalsegmentator.python_api import totalsegmentator

    if str(os.getenv("MEDAI_TOTALSEG_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        ts_config.send_usage_stats = lambda *a, **k: None
        ts_api.send_usage_stats = lambda *a, **k: None

    args.output.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        args.image,
        args.output,
        task=args.task,
        fast=bool(args.fast),
        roi_subset=args.roi_subset or None,
        device=args.device or "gpu",
        statistics=bool(args.statistics),
        preview=bool(args.preview),
        quiet=bool(args.quiet),
        output_type="nifti",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
