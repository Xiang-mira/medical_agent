from __future__ import annotations

from scheduler.config import load_config
from scheduler.preflight import run_preflight


def test_local_config_loads_and_preflights_without_hpc_manifests():
    cfg = load_config("configs/scheduler.local.example.yaml")
    report = run_preflight(cfg, require_hpc_paths=False)
    assert report["status"] == "success"
    assert report["checks"]["target_config"]["target_count"] == 373
    assert report["checks"]["target_mapping"]["pilot_target_count"] == 338
    assert report["checks"]["target_mapping"]["unresolved_target_count"] == 35
