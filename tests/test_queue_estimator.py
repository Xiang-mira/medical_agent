from __future__ import annotations

from scheduler.queue_estimator import estimate_from_history, heuristic_wait


def test_history_wait_low_confidence_when_samples_sparse():
    out = estimate_from_history([10, 20])
    assert out["wait_estimate"] == "unknown"
    assert out["confidence"] == "low"


def test_history_wait_percentiles():
    out = estimate_from_history([0, 10, 20, 30, 40])
    assert out["estimated_wait_p50"] == 20
    assert out["estimated_wait_p90"] == 40


def test_heuristic_distinguishes_allocatable_shape():
    out = heuristic_wait({"nodes_with_at_least_2_free_gpus": 0, "queued_jobs": 3}, gpu_count=2)
    assert out["queue_expected"] is True
    assert out["confidence"] == "low"
