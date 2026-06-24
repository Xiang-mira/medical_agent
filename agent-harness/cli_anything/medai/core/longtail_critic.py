"""Self-supervised structural-corruption critic for long-tail organ masks.

This critic distinguishes high-confidence seeds from programmatically corrupted
versions. Its output is explicitly *not* segmentation accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .auto_label_core import stable_case_fold


FEATURE_NAMES = ["volume_fraction", "component_log", "surface_ratio", "bbox_fill", "centroid_x", "centroid_y", "centroid_z"]


def mask_features(array: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, label
    mask = np.asarray(array) > 0
    volume = int(mask.sum())
    if volume == 0:
        return np.asarray([0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5], dtype=np.float32)
    components = int(label(mask)[1])
    surface = np.logical_xor(mask, binary_erosion(mask))
    coords = np.argwhere(mask)
    spans = coords.max(axis=0) - coords.min(axis=0) + 1
    bbox_volume = int(np.prod(spans))
    centroid = coords.mean(axis=0) / np.maximum(np.asarray(mask.shape) - 1, 1)
    return np.asarray([
        volume / mask.size,
        np.log1p(components) / np.log(32.0),
        int(surface.sum()) / max(volume, 1),
        volume / max(bbox_volume, 1),
        *centroid.tolist(),
    ], dtype=np.float32)


def synthetic_corruptions(mask: np.ndarray, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    from scipy.ndimage import binary_dilation, binary_erosion
    rng = rng or np.random.default_rng(42)
    src = np.asarray(mask) > 0
    shifted = np.roll(src, shift=max(2, src.shape[0] // 12), axis=0)
    fragment = src.copy()
    fragment[tuple(slice(0, max(1, s // 3)) for s in src.shape)] = False
    extra = src.copy()
    point = tuple(int(rng.integers(0, max(1, s))) for s in src.shape)
    radius = max(1, min(src.shape) // 20)
    slices = tuple(slice(max(0, p - radius), min(s, p + radius + 1)) for p, s in zip(point, src.shape))
    extra[slices] = True
    return {
        "shift": shifted,
        "flip_lr": np.flip(src, axis=0).copy(),
        "dilate": binary_dilation(src, iterations=max(1, min(src.shape) // 32)),
        "erode": binary_erosion(src, iterations=max(1, min(src.shape) // 32)),
        "missing_fragment": fragment,
        "added_fragment": extra,
    }


def build_pairwise_dataset(passports: list[dict[str, Any]], output_path: str | Path, seed: int = 42) -> dict[str, Any]:
    import nibabel as nib
    rows = []
    rng = np.random.default_rng(seed)
    for passport in passports:
        if passport.get("grade") != "A" or int(passport.get("independent_family_count") or 0) < 2:
            continue
        path = Path(passport.get("mask_path") or "")
        if not path.exists():
            continue
        source = np.asanyarray(nib.load(str(path)).dataobj) > 0
        clean = mask_features(source).tolist()
        case_id = str(passport.get("case_id") or path.parent.name)
        for corruption, damaged in synthetic_corruptions(source, rng).items():
            rows.append({
                "case_id": case_id, "organ": passport.get("organ"), "fold": stable_case_fold(case_id),
                "clean_features": clean, "corrupted_features": mask_features(damaged).tolist(),
                "corruption": corruption, "target": "clean_ranks_above_corrupted",
            })
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": "longtail_critic_pairs_v1", "feature_names": FEATURE_NAMES, "pairs": rows}, indent=2), encoding="utf-8")
    return {"status": "success", "output": str(out), "num_pairs": len(rows), "num_cases": len({r['case_id'] for r in rows})}


def train_pairwise_ranker(dataset_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    """Fit a small deterministic pairwise logistic ranker with case-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    doc = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    pairs = doc.get("pairs", [])
    if not pairs:
        return {"status": "failed", "reason": "no A-grade pairwise seeds"}
    x, y, folds = [], [], []
    absolute_x, corruption_y, absolute_folds = [], [], []
    for row in pairs:
        clean = np.asarray(row["clean_features"]); corrupted = np.asarray(row["corrupted_features"])
        delta = clean - corrupted
        x.extend([delta, -delta]); y.extend([1, 0]); folds.extend([int(row["fold"]), int(row["fold"])])
        absolute_x.extend([clean, corrupted]); corruption_y.extend([0, 1]); absolute_folds.extend([int(row["fold"]), int(row["fold"])])
    x = np.asarray(x); y = np.asarray(y); folds = np.asarray(folds)
    absolute_x = np.asarray(absolute_x); corruption_y = np.asarray(corruption_y); absolute_folds = np.asarray(absolute_folds)
    fold_auc = {}
    for fold in range(5):
        train = folds != fold; test = folds == fold
        if train.sum() < 2 or test.sum() < 2 or len(np.unique(y[train])) < 2:
            continue
        model = LogisticRegression(random_state=42, max_iter=1000).fit(x[train], y[train])
        fold_auc[str(fold)] = float(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    model = LogisticRegression(random_state=42, max_iter=1000).fit(x, y)
    corruption_model = LogisticRegression(random_state=42, max_iter=1000).fit(absolute_x, corruption_y)
    out = Path(model_path); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "longtail_critic_ranker_v1", "feature_names": FEATURE_NAMES,
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "fold_auc": fold_auc,
        "corruption_coef": corruption_model.coef_[0].tolist(),
        "corruption_intercept": float(corruption_model.intercept_[0]),
        "output_semantics": "structural_corruption_probability_not_segmentation_accuracy",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "success", "output": str(out), "num_pairs": len(pairs), "fold_auc": fold_auc}


def structural_corruption_probability(mask: np.ndarray, model_doc: dict[str, Any]) -> float:
    features = mask_features(mask)
    logit = float(np.dot(np.asarray(model_doc["corruption_coef"], dtype=float), features) + float(model_doc["corruption_intercept"]))
    return float(1.0 / (1.0 + np.exp(-logit)))
