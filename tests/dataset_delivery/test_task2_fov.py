from __future__ import annotations


def _landmarks(**voxels: int) -> dict[str, dict]:
    names = {
        "brain", "skull", "eyeball_left", "eyeball_right",
        "lung_left", "lung_right", "heart", "aorta",
        "airway", "trachea", "bronchus", "lung_trachea_bronchia",
    }
    return {
        name: {
            "readable": True,
            "foreground_voxels": int(voxels.get(name, 0)),
            "geometry_match": True,
        }
        for name in names
    }


def test_strong_head_evidence_allows_head_targets():
    from tools.dataset_delivery.task2_fov import target_fov_eligibility

    result = target_fov_eligibility(
        "brain_ventricle",
        landmarks=_landmarks(brain=20000, skull=5000, eyeball_left=100),
    )

    assert result["eligible"] is True
    assert result["target_group"] == "head"
    assert result["reason"] == "strong_head_evidence"


def test_weak_head_landmarks_do_not_make_head_targets_eligible():
    from tools.dataset_delivery.task2_fov import target_fov_eligibility

    result = target_fov_eligibility(
        "cerebrospinal_fluid",
        landmarks=_landmarks(brain=290, skull=5279, lung_left=537133, lung_right=516274, heart=366873, aorta=65435),
    )

    assert result["eligible"] is False
    assert result["fov_status"] == "out_of_fov"
    assert result["reason"] == "head_evidence_missing"


def test_bilateral_thorax_evidence_allows_pulmonary_vascular_targets():
    from tools.dataset_delivery.task2_fov import target_fov_eligibility

    result = target_fov_eligibility(
        "lung_pulmonary_arteries",
        landmarks=_landmarks(lung_left=50000, lung_right=50000, heart=2000),
    )

    assert result["eligible"] is True
    assert result["target_group"] == "pulmonary_vascular"
    assert result["reason"] == "thorax_lung_heart_aorta_evidence"


def test_lung_evidence_alone_does_not_allow_central_airway_targets():
    from tools.dataset_delivery.task2_fov import target_fov_eligibility

    result = target_fov_eligibility(
        "airway_wall",
        landmarks=_landmarks(lung_left=50000, lung_right=50000, heart=2000, aorta=2000),
    )

    assert result["eligible"] is False
    assert result["target_group"] == "central_airway"
    assert result["reason"] == "central_airway_evidence_missing"


def test_central_airway_evidence_allows_airway_targets():
    from tools.dataset_delivery.task2_fov import target_fov_eligibility

    result = target_fov_eligibility(
        "airway_tree",
        landmarks=_landmarks(lung_left=50000, lung_right=50000, heart=2000, trachea=500),
    )

    assert result["eligible"] is True
    assert result["target_group"] == "central_airway"
    assert result["reason"] == "central_airway_evidence"
