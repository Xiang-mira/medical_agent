from __future__ import annotations


def make_vlm_label_expert_stub(case_id: str, organ: str, reason: str | None = None) -> dict:
    return {"stage": "vlm_label_expert_stub", "status": "not_implemented", "case_id": case_id, "organ": organ, "reason": reason, "why_stub": "ScaleMAI uses VLM Label Expert after ShapeKit, but this package only reserves the interface; no external API call is made."}
