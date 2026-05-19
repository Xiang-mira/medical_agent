from __future__ import annotations

import base64
import json
from pathlib import Path

from .label_verifier import verify_annotation
from .projection_builder import build_projection


def _encode_image_b64(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def _call_ollama(model: str, prompt: str, image_paths: list[str], timeout: int = 600) -> dict:
    try:
        import requests as req_lib

        images_b64 = []
        for ip in image_paths:
            b64 = _encode_image_b64(ip)
            if b64:
                images_b64.append(b64)

        msg: dict = {"role": "user", "content": prompt}
        if images_b64:
            msg["images"] = images_b64

        payload = {
            "model": model,
            "messages": [msg],
            "stream": False,
            "think": False,
            "options": {"num_predict": 500},
        }

        resp = req_lib.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=timeout,
            proxies={"http": None, "https": None},  # bypass system proxy for localhost
        )
        resp.raise_for_status()
        result = resp.json()
        content = result.get("message", {}).get("content", "")
        return {"status": "success", "raw_response": content}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def _call_vllm(model: str, prompt: str, image_paths: list[str], base_url: str = "http://localhost:8000/v1", timeout: int = 600) -> dict:
    """Call a vLLM or OpenAI-compatible VLM server (as used by LabelCritic)."""
    try:
        import requests as req_lib

        content_parts: list[dict] = []
        for ip in image_paths:
            b64 = _encode_image_b64(ip)
            if b64:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content_parts.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": 500,
        }

        resp = req_lib.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"status": "success", "raw_response": content}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 models emit before the answer."""
    import re
    # Remove <think> blocks (may span multiple lines)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _parse_vlm_response(raw_text: str) -> dict:
    """Extract JSON from VLM response; gracefully handle partial JSON and thinking blocks."""
    import re
    text = _strip_thinking(raw_text)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            winner = parsed.get("winner", "uncertain")
            confidence = float(parsed.get("confidence", 0.5))
            reason = parsed.get("reason", "")
            return {"winner": winner, "confidence": confidence, "reason": reason, "parse_status": "success"}
    except Exception:
        pass
    # Strict fallback: only match unambiguous affirmative selection phrases.
    # "Candidate A is worse than B" must NOT trigger winner=A.
    _WIN_A = re.compile(
        r'\b(winner\s*[=:]\s*["\']?a["\']?|'
        r'select\s+a\b|choose\s+a\b|prefer\s+a\b|'
        r'a\s+is\s+(better|correct|superior|more\s+accurate|more\s+plausible))',
        re.I,
    )
    _WIN_B = re.compile(
        r'\b(winner\s*[=:]\s*["\']?b["\']?|'
        r'select\s+b\b|choose\s+b\b|prefer\s+b\b|'
        r'b\s+is\s+(better|correct|superior|more\s+accurate|more\s+plausible))',
        re.I,
    )
    if _WIN_A.search(text):
        return {"winner": "A", "confidence": 0.55, "reason": text[:300], "parse_status": "keyword_fallback"}
    if _WIN_B.search(text):
        return {"winner": "B", "confidence": 0.55, "reason": text[:300], "parse_status": "keyword_fallback"}
    # Medical context: uncertain is safer than a wrong guess.
    return {"winner": "uncertain", "confidence": 0.0, "reason": raw_text[:300], "parse_status": "unparseable"}


def _make_text_prompt(organ: str, organ_desc: str, verifier: dict) -> str:
    dice = verifier.get("dice", "unknown")
    ann_vox = verifier.get("current_voxels", "unknown")
    pred_vox = verifier.get("prediction_voxels", "unknown")
    return f"""You are a medical image annotation expert reviewing CT segmentation quality.

Organ: {organ}
Description: {organ_desc}

Annotation statistics:
- Candidate A (current annotation): {ann_vox} voxels
- Candidate B (model prediction): {pred_vox} voxels
- Dice Similarity Coefficient between A and B: {dice}

The {organ} typically has a consistent volume range in CT. Large discrepancies in voxel count suggest one annotation may be incorrect.

Which candidate annotation is more likely to be correct?
Consider: if DSC is moderate (0.5-0.8), both may be partially correct with boundary differences.
If one has very few voxels compared to the other, it may be under-segmented.

Respond with JSON only:
{{"winner": "A" or "B" or "uncertain", "confidence": 0.0-1.0, "reason": "brief anatomical reasoning"}}"""


VLM_PROMPT_TEMPLATE = """You are a medical image annotation expert reviewing CT segmentation masks.

Organ: {organ}
Task: Compare two candidate segmentation annotations for this organ.

The image shows a CT slice with two overlays side by side:
- LEFT panel = Candidate A (current annotation)
- RIGHT panel = Candidate B (model prediction)

The red/highlighted region in each panel shows where the model placed the {organ} mask.

The {organ} in abdominal CT typically appears as:
{organ_description}

Which candidate annotation is anatomically more plausible?

Respond with JSON only, no other text:
{{"winner": "A" or "B" or "uncertain", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

ORGAN_DESCRIPTIONS = {
    "pancreas": "an elongated soft-tissue structure in the upper abdomen, posterior to the stomach, anterior to the spine",
    "liver": "the largest solid organ, occupying the right upper abdomen, homogeneous density",
    "spleen": "a rounded solid organ in the left upper abdomen, similar density to liver",
    "kidney_left": "a bean-shaped organ in the left retroperitoneum, with bright cortex and darker medulla",
    "kidney_right": "a bean-shaped organ in the right retroperitoneum, with bright cortex and darker medulla",
    "aorta": "a tubular structure along the spine, high-density on contrast CT",
    "stomach": "a hollow viscus in the left upper abdomen, variable in shape depending on filling",
}


def run_vlm_label_expert(
    ct_image: str | Path,
    annotation_a: str | Path | None,
    annotation_b: str | Path | None,
    organ: str,
    output_folder: str | Path,
    vlm_backend: str = "ollama",
    vlm_model: str = "qwen2.5vl:7b",
    case_id: str | None = None,
    dsc_replace_threshold: float = 0.0,
    dsc_vlm_threshold: float = 0.5,
    strict_alignment: bool = False,
    vllm_base_url: str = "http://localhost:8000/v1",
) -> dict:
    """
    Full VLM Label Expert pipeline:
    1. Label Verifier (DSC check)
    2. 2D projection builder
    3. VLM pairwise comparison
    4. Return decision JSON
    """
    out_dir = Path(output_folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "stage": "vlm_label_expert",
        "organ": organ,
        "case_id": case_id,
        "vlm_backend": vlm_backend,
        "vlm_model": vlm_model,
        "vllm_base_url": vllm_base_url if vlm_backend == "vllm" else None,
        "ct_image": str(ct_image),
        "annotation_a": str(annotation_a) if annotation_a else None,
        "annotation_b": str(annotation_b) if annotation_b else None,
    }

    # Step 1: Label Verifier — check if VLM is even needed
    verifier = verify_annotation(annotation_a, annotation_b, organ,
                                 dsc_replace_threshold, dsc_vlm_threshold)
    result["label_verifier"] = verifier

    if verifier.get("decision") == "accept":
        result.update({"status": "success", "decision": "accept",
                       "winner": "A", "confidence": 1.0,
                       "reason": f"DSC={verifier.get('dice')} above threshold; annotation accepted without VLM.",
                       "vlm_called": False})
        return result

    if verifier.get("decision") == "auto_replace_candidate":
        result.update({"status": "success", "decision": "auto_replace_candidate",
                       "winner": "B", "confidence": 0.9,
                       "reason": verifier.get("reason", ""),
                       "vlm_called": False})
        return result

    # Step 2: Build 2D projection for VLM input
    # Stub backend skips projection entirely — avoids running the expensive LabelCritic script.
    if vlm_backend == "stub":
        result.update({
            "status": "warning",
            "decision": "review_queue",
            "winner": "uncertain",
            "confidence": 0.0,
            "reason": "VLM backend is stub. Routed to human review.",
            "vlm_called": False,
            "why_stub": "Set --vlm-backend ollama and ensure Ollama is running with a vision model.",
        })
        return result

    proj_dir = out_dir / "projections"
    proj = build_projection(ct_image, annotation_a, annotation_b, proj_dir, organ,
                            strict_alignment=strict_alignment)
    result["projection"] = proj

    if proj.get("status") == "failed":
        result.update({
            "status": "warning",
            "decision": "review_queue",
            "winner": "uncertain",
            "confidence": 0.0,
            "reason": f"Projection failed: {proj.get('reason', 'unknown')}. Routed to human review.",
            "vlm_called": False,
        })
        return result

    # Step 3: Call VLM if projections available and backend is not stub
    projection_images = proj.get("saved_projections", [])
    png_images = [p for p in projection_images if p.endswith(".png")]

    if vlm_backend == "stub" or not png_images:
        result.update({
            "status": "warning",
            "decision": "review_queue",
            "winner": "uncertain",
            "confidence": 0.0,
            "reason": "VLM backend is stub or no projection images available. Routed to human review.",
            "vlm_called": False,
            "why_stub": "Set --vlm-backend ollama and ensure Ollama is running with a vision model.",
        })
        return result

    if vlm_backend in ("ollama", "vllm"):
        organ_desc = ORGAN_DESCRIPTIONS.get(organ, "a solid abdominal organ")
        prompt = VLM_PROMPT_TEMPLATE.format(organ=organ, organ_description=organ_desc)
        # Send up to 2 views (axial + coronal) so the VLM sees multiple perspectives.
        images_to_send = png_images[:2]
        result["num_projection_images_sent"] = len(images_to_send)
        if vlm_backend == "vllm":
            vlm_response = _call_vllm(vlm_model, prompt, images_to_send, base_url=vllm_base_url)
        else:
            vlm_response = _call_ollama(vlm_model, prompt, images_to_send)
        result["vlm_raw_response"] = vlm_response

        if vlm_response.get("status") != "success":
            result.update({
                "status": "warning",
                "decision": "review_queue",
                "winner": "uncertain",
                "confidence": 0.0,
                "reason": f"VLM call failed: {vlm_response.get('reason')}. Routed to human review.",
                "vlm_called": True,
            })
            return result

        parsed = _parse_vlm_response(vlm_response.get("raw_response", ""))
        result["vlm_parsed"] = parsed

        winner = parsed.get("winner", "uncertain")
        confidence = parsed.get("confidence", 0.0)

        if winner == "uncertain" or confidence < 0.5:
            decision = "review_queue"
        elif winner == "A":
            decision = "keep_annotation_a"
        else:
            decision = "replace_with_annotation_b"

        result.update({
            "status": "success",
            "decision": decision,
            "winner": winner,
            "confidence": confidence,
            "reason": parsed.get("reason", ""),
            "vlm_called": True,
        })
        return result

    result.update({"status": "failed", "decision": "review_queue",
                   "reason": f"Unknown VLM backend: {vlm_backend}", "vlm_called": False})
    return result
