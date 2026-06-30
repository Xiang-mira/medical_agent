from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROMPT_BANK_CATEGORIES = (
    "simple_instruction",
    "anatomical_location",
    "ct_appearance",
    "synonyms_medical_terms",
    "natural_language",
)


SEMANTICALLY_RISKY_PROMPT_TERMS = (
    "lumen only",
    "gastric lumen",
    "colonic lumen",
    "bowel lumen",
    "vesical lumen",
    "all vessels",
    "all veins",
    "all arteries",
    "vascular tree",
)

SAFE_PROMPT_REPLACEMENTS = {
    "segment the gastric lumen": "segment the stomach organ according to the dataset label definition",
    "segment the colonic lumen": "segment the colon organ / large bowel structure according to the dataset label definition",
    "segment the bowel lumen": "segment the bowel organ region according to the dataset label definition",
    "segment the vesical lumen": "segment the bladder according to the dataset label definition",
    "segment the aortic lumen": "segment the aorta according to the dataset label definition",
}


def sanitize_prompt_text(prompt: str) -> str:
    text = str(prompt or "").strip()
    low = text.lower()
    for bad, replacement in SAFE_PROMPT_REPLACEMENTS.items():
        if bad in low:
            return replacement
    return text


def prompt_semantic_flags(prompt: str) -> list[str]:
    low = str(prompt or "").lower()
    return [term for term in SEMANTICALLY_RISKY_PROMPT_TERMS if term in low]


def filter_semantically_risky_prompts(prompts: list[str]) -> list[str]:
    filtered: list[str] = []
    for prompt in prompts:
        clean = sanitize_prompt_text(str(prompt or "").strip())
        if not clean:
            continue
        if prompt_semantic_flags(clean):
            continue
        if clean not in filtered:
            filtered.append(clean)
    return filtered


CURATED_ORGAN_FACTS: dict[str, dict[str, Any]] = {
    "liver": {
        "aliases": ["liver", "hepatic parenchyma", "hepatic organ"],
        "region": "right upper abdomen, immediately inferior to the diaphragm",
        "landmarks": ["diaphragm", "stomach", "right kidney", "gallbladder"],
        "appearance": "large solid soft-tissue organ with a wedge-like contour and smooth capsule on CT",
    },
    "pancreas": {
        "aliases": ["pancreas", "pancreatic gland", "pancreatic parenchyma"],
        "region": "upper retroperitoneal abdomen, posterior to the stomach",
        "landmarks": ["duodenum", "splenic vessels", "stomach", "portal confluence"],
        "appearance": "elongated lobulated soft-tissue gland extending from the duodenal curve toward the splenic hilum",
    },
    "spleen": {
        "aliases": ["spleen", "splenic organ", "splenic parenchyma"],
        "region": "left upper abdomen beneath the left hemidiaphragm",
        "landmarks": ["stomach", "left kidney", "left ribs", "diaphragm"],
        "appearance": "oval or crescent-shaped homogeneous soft-tissue organ on CT",
    },
    "kidney_left": {
        "aliases": ["left kidney", "left renal parenchyma", "left renal organ"],
        "region": "left retroperitoneum, lateral to the spine and below the spleen",
        "landmarks": ["left adrenal gland", "spleen", "psoas muscle", "renal vessels"],
        "appearance": "bean-shaped retroperitoneal organ with cortex, medulla, and renal sinus on CT",
    },
    "kidney_right": {
        "aliases": ["right kidney", "right renal parenchyma", "right renal organ"],
        "region": "right retroperitoneum, lateral to the spine and inferior to the liver",
        "landmarks": ["right adrenal gland", "liver", "psoas muscle", "renal vessels"],
        "appearance": "bean-shaped retroperitoneal organ with cortex, medulla, and renal sinus on CT",
    },
    "aorta": {
        "aliases": ["aorta", "aortic vessel"],
        "region": "midline to left paraspinal thoracoabdominal course",
        "landmarks": ["heart", "spine", "diaphragm", "iliac bifurcation"],
        "appearance": "long tubular arterial structure with contrast-filled or soft-tissue-density lumen depending on CT phase",
    },
    "inferior_vena_cava": {
        "aliases": ["inferior vena cava", "IVC", "caval vein"],
        "region": "right paraspinal retroperitoneum, anterior to the vertebral bodies",
        "landmarks": ["liver", "right kidney", "aorta", "renal veins"],
        "appearance": "long venous tubular structure, usually right of the aorta on axial CT",
    },
    "stomach": {
        "aliases": ["stomach", "gastric organ"],
        "region": "left upper abdomen between the esophagus and duodenum",
        "landmarks": ["liver", "spleen", "pancreas", "diaphragm"],
        "appearance": "hollow J-shaped or sac-like structure that may contain air, fluid, or oral contrast",
    },
    "gall_bladder": {
        "aliases": ["gallbladder", "gall bladder", "cholecystic sac"],
        "region": "right upper abdomen along the inferior liver surface",
        "landmarks": ["liver", "common bile duct", "duodenum", "right hepatic lobe"],
        "appearance": "small pear-shaped fluid-density sac adjacent to the liver",
    },
    "hepatic_vessel": {
        "aliases": ["hepatic vessel", "hepatic vasculature", "liver vessel"],
        "region": "within the liver and porta hepatis region of the upper abdomen",
        "landmarks": ["liver parenchyma", "portal vein", "hepatic veins", "inferior vena cava"],
        "appearance": "branching tubular vascular structure coursing through or adjacent to liver parenchyma",
    },
    "liver_hepatic_vein": {
        "aliases": ["hepatic vein", "liver hepatic vein", "hepatic venous branch"],
        "region": "within the liver draining toward the inferior vena cava",
        "landmarks": ["liver parenchyma", "inferior vena cava", "right hepatic lobe", "middle hepatic vein"],
        "appearance": "branching venous structure converging toward the inferior vena cava",
    },
    "liver_portal_vein": {
        "aliases": ["portal vein", "hepatic portal vein", "portal venous branch"],
        "region": "porta hepatis and central liver branching into left and right portal veins",
        "landmarks": ["liver parenchyma", "porta hepatis", "bile duct", "hepatic artery"],
        "appearance": "branching portal venous structure entering and dividing within the liver",
    },
    "duodenum": {
        "aliases": ["duodenum", "duodenal bowel", "first small-bowel segment"],
        "region": "upper abdomen curving around the pancreatic head",
        "landmarks": ["pancreatic head", "stomach", "common bile duct", "right kidney"],
        "appearance": "C-shaped bowel loop with variable air, fluid, or contrast on CT",
    },
    "colon": {
        "aliases": ["colon", "large bowel"],
        "region": "peripheral abdomen and pelvis following the large-bowel frame",
        "landmarks": ["small bowel", "abdominal wall", "rectum", "cecum"],
        "appearance": "haustrated bowel structure with variable gas, stool, fluid, or contrast",
    },
    "bladder": {
        "aliases": ["urinary bladder", "bladder"],
        "region": "anterior pelvis, inferior to the peritoneal cavity",
        "landmarks": ["pubic symphysis", "prostate or uterus", "rectum", "pelvic sidewalls"],
        "appearance": "rounded fluid-density pelvic reservoir with a thin wall when distended",
    },
    "heart": {
        "aliases": ["heart", "cardiac silhouette", "cardiac chambers"],
        "region": "middle mediastinum within the thorax",
        "landmarks": ["lungs", "diaphragm", "aorta", "sternum"],
        "appearance": "central soft-tissue cardiac structure containing myocardial wall and blood pool",
    },
    "lung": {
        "aliases": ["lungs", "pulmonary parenchyma", "lung fields"],
        "region": "bilateral thoracic cavities inside the rib cage",
        "landmarks": ["ribs", "mediastinum", "diaphragm", "bronchi"],
        "appearance": "low-attenuation air-filled parenchyma bounded by pleura",
    },
}


ROOT_SYNONYMS = {
    "liver": ["hepatic structure"],
    "hepatic": ["hepatic structure"],
    "kidney": ["renal structure"],
    "renal": ["renal structure"],
    "lung": ["pulmonary structure"],
    "pulmonary": ["pulmonary structure"],
    "heart": ["cardiac structure"],
    "aorta": ["aortic structure"],
    "artery": ["arterial vessel"],
    "vein": ["venous vessel"],
    "vessel": ["vascular structure"],
    "vena_cava": ["caval vein", "IVC"],
    "bladder": ["vesical", "urinary bladder"],
    "gall": ["cholecystic", "gallbladder"],
    "bile": ["biliary"],
    "brain": ["cerebral"],
    "spleen": ["splenic"],
    "pancreas": ["pancreatic"],
    "stomach": ["gastric"],
    "duodenum": ["duodenal"],
    "colon": ["colonic", "large bowel"],
    "intestine": ["bowel"],
    "esophagus": ["esophageal"],
    "adrenal": ["suprarenal"],
    "muscle": ["muscular structure"],
    "bone": ["osseous structure"],
    "cartilage": ["cartilaginous structure"],
    "gland": ["glandular structure"],
}


REGION_RULES = [
    (("brain", "cerebellum", "ventricle", "cortex", "nucleus", "capsule", "eyeball", "eye", "cochlear"), "head and craniofacial region", ["skull", "orbits", "brainstem", "adjacent soft tissues"]),
    (("pharynx", "larynx", "hyoid", "scalene", "carotid", "jugular", "thyroid", "esophagus"), "neck and upper aerodigestive tract", ["airway", "pharynx", "cervical spine", "major neck vessels"]),
    (("lung", "bronch", "airway", "heart", "mediastinum", "aorta", "coronary", "atrial"), "thorax or upper mediastinum", ["lungs", "mediastinum", "heart", "spine"]),
    (("liver", "hepatic", "portal", "spleen", "pancreas", "gall", "bile", "kidney", "adrenal", "duodenum", "stomach", "colon", "intestine", "aorta", "vena", "iliac", "celiac"), "abdomen and retroperitoneum", ["diaphragm", "spine", "bowel", "solid organs"]),
    (("bladder", "prostate", "uterus", "ovary", "rectum", "gonad", "hip", "gluteus"), "pelvis and lower abdomen", ["pelvic bones", "bladder", "rectum", "pelvic sidewalls"]),
    (("femur", "fibula", "tibia", "humerus", "clavicula", "carpal", "metacarpal", "metatarsal", "fingers"), "appendicular skeleton or extremity field of view", ["adjacent bones", "muscles", "subcutaneous fat", "joints"]),
]


def display_name(organ: str) -> str:
    raw = str(organ).replace("postcava", "inferior_vena_cava").strip()
    segment = re.fullmatch(r"liver_segment_(\d+)", raw)
    if segment:
        return f"liver segment {segment.group(1)}"
    parts = raw.replace("(", " ").replace(")", " ").split("_")
    parts = [p for p in parts if p]
    side = None
    if parts and parts[-1] in {"left", "right"}:
        side = parts.pop()
    if parts and parts[0] in {"artery", "vein"}:
        vessel_type = parts.pop(0)
        parts.append(vessel_type)
    if side:
        parts.insert(0, side)
    return " ".join(parts).strip()


def _side_phrase(organ: str) -> str:
    if organ.endswith("_left") or "_left_" in organ:
        return "left-sided "
    if organ.endswith("_right") or "_right_" in organ:
        return "right-sided "
    return ""


def _infer_region_and_landmarks(organ: str) -> tuple[str, list[str]]:
    lower = organ.lower()
    for keys, region, landmarks in REGION_RULES:
        if any(key in lower for key in keys):
            return region, list(landmarks)
    return "expected anatomic field of view on CT", ["nearby organs", "vessels", "bones", "soft-tissue planes"]


def _infer_appearance(organ: str) -> str:
    lower = organ.lower()
    if any(k in lower for k in ("artery", "aorta", "trunk")):
        return "tubular arterial structure following a predictable vascular course"
    if any(k in lower for k in ("vein", "vena", "cava", "portal")):
        return "tubular venous structure, often adjacent to arteries or solid organs"
    if "vessel" in lower:
        return "tubular or branching vascular structure following an expected anatomic course"
    if any(k in lower for k in ("bone", "femur", "fibula", "humerus", "clavicula", "carpal", "metatarsal", "mandible")):
        return "high-attenuation osseous structure with cortical and trabecular components"
    if any(k in lower for k in ("muscle", "scalene", "pterygoid", "rectus", "gluteus", "iliopsoas", "masseter")):
        return "soft-tissue muscle with elongated fiber-like shape and smooth fascial margins"
    if any(k in lower for k in ("airway", "bronch", "trachea")):
        return "air-filled branching tubular structure with thin soft-tissue walls"
    if any(k in lower for k in ("bowel", "colon", "intestine", "stomach", "duodenum", "esophagus")):
        return "hollow gastrointestinal structure with variable air, fluid, or contrast"
    if any(k in lower for k in ("fat", "breast", "mammary")):
        return "predominantly low-attenuation soft tissue or fat-containing structure"
    if any(k in lower for k in ("cartilage", "hyoid")):
        return "cartilaginous or small mineralized structure near surrounding soft tissue"
    return "soft-tissue structure with CT boundaries guided by adjacent anatomy"


def _aliases_for(organ: str, name: str) -> list[str]:
    aliases = [name]
    lower = organ.lower()
    segment = re.fullmatch(r"liver_segment_(\d+)", lower)
    if segment:
        n = segment.group(1)
        return _dedupe([f"hepatic segment {n}", name, f"Couinaud segment {n}", f"liver Couinaud segment {n}"])
    tokens = re.findall(r"[a-z0-9]+", lower)

    def has_root(root: str) -> bool:
        root_tokens = root.split("_")
        if len(root_tokens) == 1:
            return root_tokens[0] in tokens
        return " ".join(root_tokens) in " ".join(tokens)

    for root, terms in ROOT_SYNONYMS.items():
        if has_root(root):
            for term in terms:
                alias = f"{name} ({term})"
                if alias not in aliases:
                    aliases.append(alias)
    compact = name.replace(" gall bladder", " gallbladder").replace("vena cava", "IVC")
    if compact and compact not in aliases:
        aliases.append(compact)
    return aliases[:4]


def _article_for(text: str) -> str:
    return "an" if str(text).strip().lower()[:1] in {"a", "e", "i", "o", "u"} else "a"


def _location_phrase(region: str) -> str:
    region = str(region).strip()
    if region.lower().startswith(("within ", "at ", "on ", "between ", "along ", "inside ")):
        return region
    return f"in the {region}"


def _segment_prompt(alias: str) -> str:
    if re.fullmatch(r"(hepatic|liver|Couinaud|liver Couinaud) segment \d+", str(alias), flags=re.I):
        return f"segment {alias}"
    return f"segment the {alias}"


def _outline_prompt(alias: str) -> str:
    if re.fullmatch(r"(hepatic|liver|Couinaud|liver Couinaud) segment \d+", str(alias), flags=re.I):
        return f"outline {alias}"
    return f"outline the {alias}"


def _find_prompt(alias: str) -> str:
    if re.fullmatch(r"(hepatic|liver|Couinaud|liver Couinaud) segment \d+", str(alias), flags=re.I):
        return f"Find {alias}"
    return f"Find the {alias}"


def _sentence_start(text: str) -> str:
    return str(text)[:1].upper() + str(text)[1:]


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(str(item).split())
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            out.append(clean)
    return out


def build_organ_prompt_entry(organ: str) -> dict[str, Any]:
    name = display_name(organ)
    curated = CURATED_ORGAN_FACTS.get(organ, {})
    aliases = list(curated.get("aliases") or _aliases_for(organ, name))
    region, landmarks = _infer_region_and_landmarks(organ)
    region = str(curated.get("region") or region)
    landmarks = list(curated.get("landmarks") or landmarks)
    appearance = str(curated.get("appearance") or _infer_appearance(organ))
    primary = aliases[0] if aliases else name
    landmark_text = ", ".join(landmarks[:4])
    location_phrase = _location_phrase(region)

    prompts = {
        "simple_instruction": _dedupe([
            _segment_prompt(primary),
            _outline_prompt(primary),
        ]),
        "anatomical_location": _dedupe([
            f"{_sentence_start(_segment_prompt(primary))} {location_phrase}; use nearby landmarks such as {landmark_text}.",
            f"{_find_prompt(primary)} by its expected location {location_phrase}, adjacent to {landmark_text}.",
        ]),
        "ct_appearance": _dedupe([
            f"Identify the {primary} on CT as {_article_for(appearance)} {appearance}.",
            f"Create a mask for the {primary}, following its CT appearance as {_article_for(appearance)} {appearance}.",
        ]),
        "synonyms_medical_terms": _dedupe([
            _segment_prompt(alias) for alias in aliases
        ]),
        "natural_language": _dedupe([
            f"Please delineate the {primary} on this CT volume while respecting its expected shape, location, and boundaries.",
            f"Mark only the voxels belonging to the {primary}; avoid adjacent structures such as {landmark_text}.",
        ]),
    }
    return {
        "canonical_prompt": _segment_prompt(primary),
        "display_name": name,
        "aliases": aliases,
        "region": region,
        "landmarks": landmarks,
        "ct_appearance": appearance,
        "prompts": prompts,
    }


def build_prompt_bank(organs: list[str]) -> dict[str, dict[str, Any]]:
    return {organ: build_organ_prompt_entry(organ) for organ in organs}


def flatten_prompt_bank_entry(entry: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    canonical = sanitize_prompt_text(str(entry.get("canonical_prompt") or "").strip())
    if canonical and not prompt_semantic_flags(canonical):
        prompts.append(canonical)
    for category in PROMPT_BANK_CATEGORIES:
        prompts.extend(str(p).strip() for p in (entry.get("prompts", {}).get(category) or []))
    return _dedupe(filter_semantically_risky_prompts(prompts))


def prompt_category_for(entry: dict[str, Any], prompt: str) -> str:
    if prompt == entry.get("canonical_prompt"):
        return "canonical"
    for category in PROMPT_BANK_CATEGORIES:
        if prompt in (entry.get("prompts", {}).get(category) or []):
            return category
    return "unknown"


def prompt_record_for(entry: dict[str, Any], prompt: str) -> dict[str, Any] | None:
    """Return optional provenance for a prompt without changing the legacy schema."""
    for record in entry.get("prompt_records", []) or []:
        if isinstance(record, dict) and str(record.get("text") or "").strip() == str(prompt).strip():
            return dict(record)
    return None


def select_balanced_prompt_variants(entry: dict[str, Any], *, seed: str) -> list[str]:
    """Select canonical plus one deterministic variant from every prompt category."""
    selected: list[str] = []
    canonical = str(entry.get("canonical_prompt") or "").strip()
    if canonical:
        selected.append(canonical)
    category_prompts = entry.get("prompts", {}) or {}
    for category in PROMPT_BANK_CATEGORIES:
        choices = _dedupe(filter_semantically_risky_prompts([str(x) for x in (category_prompts.get(category) or [])]))
        if not choices:
            continue
        digest = hashlib.sha1(f"{seed}:{category}".encode("utf-8")).hexdigest()
        start = int(digest[:8], 16) % len(choices)
        selected_lower = {item.lower() for item in selected}
        for offset in range(len(choices)):
            prompt = choices[(start + offset) % len(choices)]
            if prompt.lower() not in selected_lower:
                selected.append(prompt)
                break
    return selected


def select_prompt_for_organ(doc: dict[str, Any], organ: str, *, mode: str = "canonical", seed: str | int | None = None) -> str:
    bank = doc.get("organ_prompt_bank", {}) or {}
    entry = bank.get(organ) or {}
    variants = flatten_prompt_bank_entry(entry) if entry else []
    if not variants:
        variants = [str((doc.get("organ_to_prompt", {}) or {}).get(organ) or display_name(organ))]
    if mode in {"random", "sample"}:
        import random

        return random.choice(variants)
    if mode in {"hash", "deterministic"}:
        raw = f"{seed if seed is not None else os.getenv('MEDAI_PROMPT_SEED', '0')}:{organ}".encode("utf-8")
        idx = int(hashlib.sha1(raw).hexdigest()[:8], 16) % len(variants)
        return variants[idx]
    return variants[0]


def load_prompt_bank_description(organ: str, config_path: str | Path | None = None) -> str | None:
    path = Path(config_path) if config_path else None
    if path is None:
        env_path = os.getenv("MEDAI_PROMPT_TARGET_CONFIG")
        path = Path(env_path) if env_path else Path(__file__).resolve().parents[4] / "configs" / "student_3d_prompt_target_organs.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entry = (doc.get("organ_prompt_bank", {}) or {}).get(organ)
    if not isinstance(entry, dict):
        return None
    aliases = ", ".join(entry.get("aliases") or [display_name(organ)])
    landmarks = ", ".join(entry.get("landmarks") or [])
    return (
        "When evaluating and comparing the overlays, consider the following organ-specific CT information:\n"
        f"a) Target names and synonyms: {aliases}.\n"
        f"b) Expected location: {entry.get('region')}.\n"
        f"c) CT appearance: {entry.get('ct_appearance')}.\n"
        f"d) Nearby structures and exclusion landmarks: {landmarks}.\n"
    )
