from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def format_bullets(items):
    if not items:
        return '- none provided'
    return '\n'.join(f'- {x}' for x in items)


def build_labelcreator_prompt(entry: Dict[str, Any], case_id: str, candidate_table: str, mode: str = 'ranking') -> str:
    aliases = ', '.join(entry.get('aliases') or [entry.get('display_name', entry['canonical_organ'])])
    fields = {
        'case_id': case_id,
        'canonical_organ': entry['canonical_organ'],
        'display_name': entry.get('display_name', entry['canonical_organ'].replace('_',' ')),
        'aliases': aliases,
        'expected_region': entry.get('expected_region',''),
        'landmarks': ', '.join(entry.get('landmarks', [])) or 'visible anatomical landmarks',
        'ct_appearance': entry.get('ct_appearance',''),
        'shape_and_continuity_prior': entry.get('shape_and_continuity_prior',''),
        'anatomical_constraints': format_bullets(entry.get('anatomical_constraints', [])),
        'common_failure_modes': format_bullets(entry.get('common_failure_modes', [])),
        'rejection_rules': format_bullets(entry.get('rejection_rules', [])),
        'candidate_table': candidate_table,
    }
    if mode == 'single':
        return SINGLE_CANDIDATE_TEMPLATE.format(**fields)
    return RANKING_TEMPLATE.format(**fields)


RANKING_TEMPLATE = """You are LabelCritic, a CT segmentation-label quality judge. I will send AP-like frontal CT projections and candidate mask overlays for the same CT case.

Case ID: {case_id}
Target organ/structure: {canonical_organ} ({display_name})
Aliases: {aliases}

Organ-specific CT prior:
- Expected region: {expected_region}
- Landmarks: {landmarks}
- CT appearance: {ct_appearance}
- Shape and continuity prior: {shape_and_continuity_prior}

Anatomical constraints:
{anatomical_constraints}

Common failure modes to check:
{common_failure_modes}

Rejection / downgrade rules:
{rejection_rules}

Candidate metadata:
{candidate_table}

Task:
1. Evaluate each candidate overlay individually.
2. Check whether the candidate is actually the requested target, not a neighboring organ or wrong side.
3. Compare candidates using location, CT appearance, shape, continuity, boundary plausibility, landmark relationship, leakage, disconnected components, and over/under-segmentation.
4. Prefer the anatomically most plausible candidate. Do not prefer a candidate only because it has larger foreground area or because its teacher family has more checkpoints.
5. If all candidates are clearly poor, return no_acceptable_candidate.
6. If the visual evidence is insufficient, return uncertain_needs_review rather than forcing a confident selection.

Return strict JSON only with these keys: target, decision, best_candidate_id, acceptable, confidence, ranking, selected_reason, rejected_reasons, failure_modes, should_enter_student_training, recommended_grade, review_needed.
"""

SINGLE_CANDIDATE_TEMPLATE = """You are LabelCritic, a CT segmentation-label quality judge. I will send one AP-like CT projection with one candidate mask overlay.

Case ID: {case_id}
Target organ/structure: {canonical_organ} ({display_name})
Aliases: {aliases}
Expected region: {expected_region}
Landmarks: {landmarks}
CT appearance: {ct_appearance}
Shape/continuity prior: {shape_and_continuity_prior}

Check the candidate against these constraints:
{anatomical_constraints}

Common failures:
{common_failure_modes}

Return strict JSON only with these keys: target, quality, confidence, is_requested_target, main_evidence_for, main_evidence_against, failure_modes, recommended_grade, should_enter_student_training, review_needed.
"""
