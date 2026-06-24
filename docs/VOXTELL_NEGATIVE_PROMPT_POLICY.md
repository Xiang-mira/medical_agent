# VoxTell Negative Prompt Policy

## Core Rule

Do not sample an arbitrary missing organ from the 373-organ target list as a
negative prompt. A missing mask in one case is not proof that the structure is
absent from the CT scan range.

## Allowed Negative Sources

1. `nonmedical_absent_object`
   - Examples: `segment the cat`, `segment the dog`, `segment the car`.
   - Rationale: these are outside the medical CT anatomy label space and test
     whether the prompt-conditioned model suppresses masks for irrelevant text.

2. `out_of_scan_anatomy_with_coverage_evidence`
   - Examples: `segment the head`, `segment the brain`, `segment the skull`.
   - Required evidence: case metadata must indicate an abdomen/pelvis scan
     coverage and must not indicate head, brain, cranial, skull, neck, or
     head-neck coverage.
   - Rationale: an abdominal CT can safely use head/brain/skull prompts as
     out-of-scan anatomy negatives only when scan coverage metadata supports it.

3. `explicit_confirmed_absent_anatomy`
   - Examples: an organ listed in `confirmed_absent_organs`,
     `out_of_scan_organs`, `negative_organs`, or `absent_organs` in case
     metadata.
   - Required evidence: explicit case-level metadata confirming absence.

## Disallowed Sources

The manifest builder no longer creates negative samples from:

- target organs that are simply not selected in the current case;
- low-confidence or rejected teacher outputs;
- anatomically incompatible organ assumptions without scan/metadata evidence;
- previous student empty outputs or retry queues.

## Zero Mask Semantics

A zero mask is only used as the target mask for an allowed negative prompt. A
zero/empty organ mask by itself does not mean the organ is a negative sample,
and it does not prove the organ is absent from the CT scan.

## Pipeline Safeguards

The policy is enforced at three points:

1. Manifest generation only emits allowed negative sources and records
   `negative_evidence`, `negative_prompt_category`, and `zero_mask_role`.
2. The VoxTell prompt-student training loader skips legacy or hand-edited
   negative rows whose source is not allowed, whose zero-mask role is missing,
   or whose anatomy-based negative lacks evidence.
3. Round1 negative-suppression evaluation samples only allowed negative rows,
   so teacher-facing suppression metrics are not computed from unsafe legacy
   negatives.
