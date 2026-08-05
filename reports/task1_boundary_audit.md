# Task 1 Boundary Audit

- status: `success`
- mapping rows: `107`
- confirmed: `105`
- pending_review: `2`
- task2_generate: `0`
- exclude: `0`
- Task 2 overlap: `0`
- many-to-one alias groups: `8`

## High Risk Decisions
- `celiac_truck -> celiac_trunk`: `task1_rename/confirmed`; spelling_typo; evidence: manual exact-equivalence review: celiac_truck is a spelling typo for celiac_trunk; same celiac trunk class; no voxel operation required
- `celiac_aa -> celiac_aa_celiac_artery`: `task1_rename/confirmed`; synonym_format; evidence: taxonomy target spells the same class as celiac_aa (celiac_artery); same anatomy, laterality, and granularity; no voxel operation required
- `celiac_artery -> celiac_aa_celiac_artery`: `task1_rename/confirmed`; synonym_format; evidence: taxonomy target spells the same class as celiac_aa (celiac_artery); same anatomy, laterality, and granularity; no voxel operation required
- `parotid_gland -> parotid_glands`: `boundary_review/pending_review`; insufficient_evidence_single_plural_aggregate; evidence: taxonomy has left/right parotid glands and an aggregate parotid_glands class; source singular lacks repository evidence proving bilateral aggregate equivalence
- `submandibular_gland -> submandibular_glands`: `boundary_review/pending_review`; insufficient_evidence_single_plural_aggregate; evidence: taxonomy has left/right submandibular glands and an aggregate submandibular_glands class; source singular lacks repository evidence proving bilateral aggregate equivalence
