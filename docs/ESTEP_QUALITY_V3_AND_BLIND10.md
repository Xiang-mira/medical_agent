# E-step quality contract v3 and blind-10 protocol

## Formal sequence

1. Run the 20-case E-step and require `round1/estep/formal_gate.json` to pass.
2. Train the prompt student on all eligible labels from all 20 cases.
3. Freeze the quality-gated `inference_model_dir`.
4. Run `blind10_batch1.csv` in three isolated phases:

```bash
python scripts/run_blind10_protocol.py student \
  --student-model-dir /path/to/voxtell_finetuned_model
python scripts/run_blind10_protocol.py agent
python scripts/run_blind10_protocol.py evaluate
```

The case list contains only `case_id,ct_path`. The evaluate phase refuses to
run until both inference phases have completed and fingerprints predictions
before opening any reference path.

## Quality contract

- FOV comes from CT and independent model landmarks, never `annotation_folder`.
- Geometry mismatch is a hard candidate failure.
- `postcava` is an input alias of canonical `inferior_vena_cava`.
- D labels are moved to the rejected audit area and are never published or
  distilled.
- C labels require a soft target and no hard QC flags.
- High-risk ShapeKit failure blocks ducts, vessels and lesions.
- Pancreatic duct uses dedicated volume, component, Dice, HD95 and parent
  containment checks.

## Reporting language

PanTS LabelTr is treated as `historical_pseudo`. Results are reported as
held-out reference agreement, not expert or clinical accuracy.
