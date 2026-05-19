# ITK-SNAP sanity-check guide

The teacher asked for only a small number of manual sanity checks, not full manual annotation.

Use this after running the pipeline:

1. Open `outputs/run_pants50_real/review_queue.jsonl`.
2. Pick 1–2 uncertain cases, especially those with low DICE or VLM uncertainty.
3. Open the case CT in ITK-SNAP.
4. Add overlays:
   - original reference mask
   - best candidate mask
   - ShapeKit-refined mask if available
   - final updated annotation
5. Save screenshots into:

```text
outputs/run_pants50_real/itk_snap_examples/
```

Suggested screenshot names:

```text
case_id_organ_original_vs_updated.png
case_id_organ_candidate_comparison.png
```

These screenshots are qualitative evidence. The quantitative evidence remains `dice_metrics.csv` and `round_metrics.csv`.
