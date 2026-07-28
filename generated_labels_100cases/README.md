# Generated Labels 100 Cases

This directory is the final overlay target for validated generated labels.

It is intentionally empty in Git until the fixed 100-case manifest, confirmed generate-organ table, Teacher inference, validation, and packaging steps are run on HPC.

Preview merge:

```bash
rsync -av --ignore-existing --dry-run \
  generated_labels_100cases/ \
  /official/AbdomenAtlasPro/
```

Final merge:

```bash
rsync -av --ignore-existing \
  generated_labels_100cases/ \
  /official/AbdomenAtlasPro/
```
