# BDMAP / PanTS-Style Data Layout

## Source Layout

The project now uses a case-level source dataset layout modeled after the local
PanTS tools:

```text
case_xxx/
  image.nii.gz
  segmentations/
    liver.nii.gz
    pancreas.nii.gz
    kidney_left.nii.gz
    ...
  label_mapping.json
  selection_metadata.json
```

PanTS itself stores CT images as `ImageTr/<case>/ct.nii.gz` and labels as
`LabelTr/<case>/segmentations/*.nii.gz`. Our standardized project export keeps
the same one-case / one-segmentation-folder idea, while naming the CT
`image.nii.gz` for model-agnostic training input.

## Mapping Layers

Teacher labels are not reused directly as student targets. Every mask is routed
through three explicit layers:

1. teacher target ID / teacher output name;
2. canonical organ name;
3. student target ID from `configs/student_3d_prompt_target_organs.json`.

This mapping is written to `label_mapping.json` per case. If a teacher target ID
is unavailable, it is recorded as `null` rather than fabricated.

## Training Conversion

The source dataset remains one CT plus one binary mask per organ. The VoxTell
prompt-student manifest builder then converts it into model-specific samples:

```text
CT image + free-text prompt -> binary organ mask
```

Example:

```text
case_xxx/image.nii.gz
prompt: "segment the pancreas"
target: case_xxx/segmentations/pancreas.nii.gz
```

This keeps the raw dataset organization close to PanTS/BDMAP conventions while
allowing the data loader to produce prompt-conditioned training examples.

