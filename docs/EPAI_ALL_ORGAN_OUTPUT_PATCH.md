# ePAI 2025-04-21 all-organ output note

## Verified checkpoint

The teacher-specified checkpoint is:

```bash
wget http://www.cs.jhu.edu/~zongwei/model/qchen76_2025_0421.tar.gz
tar -xzvf qchen76_2025_0421.tar.gz
```

Its `dataset.json` is `Dataset1017_ePAI_3MM` and defines background plus 25 foreground labels:

```text
1  aorta
2  adrenal_gland_left
3  adrenal_gland_right
4  common_bile_duct
5  celiac_aa
6  colon
7  duodenum
8  gall_bladder
9  postcava
10 kidney_left
11 kidney_right
12 liver
13 pancreas
14 pancreatic_duct
15 superior_mesenteric_artery
16 intestine
17 spleen
18 stomach
19 veins
20 renal_vein_left
21 renal_vein_right
22 cbd_stent
23 pancreatic_pdac
24 pancreatic_cyst
25 pancreatic_pnet
```

So the 2025-04-21 checkpoint is not pancreas-only. It is the 20+ organ plus pancreatic duct/tumor checkpoint the teacher referred to.

## The source-code switch

The ePAI source still contains pancreas/tumor-only export filters in these paths:

```text
backup_model/3D-TransUNet/inference.py
binary/nnunetv2/inference/export_prediction.py
train/nnunetv2/inference/export_prediction.py
```

Those blocks keep only labels `[13, 14, 23, 24, 25]` unless disabled. MedAI patches them so the default behavior is:

```bash
--output_label_mode all_organs
EPAI_OUTPUT_LABEL_MODE=all_organs
```

The original behavior is still available for debugging:

```bash
--output_label_mode pancreas_only
EPAI_OUTPUT_LABEL_MODE=pancreas_only
```

## MedAI wrapper behavior

For the 2025-04-21 checkpoint, MedAI uses the model-folder entrypoint from ePAI:

```bash
python scripts/nnunetv2_predict_and_split.py \
  --image <ct.nii.gz> \
  --output <case_output> \
  --dataset-id 1017 \
  --nnunet-results checkpoints \
  --dataset-json checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json \
  --model-folder checkpoints/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --workdir third_party/ePAI-main/train \
  --trainer nnUNetTrainer \
  --plans nnUNetPlans \
  --configuration 3d_fullres \
  --folds all \
  --checkpoint-name checkpoint_final.pth \
  --output-label-mode all_organs
```

The wrapper then splits the combined label map into:

```text
case_id/segmentations/*.nii.gz
```

The private checkpoint itself is not bundled in the source zip. Extract it under `checkpoints/qchen76_2025_0421/` or create a read-only link/junction with the same layout.

