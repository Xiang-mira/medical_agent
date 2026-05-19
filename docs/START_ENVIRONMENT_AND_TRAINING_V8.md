# Start Environment and Training Guide (v8)

## Purpose

Use this guide after uploading the final zip to Colab or a Linux server. The project does not include private checkpoint weights or PanTS NIfTI data; you must mount/link them at runtime.

## Step 1. Mount Google Drive in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2. Unzip and enter project

```bash
!unzip -q /content/medai_agent_loop_task2_final_v8_ready.zip -d /content/
%cd /content/medai_agent_loop_task2_final_v8
```

## Step 3. Link checkpoint folders without copying private weights

```bash
python scripts/prepare_colab_checkpoint_links.py \
  --drive-checkpoints /content/drive/MyDrive/checkpoints \
  --project-root . \
  --overwrite
```

This creates symlinks:

```text
checkpoints/CADS_series -> /content/drive/MyDrive/checkpoints/CADS_series
checkpoints/MOOSE_series -> /content/drive/MyDrive/checkpoints/MOOSE_series
checkpoints/nnUNet_private -> /content/drive/MyDrive/checkpoints/nnUNet_private
checkpoints/UNEST -> /content/drive/MyDrive/checkpoints/UNEST
checkpoints/VSmTrans -> /content/drive/MyDrive/checkpoints/VSmTrans
```

It does not copy, delete, or modify teacher checkpoint files.

## Step 4. Install dependencies

For safe base install:

```bash
pip install -r agent-harness/requirements.txt
```

For real inference/training on GPU, install in this exact order to avoid ePAI/nnunetv2 conflicts:

```bash
# 1. ePAI's modified nnunetv2 fork — MUST come first
#    It registers nnUNetv2_predict_from_modelfolder with ePAI-specific args
#    (--input_csv, --output_csv, --output_label_mode) that the standard PyPI
#    package does not have.
pip install -e third_party/ePAI-main/train

# 2. TotalSegmentator and standard nnunetv2 WITHOUT overwriting the ePAI fork
pip install TotalSegmentator --no-deps
pip install nnunetv2 --no-deps

# 3. Fill in any remaining missing deps as reported at runtime
pip install batchgeneratorsv2 acvl-utils dynamic-network-architectures blosc2
```

If Colab creates a NumPy conflict, pin a compatible version and restart runtime:

```bash
pip install --force-reinstall "numpy==2.0.2"
```

Then restart runtime and re-enter the project directory.

## Step 5. Verify project

```bash
python scripts/verify_final_v8_integrity.py
python run_medai_cli.py --json model-inventory
python run_medai_cli.py --json route-models --organs pancreas,liver,aorta,pancreatic_duct,kidney_cortex
```

## Step 6. Download PanTS data and select 50 tumor cases

### Disk space management (critical for 50 GB data disks, e.g. AutoDL)

PanTS images are packed in 1000-case tar blocks (~34 GB each). You cannot download
individual cases — you must download the whole block and extract only what you need.
Run the two download phases **sequentially** so the image tar is deleted before the
label tar arrives:

| Phase | What happens | Peak disk | After tar deleted |
|-------|-------------|-----------|-------------------|
| 1 — images | download ~34 GB tar, extract 50 cases, delete tar | ~44 GB | ~10 GB |
| 2 — labels | download ~15 GB tar, extract 50 cases, delete tar | ~25 GB | ~15 GB |

**Never run both phases at the same time on a 50 GB disk.**

### For China/AutoDL servers — set HuggingFace mirror first

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Phase 1: download images (run inside tmux to survive SSH disconnects)

```bash
tmux new -s pants_dl   # or attach: tmux attach -t pants_dl
export HF_ENDPOINT=https://hf-mirror.com   # China servers only

python scripts/download_pants50_selective.py \
  --pants-root third_party/PanTS-main \
  --download-images \
  --yes
# Script prints free disk at start and again after tar is deleted.
# Ends with: "[phase-1] Image extraction complete. Disk reclaimed."
```

### Phase 2: download labels (after Phase 1 is confirmed complete)

```bash
python scripts/download_pants50_selective.py \
  --pants-root third_party/PanTS-main \
  --download-labels \
  --yes
# Ends with: "[phase-2] Label extraction complete. Final disk usage ~15 GB."
```

### Select the 50 tumor cases and validate

```bash
python run_medai_cli.py --json pants-select-50 \
  --pants-root third_party/PanTS-main \
  --output data_manifest/case_list_50_tumor.csv \
  --split train \
  --num-cases 50

python scripts/validate_case_list_50.py --case-list data_manifest/case_list_50_tumor.csv
```

## Step 7. Run inference/refinement loop

Start with 1–2 cases first. Do not run all models at once until paths are verified.

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor.csv \
  --models epai_20250421,cads,vsmtrans,totalsegmentator \
  --organs pancreas,pancreatic_duct,liver,spleen,duodenum,colon,stomach,aorta,postcava \
  --output outputs/run_pants50_round1 \
  --enable-shapekit \
  --enable-critic \
  --critic-backend labelcritic \
  --critic-base-url http://localhost \
  --critic-port 8000
```

For offline dry-run only:

```bash
python run_medai_cli.py --json run-loop \
  --case-list data_manifest/case_list_50_tumor_template.csv \
  --models mock_seg \
  --organs pancreas,liver,aorta \
  --output outputs/dry_run \
  --dry-run \
  --critic-backend stub
```

## Step 8. Selected-model-aware M-step

For pancreas, the preferred M-step target is ePAI:

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_round1/training_manifest.json \
  --output-folder outputs/mstep_epai_round1 \
  --target-model epai_20250421 \
  --ct-source-root third_party/PanTS-main/data \
  --max-epochs 5
```

For a model that is not trainable in this project, such as TotalSegmentator, the command will explicitly return `not_trainable_in_current_project`.

## Step 9. Build teacher-facing outputs

```bash
python run_medai_cli.py --json build-samples \
  --run-output outputs/run_pants50_round1 \
  --output-jsonl outputs/run_pants50_round1/per_case_samples.jsonl \
  --organs pancreas,pancreatic_duct,liver,spleen,duodenum,colon,stomach,aorta,postcava

python run_medai_cli.py --json convergence-table \
  --round-csvs outputs/run_pants50_round1/round_metrics.csv

python run_medai_cli.py --json itksnap-review \
  --review-queue outputs/run_pants50_round1/review_queue.jsonl \
  --output-script outputs/run_pants50_round1/open_itksnap_cases.sh
```


## V9 additional optional M-step backends

TotalSegmentator and VISTA3D are now selectable conditional M-step targets. They should not replace ePAI as the default PanTS pancreas target, but they can be intentionally selected when their training/fine-tuning environment is ready.

### TotalSegmentator-style public nnUNet M-step

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_totalseg_round1 \
  --target-model totalsegmentator \
  --ct-source-root third_party/PanTS-main/data \
  --dry-run
```

This uses the bundled public recipe in `third_party/TotalSegmentator-master/resources/`. It does not claim to reproduce the released TotalSegmentator v2 model.

### VISTA3D MONAI bundle fine-tuning

```bash
python run_medai_cli.py --json mstep-update \
  --training-manifest outputs/run_pants50_real/training_manifest.json \
  --output-folder outputs/mstep_vista3d_round1 \
  --target-model vista3d \
  --ct-source-root third_party/PanTS-main/data \
  --pretrained-weights /path/to/vista3d_checkpoint.pt \
  --dry-run
```

This uses `configs/train.json` and related MONAI bundle configs under `third_party/VISTA3D-Inference-Pipeline-master/`. Real fine-tuning requires MONAI bundle dependencies, a VISTA3D checkpoint, and enough GPU memory.
