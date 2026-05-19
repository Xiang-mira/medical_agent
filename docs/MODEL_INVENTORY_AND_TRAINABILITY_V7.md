# Teacher Model Inventory and Trainability (v7)

The current registry contains **18 teacher-provided or teacher-referenced model families**, excluding the synthetic `mock_seg` dry-run backend.

## Directly integrated model families

These have source folders, lightweight Drive exports, model cards, or wrappers integrated in the project.

| Model key | Source material | Current runnable status | Trainable in this project? | Why |
|---|---|---|---|---|
| `epai_20250421` | ePAI source + qchen76_2025_0421 Dataset1017 | Ready if teacher checkpoint folder is mounted | Yes, if nnUNet-compatible checkpoint/training state is present | Verified 25-class abdominal organ/duct/tumor nnUNet-style model; preferred M-step target for PanTS pancreas tasks and a candidate for related organ refinement. |
| `cads` | CADS_series + class_checkpoint_map | Ready if CADS checkpoint folder is mounted | Yes, conditional | Exposed as nnUNet-style Dataset551; can be fine-tuned if full training/checkpoint state exists. |
| `moose` | MOOSE_series + class_checkpoint_map | Partially ready if checkpoint folder is mounted | Yes, conditional | Current lightweight export exposes limited MOOSE datasets; trainability depends on full nnUNet training state. |
| `moose3_0` | MOOSE_series + class_checkpoint_map | Partially ready if checkpoint folder is mounted | Yes, conditional | Same as MOOSE; use only when relevant dataset/checkpoint is available. |
| `vsmtrans` | VSmTrans lightweight export + class_checkpoint_map | Ready if VSmTrans/nnUNet results are mounted | Yes, conditional | Routed to abdominal organs; fine-tuning depends on full nnUNet-compatible state. |
| `nnunet_private` | Teacher Drive top-level nnUNet_private folder | Ready if Dataset224 files are mounted | Yes, conditional | Private AbdomenAtlas-style organ backend; suitable trainable M-step backend if full state exists. |
| `saros_nnunet` | nnUNet_private + class_checkpoint_map | Ready if Dataset1345 files are mounted | Yes, conditional | Private nnUNet-style SAROS model; not first-choice for PanTS pancreas but trainable if files exist. |
| `atlasnet` | “Another version of ShapeKit” model card | Ready if downloaded ATLAS-Net weights are present | Yes, conditional | ATLAS-Net is an nnUNet v2 abdominal 25-class model; training/fine-tuning needs compatible weights/training files. |
| `totalsegmentator` | TotalSegmentator source/package | Ready if installed | No, not in this project | Used as public baseline/E-step candidate. Retraining released TotalSegmentator would require its full upstream training recipe. |
| `vista3d` | VISTA3D inference pipeline | Ready if VISTA3D env/checkpoint exists | No, not with current wrapper | Integrated as foundation inference candidate; training requires MONAI/VISTA3D training recipe. |
| `unest` | UNEST lightweight run script | Ready if UNEST env/checkpoint exists | No, external script required | Kidney cortex/medulla inference candidate; no training wrapper included. |

## Template-only model families from class_checkpoint_map.xlsx

These names appear in the teacher-provided class map, but no complete runnable script/checkpoint folder was available in the lightweight export. They are kept as registry templates so they can be activated later.

| Model key | Status | Trainable? | Reason |
|---|---|---|---|
| `airrc` | Template | Unknown / no | Appears in class map; no complete inference/training script. |
| `atm` | Template | Unknown / no | Appears in class map; no complete inference/training script. |
| `dap` | Template | Unknown / no | Appears in class map; no complete inference/training script. |
| `duke` | Template | Unknown / no | Appears in class map; no complete inference/training script. |
| `goacc` | Template | Unknown / no | Appears in class map; no complete inference/training script. |
| `pedro` | Template | Unknown / no | Appears in class map; no complete inference/training script. |
| `vsnet` | Template | Unknown / no | Appears in class map; no complete inference/training script. |

## Practical routing policy

| Task / organ group | Primary model | Auxiliary candidates | M-step target |
|---|---|---|---|
| Pancreas / pancreatic duct / pancreatic tumor | `epai_20250421` | `atlasnet`, `vsmtrans`, `cads`, `moose3_0`, `totalsegmentator`, `vista3d` | `epai_20250421` if checkpoint is compatible. |
| Broad abdominal organs | `vsmtrans` | `cads`, `moose3_0`, `atlasnet`, `totalsegmentator`, `vista3d`, `nnunet_private` | `vsmtrans` or `cads`, depending on available checkpoint. |
| Aorta / vascular structures | `cads` | `vista3d`, `moose3_0`, `atlasnet`, `vsmtrans`, `totalsegmentator` | `cads` if checkpoint is compatible. |
| Kidney cortex / medulla | `unest` | `nnunet_private`, `cads`, `vsmtrans` | No current UNEST M-step; use external training script or compatible nnUNet fallback. |

Use `python run_medai_cli.py --json model-inventory` and `python run_medai_cli.py --json route-models --organs ...` to inspect this programmatically.
