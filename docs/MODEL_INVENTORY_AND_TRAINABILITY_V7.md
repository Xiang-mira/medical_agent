# Teacher Model Inventory and HF Release Status

The current HF release at `https://huggingface.co/Xiang-mira/MedIA-Agentic-AI` contains 23 teacher models and one VoxTell-style student checkpoint. `mock_seg` and `epai_finetuned` are intentionally excluded from the HF release.

## Released teacher models

| Model key | Backend | HF path | Checkpoint | Organs | Notes |
|---|---|---|---|---:|---|
| `cads551` | `nnunetv2` | `teacher_models/cads551` | `fold_all/checkpoint_final.pth` | 17 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads552` | `nnunetv2` | `teacher_models/cads552` | `fold_all/checkpoint_final.pth` | 24 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads553` | `nnunetv2` | `teacher_models/cads553` | `fold_all/checkpoint_final.pth` | 18 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads554` | `nnunetv2` | `teacher_models/cads554` | `fold_all/checkpoint_final.pth` | 21 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads555` | `nnunetv2` | `teacher_models/cads555` | `fold_all/checkpoint_final.pth` | 24 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads556` | `nnunetv2` | `teacher_models/cads556` | `fold_all/checkpoint_final.pth` | 15 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads557` | `nnunetv2` | `teacher_models/cads557` | `fold_all/checkpoint_final.pth` | 9 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads558` | `nnunetv2` | `teacher_models/cads558` | `fold_all/checkpoint_final.pth` | 29 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `cads559` | `nnunetv2` | `teacher_models/cads559` | `fold_all/checkpoint_final.pth` | 10 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `moose666` | `nnunetv2` | `teacher_models/moose666` | `fold_all/checkpoint_final.pth` | 31 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `moose888` | `nnunetv2` | `teacher_models/moose888` | `fold_all/checkpoint_final.pth` | 13 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `nnunet_private` | `nnunetv2` | `teacher_models/nnunet_private` | `fold_all/checkpoint_final.pth` | 34 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `saros_nnunet` | `nnunetv2` | `teacher_models/saros_nnunet` | `fold_all/checkpoint_final.pth` | 13 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `atm` | `nnunetv2` | `teacher_models/atm` | `fold_all/checkpoint_final.pth` | 1 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `airrc` | `nnunetv2` | `teacher_models/airrc` | `fold_all/checkpoint_final.pth` | 4 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `lvp` | `nnunetv2` | `teacher_models/lvp` | `fold_all/checkpoint_final.pth` | 2 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `daps` | `nnunetv2` | `teacher_models/daps` | `fold_all/checkpoint_best.pth` | 30 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `epai_20250421` | `nnunetv2` | `teacher_models/epai_20250421` | `fold_all/checkpoint_final.pth` | redacted | Sensitive internal research asset. Public metadata is redacted; runtime internals and full private label semantics are intentionally omitted. |
| `vsmtrans` | `nnunetv2` | `teacher_models/vsmtrans` | `fold_0/checkpoint_final.pth` | 25 | Internal research collaboration asset. Do not externally advertise as a public JHU model. |
| `vista3d` | `vista3d` | `teacher_models/vista3d` | `models/model.pt` | 99 | Reusable research asset; backend-specific runtime required. |
| `unest` | `unest` | `teacher_models/unest` | `models/model.pt` | 3 | Reusable research asset; backend-specific runtime required. |
| `totalsegmentator` | `external_totalsegmentator_runtime` | `teacher_models/totalsegmentator` | `null` | 121 | External public runtime. Official TotalSegmentator weights are not mirrored in this HF asset repo. |
| `atlasnet` | `atlasnet_wrapper_over_nnunetv2` | `teacher_models/atlasnet` | `fold_all/checkpoint_final.pth` | 25 | Public ATLAS-Net-derived teacher route adapted to the MedIA output contract. |


## Released student model

| Model key | Backend | HF path | Checkpoint | Notes |
|---|---|---|---|---|
| `voxtell_style_student_round1` | `voxtell_style_3d_prompt` | `student_models/voxtell_style_student_round1` | `voxtell_finetuned_model/fold_0/checkpoint_final.pth` | Project prompt-distillation student initialized from official VoxTell assets; not official VoxTell finetuning. |

## Excluded entries

| Entry | Reason |
|---|---|
| `mock_seg` | Synthetic dry-run backend with no real weights. |
| `epai_finetuned` | Local experimental M-step output; not a current reusable HF release. |

Use `configs/hf_model_manifest.yaml` for download paths, commands, supported organs, and expected GPU memory.
