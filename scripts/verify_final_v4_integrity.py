#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import yaml

required = [
    'third_party/ePAI-main/README.md',
    'third_party/ePAI-main/backup_model/3D-TransUNet/inference.py',
    'third_party/VISTA3D-Inference-Pipeline-master/README.md',
    'third_party/TotalSegmentator-master/README.md',
    'third_party/UNEST_renalStructures_lightweight/run_UNEST.sh',
    'third_party/VSmTrans_lightweight/README.md',
    'third_party/LabelCritic-main/CompareOrgan.py',
    'third_party/LabelCritic-main/ProjectDatasetFlex_single.py',
    'third_party/LabelCritic-main/projection.py',
    'third_party/ShapeKit-main/main.py',
    'configs/atlasnet_label_map.json',
    'scripts/patch_epai_enable_all_organs.py',
    'scripts/atlasnet_predict_and_split.py',
    'scripts/vista3d_predict_and_split.py',
    'scripts/unest_predict_and_split.py',
    'agent-harness/cli_anything/medai/core/labelcritic_projection_runner.py',
    'agent-harness/cli_anything/medai/core/projection_builder.py',
    'docs/EPAI_ALL_ORGAN_OUTPUT_PATCH.md',
    'docs/FINAL_DELIVERABLE_V4_STATUS.md',
    'docs/LABELCRITIC_PROJECTION_UPDATE.md',
    'docs/TASK_COMPLETION_MATRIX_V4.md',
]
missing = [p for p in required if not Path(p).exists()]

patch_text = Path('third_party/ePAI-main/backup_model/3D-TransUNet/inference.py').read_text(encoding='utf-8') if Path('third_party/ePAI-main/backup_model/3D-TransUNet/inference.py').exists() else ''
export_text = Path('third_party/ePAI-main/binary/nnunetv2/inference/export_prediction.py').read_text(encoding='utf-8') if Path('third_party/ePAI-main/binary/nnunetv2/inference/export_prediction.py').exists() else ''
projection_text = Path('agent-harness/cli_anything/medai/core/projection_builder.py').read_text(encoding='utf-8') if Path('agent-harness/cli_anything/medai/core/projection_builder.py').exists() else ''
runner_text = Path('agent-harness/cli_anything/medai/core/labelcritic_projection_runner.py').read_text(encoding='utf-8') if Path('agent-harness/cli_anything/medai/core/labelcritic_projection_runner.py').exists() else ''

all_organ_patch_present = 'output_label_mode' in patch_text and 'all_organs' in patch_text and 'EPAI_OUTPUT_LABEL_MODE' in export_text
labelcritic_projection_present = all(s in projection_text + runner_text for s in ['ProjectDatasetFlex_single.py', 'projection_backend', 'LabelCritic'])
registry_ok = False
registry_models = []
try:
    reg = yaml.safe_load(Path('configs/model_registry.yaml').read_text(encoding='utf-8'))
    models = reg.get('models', {})
    registry_models = sorted(models.keys())
    registry_ok = all(k in models for k in ['epai_20250421', 'atlasnet', 'vista3d', 'unest', 'totalsegmentator', 'cads', 'moose3_0', 'vsmtrans'])
except Exception:
    registry_ok = False

status = 'success' if not missing and all_organ_patch_present and labelcritic_projection_present and registry_ok else 'failed'
print(json.dumps({
    'status': status,
    'missing': missing,
    'epai_all_organ_patch_present': all_organ_patch_present,
    'labelcritic_projection_present': labelcritic_projection_present,
    'registry_models_ok': registry_ok,
    'registry_models_checked': registry_models,
}, indent=2, ensure_ascii=False))
sys.exit(0 if status == 'success' else 1)
