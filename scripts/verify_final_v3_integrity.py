#!/usr/bin/env python3
from pathlib import Path
import json, sys, yaml

required = [
 'third_party/ePAI-main/README.md',
 'third_party/ePAI-main/backup_model/3D-TransUNet/inference.py',
 'third_party/VISTA3D-Inference-Pipeline-master/README.md',
 'third_party/TotalSegmentator-master/README.md',
 'third_party/UNEST_renalStructures_lightweight/run_UNEST.sh',
 'third_party/VSmTrans_lightweight/README.md',
 'third_party/LabelCritic-main/CompareOrgan.py',
 'third_party/ShapeKit-main/main.py',
 'configs/atlasnet_label_map.json',
 'scripts/patch_epai_enable_all_organs.py',
 'scripts/atlasnet_predict_and_split.py',
 'scripts/vista3d_predict_and_split.py',
 'scripts/unest_predict_and_split.py',
 'docs/EPAI_ALL_ORGAN_OUTPUT_PATCH.md',
 'docs/FINAL_DELIVERABLE_V3_STATUS.md',
]
missing = [p for p in required if not Path(p).exists()]
patch_text = Path('third_party/ePAI-main/backup_model/3D-TransUNet/inference.py').read_text(encoding='utf-8') if Path('third_party/ePAI-main/backup_model/3D-TransUNet/inference.py').exists() else ''
export_text = Path('third_party/ePAI-main/binary/nnunetv2/inference/export_prediction.py').read_text(encoding='utf-8') if Path('third_party/ePAI-main/binary/nnunetv2/inference/export_prediction.py').exists() else ''
all_organ_patch_present = 'output_label_mode' in patch_text and 'all_organs' in patch_text and 'EPAI_OUTPUT_LABEL_MODE' in export_text
registry_ok = False
try:
    reg = yaml.safe_load(Path('configs/model_registry.yaml').read_text(encoding='utf-8'))
    models = reg.get('models', {})
    registry_ok = all(k in models for k in ['epai_20250421', 'atlasnet', 'vista3d', 'unest', 'totalsegmentator'])
except Exception:
    registry_ok = False
status = 'success' if not missing and all_organ_patch_present and registry_ok else 'failed'
print(json.dumps({'status': status, 'missing': missing, 'epai_all_organ_patch_present': all_organ_patch_present, 'registry_models_ok': registry_ok}, indent=2))
sys.exit(0 if status == 'success' else 1)
