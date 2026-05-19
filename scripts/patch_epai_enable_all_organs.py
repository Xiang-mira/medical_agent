#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ePAI-main"

def patch_backup(path: Path):
    text = path.read_text(encoding="utf-8")
    if "--output_label_mode" not in text:
        text = text.replace('parser.add_argument("--save_npz", default=False, action="store_true")',
                            'parser.add_argument("--save_npz", default=False, action="store_true")\nparser.add_argument("--output_label_mode", choices=["all_organs", "pancreas_only"], default="all_organs", help="all_organs keeps all predicted labels; pancreas_only preserves original ePAI filtering to labels 13/14/23/24/25.")')
    text = text.replace('pancreas_and_tumor_labels = [13,14,23,24,25]\n        segmentation_final_pancreas_and_tumor = np.zeros_like(segmentation_final)',
                        'if getattr(args, "output_label_mode", "all_organs") == "pancreas_only":\n            pancreas_and_tumor_labels = [13,14,23,24,25]\n            segmentation_final_pancreas_and_tumor = np.zeros_like(segmentation_final)')
    path.write_text(text, encoding="utf-8")

def patch_export(path: Path):
    text = path.read_text(encoding="utf-8")
    if 'EPAI_OUTPUT_LABEL_MODE' not in text:
        text = text.replace('        # NOTE: STEP 0 ==> extract only pancreas and tumor\n        # print("\\\\033[31m Start the add-on process:\\\\033[0m", end=" --> ")\n        pancreas_and_tumor_labels = [',
                            '        # NOTE: ePAI all-organ patch: keep all labels by default; set EPAI_OUTPUT_LABEL_MODE=pancreas_only to restore filtering.\n        if os.environ.get("EPAI_OUTPUT_LABEL_MODE", "all_organs") == "pancreas_only":\n            pancreas_and_tumor_labels = [')
        text = text.replace('        segmentation_final_pancreas_and_tumor = np.zeros_like(segmentation_final)\n        largest_cc_pred_mask_pancreas_and_tumor = keep_largest_component_multi_cls(segmentation_final, pancreas_and_tumor_labels)\n        segmentation_final_pancreas_and_tumor[largest_cc_pred_mask_pancreas_and_tumor==1] = segmentation_final[largest_cc_pred_mask_pancreas_and_tumor==1]\n        segmentation_final = segmentation_final_pancreas_and_tumor',
                            '            segmentation_final_pancreas_and_tumor = np.zeros_like(segmentation_final)\n            largest_cc_pred_mask_pancreas_and_tumor = keep_largest_component_multi_cls(segmentation_final, pancreas_and_tumor_labels)\n            segmentation_final_pancreas_and_tumor[largest_cc_pred_mask_pancreas_and_tumor==1] = segmentation_final[largest_cc_pred_mask_pancreas_and_tumor==1]\n            segmentation_final = segmentation_final_pancreas_and_tumor')
    path.write_text(text, encoding="utf-8")

def patch_predict(path: Path):
    text = path.read_text(encoding="utf-8")
    if "--output_label_mode" not in text:
        text = text.replace("    parser.add_argument('--save_probabilities', action='store_true',",
                            "    parser.add_argument('--output_label_mode', choices=['all_organs', 'pancreas_only'], default='all_organs', help='all_organs keeps all predicted labels; pancreas_only preserves original ePAI filtering to labels 13/14/23/24/25.')\n    parser.add_argument('--save_probabilities', action='store_true',")
    if "EPAI_OUTPUT_LABEL_MODE" not in text:
        text = text.replace("    args = parser.parse_args()\n", "    args = parser.parse_args()\n    import os\n    os.environ['EPAI_OUTPUT_LABEL_MODE'] = getattr(args, 'output_label_mode', 'all_organs')\n")
    path.write_text(text, encoding="utf-8")

def main():
    changed = []
    targets = [
        (patch_backup, ROOT / "backup_model/3D-TransUNet/inference.py"),
        (patch_export, ROOT / "binary/nnunetv2/inference/export_prediction.py"),
        (patch_predict, ROOT / "binary/nnunetv2/inference/predict_from_raw_data.py"),
        (patch_export, ROOT / "train/nnunetv2/inference/export_prediction.py"),
        (patch_predict, ROOT / "train/nnunetv2/inference/predict_from_raw_data.py"),
    ]
    for fn, path in targets:
        if path.exists():
            fn(path); changed.append(str(path))
    print({"status": "success", "changed_files": changed, "default_output_label_mode": "all_organs"})

if __name__ == "__main__":
    main()
