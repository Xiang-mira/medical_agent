#!/usr/bin/env python3
"""
准备人工抽查用的对比文件。
对指定 case，生成：
  1. original_vista3d.nii.gz  — 原始 VISTA3D 预测（combined label map）
  2. round{N}_student.nii.gz  — 第N轮 student 预测（combined label map）
  3. ct.nii.gz                — 原始 CT（软链接）

用法：
  python prepare_visual_check.py --case PanTS_00000026 --rounds 1 2 3
"""
import argparse, yaml, subprocess, sys
import numpy as np
import nibabel as nib
from pathlib import Path

PROJECT_ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
TEACHER_MAP  = PROJECT_ROOT / "configs/teacher_branch_map.yaml"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
VISTA3D_ROOT = PROJECT_ROOT / "checkpoints/VISTA3D-Inference-Pipeline-master/VISTA3D-Inference-Pipeline-master"


def build_combined(pred_dir: Path, branch_map: dict, ref_shape, ref_affine) -> np.ndarray:
    combined = np.zeros(ref_shape, dtype=np.uint8)
    for organ, info in branch_map.items():
        label_id = info.get("vista3d_label_id", 0)
        if label_id == 0:
            continue
        p = pred_dir / f"{organ}.nii.gz"
        if not p.exists():
            continue
        arr = np.asanyarray(nib.load(str(p)).dataobj) > 0
        if arr.shape == ref_shape:
            combined[arr] = label_id
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case ID, e.g. PanTS_00000026")
    ap.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "visual_check"))
    args = ap.parse_args()

    case_id = args.case
    out_dir = Path(args.out_dir) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TEACHER_MAP) as f:
        branch_map = yaml.safe_load(f)

    # CT 路径
    ct_path = PROJECT_ROOT / "data/PanTS/ImageTr" / case_id / "ct.nii.gz"
    if not ct_path.exists():
        print(f"CT 不存在: {ct_path}")
        sys.exit(1)

    ct_img = nib.load(str(ct_path))
    ref_shape = ct_img.get_fdata().shape
    ref_affine = ct_img.affine

    # 复制 CT
    import shutil
    shutil.copy2(str(ct_path), str(out_dir / "ct.nii.gz"))
    print(f"✓ CT: {out_dir / 'ct.nii.gz'}")

    # 原始 VISTA3D 预测（用 round1 estep 里 vista3d teacher 的结果）
    vista3d_seg_dir = OUTPUT_ROOT / "round1/estep/cases" / case_id / "raw_predictions/vista3d" / case_id / "segmentations"
    if vista3d_seg_dir.exists():
        combined = build_combined(vista3d_seg_dir, branch_map, ref_shape, ref_affine)
        out_path = out_dir / "original_vista3d.nii.gz"
        nib.save(nib.Nifti1Image(combined, ref_affine), str(out_path))
        n = (combined > 0).any()
        print(f"✓ 原始 VISTA3D: {out_path}  (有分割: {n})")
    else:
        # 重新推理原始 VISTA3D
        print("原始 VISTA3D 预测不存在，重新推理...")
        import tempfile, os, json as _json
        with tempfile.TemporaryDirectory() as td:
            override = {
                "postprocessing": {
                    "_target_": "Compose",
                    "transforms": [
                        {"_target_": "monai.apps.vista3d.transforms.VistaPostTransformd", "keys": "pred"},
                        {"_target_": "Invertd", "keys": "pred", "transform": "$copy.deepcopy(@preprocessing)",
                         "orig_keys": "@image_key", "nearest_interp": True, "to_tensor": True},
                        {"_target_": "Lambdad", "func": "$lambda x: torch.nan_to_num(x, nan=255)", "keys": "pred"},
                        {"_target_": "SaveImaged", "keys": "pred", "resample": False,
                         "data_root_dir": "@input_dir", "output_dir": "@output_dir",
                         "output_ext": "@output_ext", "output_dtype": "@output_dtype",
                         "output_postfix": "@output_postfix", "separate_folder": "@separate_folder"},
                    ]
                }
            }
            override_path = Path(td) / "override.json"
            override_path.write_text(_json.dumps(override))
            env = {**os.environ, "VISTA3D_OUTPUT_DIR": td}
            cmd = [
                sys.executable, "-m", "monai.bundle", "run",
                "--config_file", f"['{VISTA3D_ROOT}/configs/inference.json','{override_path}']",
                "--bundle_root", str(VISTA3D_ROOT),
                "--input_dict", _json.dumps({"image": str(ct_path)}),
                "--input_dir", str(ct_path.parent),
                "--output_dir", td,
                "--output_postfix", "orig",
                "--separate_folder", "False",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(VISTA3D_ROOT), env=env)
            outputs = list(Path(td).glob("*.nii.gz"))
            if outputs:
                orig_arr = nib.load(str(outputs[0])).get_fdata().astype("int16")
                out_path = out_dir / "original_vista3d.nii.gz"
                nib.save(nib.Nifti1Image(orig_arr, ref_affine), str(out_path))
                print(f"✓ 原始 VISTA3D: {out_path}")
            else:
                print(f"✗ 原始 VISTA3D 推理失败: {result.stderr[-500:]}")

    # 各轮 student 预测
    for rnd in args.rounds:
        pred_dir = OUTPUT_ROOT / f"round{rnd}/student_predictions" / case_id
        if not pred_dir.exists():
            print(f"✗ Round {rnd} student 预测不存在，跳过")
            continue
        combined = build_combined(pred_dir, branch_map, ref_shape, ref_affine)
        out_path = out_dir / f"round{rnd}_student.nii.gz"
        nib.save(nib.Nifti1Image(combined, ref_affine), str(out_path))
        n_organs = len([o for o in branch_map if (pred_dir / f"{o}.nii.gz").exists()])
        print(f"✓ Round {rnd} student: {out_path}  ({n_organs} organs)")

    print(f"\n文件已准备好，用 ITK-SNAP 打开：")
    print(f"  主图像: {out_dir}/ct.nii.gz")
    print(f"  叠加层: {out_dir}/original_vista3d.nii.gz  (原始VISTA3D)")
    for rnd in args.rounds:
        print(f"  叠加层: {out_dir}/round{rnd}_student.nii.gz  (Round {rnd} student)")
    print(f"\nITK-SNAP 命令行打开：")
    print(f"  itksnap -g {out_dir}/ct.nii.gz -s {out_dir}/round3_student.nii.gz")


if __name__ == "__main__":
    main()
