import os
import nibabel as nib
import numpy as np
import shutil

import os
import shutil
import numpy as np
import nibabel as nib

def disturb_half_segmentations(input_dir, output_dir):
    # 复制文件夹结构
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    seg_dir = os.path.join(output_dir, "segmentations")
    for fname in os.listdir(seg_dir):
        if fname.endswith(".nii.gz"):
            fpath = os.path.join(seg_dir, fname)
            
            # 读取 segmentation
            img = nib.load(fpath)
            data = img.get_fdata()
            
            shape = data.shape

            # 随机选择一个轴向和切片范围
            axis = np.random.choice([0, 1, 2])
            cut_size = shape[axis] // 2   # 大约一半

            slices = [slice(None)] * 3
            if np.random.rand() > 0.5:
                slices[axis] = slice(0, cut_size)   # 前半
            else:
                slices[axis] = slice(shape[axis]-cut_size, shape[axis])  # 后半

            new_data = data.copy()
            new_data[tuple(slices)] = 1   # 🚨 这里改为置 1，而不是置 0

            # 保存到相同位置（覆盖原文件）
            new_img = nib.Nifti1Image(new_data, affine=img.affine, header=img.header)
            nib.save(new_img, fpath)

    print(f"完成！新文件夹已生成: {output_dir}")


# 示例调用
input_dir = "data/AA1.1_dhe23_ccvl40/BDMAP_00000001"
output_dir = "data/AA1.1_dhe23_ccvl40_masked/BDMAP_00000001"
disturb_half_segmentations(input_dir, output_dir)
