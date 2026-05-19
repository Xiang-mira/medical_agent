"""
Input: path to a BDMAP format CT data folder
Output: path to a nnUNet format CT data folder

Note: For the purpose of saving storage and quicker transform, we only create softlink of the source data!
"""

import os
import glob
import sys
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Example script with two input arguments")

    # Add two positional arguments
    parser.add_argument("--input_bdmap_path", type=str, help="path to a BDMAP format CT data folder")
    parser.add_argument("--output_nnunet_path", type=str, help="path to a nnUNet format CT data folder")

    args = parser.parse_args()
    if not os.path.exists(args.output_nnunet_path):
        os.makedirs(args.output_nnunet_path, exist_ok=True)

    input_paths = glob.glob(os.path.join(args.input_bdmap_path, "*", "ct.nii.gz"))
    print(f"Detected input of {len(input_paths)} CT scans.")
    for ct_path in tqdm(input_paths):
        
        source_path = ct_path
        bdmap_id = ct_path.split("/")[-2]
        target_path = os.path.join(args.output_nnunet_path, bdmap_id+"_0000.nii.gz")    # `_0000` is requested by nnUNet format dataset

        if not os.path.exists(target_path):
            os.symlink(source_path, target_path)
    print("convert to nnunet format done.")

if __name__ == "__main__":
    main()
