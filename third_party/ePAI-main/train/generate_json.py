from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *

import argparse

def main(args):
    
    labels = {
            "background": 0,
            "aorta": 1,
            "adrenal_gland_left": 2,
            "adrenal_gland_right": 3,
            "common_bile_duct": 4,
            "celiac_aa": 5,
            "colon": 6,
            "duodenum": 7,
            "gall_bladder": 8,
            "postcava": 9,
            "kidney_left": 10,
            "kidney_right": 11,
            "liver": 12,
            "pancreas": 13,
            "pancreatic_duct": 14,
            "superior_mesenteric_artery": 15,
            "intestine": 16,
            "spleen": 17,
            "stomach": 18,
            "veins": 19,
            "renal_vein_left": 20,
            "renal_vein_right": 21,
            "cbd_stent": 22,
            "pancreatic_pdac": 23,
            "pancreatic_cyst": 24,
            "pancreatic_pnet": 25
        }
    generate_dataset_json(
        join(args.raw_dir, args.dataset_name),
        {0: 'CT'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        labels,
        3145,
        '.nii.gz',
        None,
        name=args.dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient'
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_dir", type=str, default='/mnt/bodymaps/ePAI/nnUNet/raw', help='Path to the raw data directory')
    parser.add_argument("--dataset_name", type=str, default='Dataset1013_ePAI_3MM', help='Name of the target dataset')
    
    args = parser.parse_args()

    main(args)