import os, sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description="Converting BDMAP format one-hot labels to nnUNet format combined labels.")
parser.add_argument('--input_bdmap_path', type=str, required=True, help='path to BDMAP format CT folder')
parser.add_argument('--input_bdmap_gt_path', type=str, required=True, help='path to BDMAP format one-hot label folder')
parser.add_argument('--output_nnunet_gt_path', type=str, required=True, help='path to nnUNet format combined label folder')

args = parser.parse_args()

# Define the paths
source_folder = args.input_bdmap_gt_path#'/mnt/bodymaps/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/'  # Replace with the actual path where one-hot labels are stored
dest_folder = args.output_nnunet_gt_path#"/mnt/ccvl15/tlin67/Dataset_raw/Dataset802_JHH_Test/" 

# Ensure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

# Define the label mapping: label file name and corresponding index
label_mapping = [
    ('aorta.nii.gz', 1),
    ('adrenal_gland_left.nii.gz', 2),
    ('adrenal_gland_right.nii.gz', 3),
    ('common_bile_duct.nii.gz', 4),
    ('celiac_aa.nii.gz', 5),
    ('colon.nii.gz', 6),
    ('duodenum.nii.gz', 7),
    ('gall_bladder.nii.gz', 8),
    ('postcava.nii.gz', 9),
    ('kidney_left.nii.gz', 10),
    ('kidney_right.nii.gz', 11),
    ('liver.nii.gz', 12),
    ('pancreas.nii.gz', 13),
    ('pancreatic_duct.nii.gz', 14),
    ('superior_mesenteric_artery.nii.gz', 15),
    ('intestine.nii.gz', 16),
    ('spleen.nii.gz', 17),
    ('stomach.nii.gz', 18),
    ('veins.nii.gz', 19),
    ('renal_vein_left.nii.gz', 20),
    ('renal_vein_right.nii.gz', 21),
    ('cbd_stent.nii.gz', 22),
    ('pancreatic_pdac.nii.gz', 23),
    ('pancreatic_cyst.nii.gz', 24),
    ('pancreatic_pnet.nii.gz', 25)
]

def process_case(case_folder):
    source_case_folder = os.path.join(source_folder, case_folder)
    dest_case_folder = dest_folder#os.path.join(dest_folder, "labelsTr")    # BDMAP -> nnUNet

    # Check if the case exists in the source folder
    if not os.path.isdir(source_case_folder):
        return f"Case {case_folder} not found in source folder. Skipping..."

    segmentations_path = os.path.join(source_case_folder, 'segmentations')
    ct_image_path = os.path.join(args.input_bdmap_path, case_folder, 'ct.nii.gz')  # Assuming ct.nii.gz is in the destination case folder
    output_file = os.path.join(dest_case_folder, f'{case_folder}.nii.gz')
    # print(case_folder, source_case_folder, dest_case_folder, segmentations_path, ct_image_path, output_file)

    # Check if the segmentations folder exists in the source case folder
    if not os.path.isdir(segmentations_path):
        return f"Segmentations folder not found in {source_case_folder}. Skipping..."

    # Check if ct.nii.gz exists in the destination case folder
    if not os.path.exists(ct_image_path):
        return f"CT image not found in {dest_case_folder}. Skipping..."

    # Load the CT image to get the affine and header
    ct_img = nib.load(ct_image_path)
    ct_data_shape = ct_img.shape
    affine = ct_img.affine
    header = ct_img.header.copy()
    header.set_data_dtype(np.uint8)  # Set data type to uint8 for labels

    # Initialize the combined label array with zeros
    combined_data = np.zeros(ct_data_shape, dtype=np.uint8)

    # Process labels in the specified order
    for label_file_name, label_index in (label_mapping):
        label_path = os.path.join(segmentations_path, label_file_name)
        if os.path.exists(label_path):
            # Load the label image
            label_img = nib.load(label_path)
            label_data = label_img.get_fdata()
            label_data = label_data.astype(np.uint8)

            # Check if label_data shape matches ct_data_shape
            if label_data.shape != ct_data_shape:
                print(f"Label {label_file_name} shape does not match CT image shape for case {case_folder}. Skipping this label.")
                continue

            # Assign label index where the label is present
            combined_data[label_data > 0] = label_index
        else:
            # Label file does not exist for this case
            continue

    if np.any(combined_data):
        # Save the combined label image into the destination case folder
        combined_img = nib.Nifti1Image(combined_data, affine, header)
        nib.save(combined_img, output_file)
        return f"Combined labels saved for case {case_folder}"
    else:
        return f"No labels found for case {case_folder}. Skipping..."

if __name__ == '__main__':
    bdmap_id_list_for_convertion = os.listdir(source_folder)
    print("Converting", len(bdmap_id_list_for_convertion), "cases...")

    # Get the list of case folders in the destination folder
    case_folders = [
        f for f in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, f)) and f in bdmap_id_list_for_convertion
    ]
    case_folders = sorted(case_folders)  # 900 1440

    # Use multiprocessing Pool to process cases in parallel
    num_processes = int(cpu_count() * 0.8)

    with Pool(num_processes) as pool:
        # Use tqdm to display progress bar
        results = []
        for res in tqdm(pool.imap_unordered(process_case, case_folders), total=len(case_folders), desc='Processing cases'):
            if res:
                # print(res)
                results.append(res)

    print("All cases processed successfully!")
