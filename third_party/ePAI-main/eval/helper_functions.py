import os
import cv2
import copy
import tarfile
import glob
import ast
import csv
import random
import string
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from tqdm import tqdm
from scipy import ndimage
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import resample
from collections import defaultdict

# class map for the AbdomenAtlas 1.0 dataset
class_map_abdomenatlas_1_0 = {
    1: 'aorta',
    2: 'gall_bladder',
    3: 'kidney_left',
    4: 'kidney_right',
    5: 'liver',
    6: 'pancreas',
    7: 'postcava',
    8: 'spleen',
    9: 'stomach',
    }

# class map for the AbdomenAtlas 1.1 dataset
class_map_abdomenatlas_1_1 = {
    1: 'aorta', 
    2: 'gall_bladder', 
    3: 'kidney_left', 
    4: 'kidney_right', 
    5: 'liver', 
    6: 'pancreas', 
    7: 'postcava', 
    8: 'spleen', 
    9: 'stomach', 
    10: 'adrenal_gland_left', 
    11: 'adrenal_gland_right', 
    12: 'bladder', 
    13: 'celiac_trunk', 
    14: 'colon', 
    15: 'duodenum', 
    16: 'esophagus', 
    17: 'femur_left', 
    18: 'femur_right', 
    19: 'hepatic_vessel', 
    20: 'intestine', 
    21: 'lung_left', 
    22: 'lung_right', 
    23: 'portal_vein_and_splenic_vein', 
    24: 'prostate', 
    25: 'rectum'
    }

# class map for the AbdomenAtlas 1.2 dataset
class_map_abdomenatlas_1_2 = {
    1: 'aorta', 
    2: 'gall_bladder', 
    3: 'kidney_left', 
    4: 'kidney_right', 
    5: 'liver', 
    6: 'pancreas', 
    7: 'postcava', 
    8: 'spleen', 
    9: 'stomach', 
    10: 'adrenal_gland_left', 
    11: 'adrenal_gland_right', 
    12: 'bladder', 
    13: 'celiac_trunk', 
    14: 'colon', 
    15: 'duodenum', 
    16: 'esophagus', 
    17: 'femur_left', 
    18: 'femur_right', 
    19: 'hepatic_vessel', 
    20: 'intestine', 
    21: 'lung_left', 
    22: 'lung_right', 
    23: 'portal_vein_and_splenic_vein', 
    24: 'prostate', 
    25: 'rectum',
    26: 'vertebrae_L1', 
    27: 'vertebrae_L2', 
    28: 'vertebrae_L3', 
    29: 'vertebrae_L4', 
    30: 'vertebrae_L5', 
    31: 'vertebrae_T1', 
    32: 'vertebrae_T2', 
    33: 'vertebrae_T3', 
    34: 'vertebrae_T4', 
    35: 'vertebrae_T5', 
    36: 'vertebrae_T6', 
    37: 'vertebrae_T7', 
    38: 'vertebrae_T8', 
    39: 'vertebrae_T9', 
    40: 'vertebrae_T10', 
    41: 'vertebrae_T11', 
    42: 'vertebrae_T12', 
    43: 'vertebrae_C1', 
    44: 'vertebrae_C2', 
    45: 'vertebrae_C3', 
    46: 'vertebrae_C4', 
    47: 'vertebrae_C5', 
    48: 'vertebrae_C6', 
    49: 'vertebrae_C7',
    50: 'rib_left_1', 
    51: 'rib_left_2', 
    52: 'rib_left_3', 
    53: 'rib_left_4', 
    54: 'rib_left_5', 
    55: 'rib_left_6', 
    56: 'rib_left_7', 
    57: 'rib_left_8', 
    58: 'rib_left_9', 
    59: 'rib_left_10', 
    60: 'rib_left_11', 
    61: 'rib_left_12', 
    62: 'rib_right_1', 
    63: 'rib_right_2', 
    64: 'rib_right_3', 
    65: 'rib_right_4', 
    66: 'rib_right_5', 
    67: 'rib_right_6', 
    68: 'rib_right_7', 
    69: 'rib_right_8', 
    70: 'rib_right_9', 
    71: 'rib_right_10', 
    72: 'rib_right_11', 
    73: 'rib_right_12'
    }

# class map for the AbdomenAtlas 2.0 dataset
class_map_abdomenatlas_2_0 = {
    1: 'liver',
    2: 'liver_lesion',
    3: 'pancreas',
    4: 'pancreatic_lesion',
    5: 'kidney_left',
    6: 'kidney_right',
    7: 'kidney_lesion',
    8: 'kidney_tumor',
    9: 'kidney_cyst',
    10: 'colon',
    11: 'colon_lesion',
    12: 'uterus',
    13: 'endometrioma_tumor',
    14: 'esophagus',
    15: 'esophagus_tumor',
    }

# class map for the AbdomenAtlas X dataset
class_map_abdomenatlas_x = {
    1: 'liver',
    2: 'pancreas',
    3: 'kidney_left',
    4: 'kidney_right',
    5: 'colon',
    6: 'uterus',
    7: 'esophagus',
    8: 'lesion',
    }

# class map for the AbdomenAtlas ImageCAS dataset
class_map_abdomenatlas_imagecas = {
    1: 'coronary_artery',
    }

# class map for the AbdomenAtlas Report dataset
class_map_abdomenatlas_report = {
    1: 'pancreas',
    2: 'superior_mesenteric_artery',
    3: 'veins',
    4: 'celiac_aa',
    5: 'common_bile_duct',
    6: 'pancreatic_pdac',
    7: 'pancreatic_cyst',
    8: 'pancreatic_pnet',
    }

# class map for the AbdomenAtlas 3.0 dataset
class_map_abdomenatlas_3_0 = {
    1: 'liver',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'spleen',
    5: 'pancreas',
    6: 'pancreas_head',
    7: 'pancreas_body',
    8: 'pancreas_tail',
    9: 'liver_segment_1',
    10: 'liver_segment_2',
    11: 'liver_segment_3',
    12: 'liver_segment_4',
    13: 'liver_segment_5',
    14: 'liver_segment_6',
    15: 'liver_segment_7',
    16: 'liver_segment_8',
    17: 'colon',
    18: 'stomach',
    19: 'duodenum',
    20: 'common_bile_duct',
    21: 'intestine',
    22: 'aorta',
    23: 'postcava', # 'inferior vena cava'
    24: 'adrenal_gland_left',
    25: 'adrenal_gland_right',
    26: 'gall_bladder',
    27: 'bladder',
    28: 'celiac_trunk',
    29: 'esophagus',
    30: 'hepatic_vessel',
    31: 'portal_vein_and_splenic_vein',
    32: 'lung_left',
    33: 'lung_right',
    34: 'prostate',
    35: 'rectum',
    36: 'femur_left',
    37: 'femur_right',
    38: 'superior_mesenteric_artery',
    39: 'veins',
    40: 'liver_tumor',
    41: 'liver_cyst',
    42: 'liver_lesion',
    43: 'pancreatic_tumor',
    44: 'pancreatic_cyst',
    45: 'pancreatic_lesion',
    46: 'colon_tumor',
    47: 'colon_cyst',
    48: 'colon_lesion',
    49: 'kidney_tumor',
    50: 'kidney_cyst',
    51: 'kidney_lesion',
    52: 'pancreatic_pdac',
    53: 'pancreatic_pnet'
}

# class map for the AbdomenAtlas 3.1 dataset
class_map_abdomenatlas_3_1 = {
    1: 'liver',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'spleen',
    5: 'pancreas',
    6: 'pancreas_head',
    7: 'pancreas_body',
    8: 'pancreas_tail',
    9: 'liver_segment_1',
    10: 'liver_segment_2',
    11: 'liver_segment_3',
    12: 'liver_segment_4',
    13: 'liver_segment_5',
    14: 'liver_segment_6',
    15: 'liver_segment_7',
    16: 'liver_segment_8',
    17: 'colon',
    18: 'stomach',
    19: 'duodenum',
    20: 'common_bile_duct',
    21: 'intestine',
    22: 'aorta',
    23: 'postcava',
    24: 'adrenal_gland_left',
    25: 'adrenal_gland_right',
    26: 'gall_bladder',
    27: 'bladder',
    28: 'celiac_trunk',
    29: 'esophagus',
    30: 'hepatic_vessel',
    31: 'portal_vein_and_splenic_vein',
    32: 'lung_left',
    33: 'lung_right',
    34: 'lung_upper_left_lobe',
    35: 'lung_lower_left_lobe',
    36: 'lung_upper_right_lobe',
    37: 'lung_middle_right_lobe',
    38: 'lung_lower_right_lobe',
    39: 'prostate',
    40: 'rectum',
    41: 'femur_left',
    42: 'femur_right',
    43: 'superior_mesenteric_artery',
    44: 'veins',
    45: 'vertebrae_L1',
    46: 'vertebrae_L2',
    47: 'vertebrae_L3',
    48: 'vertebrae_L4',
    49: 'vertebrae_L5',
    50: 'vertebrae_T1',
    51: 'vertebrae_T2',
    52: 'vertebrae_T3',
    53: 'vertebrae_T4',
    54: 'vertebrae_T5',
    55: 'vertebrae_T6',
    56: 'vertebrae_T7',
    57: 'vertebrae_T8',
    58: 'vertebrae_T9',
    59: 'vertebrae_T10',
    60: 'vertebrae_T11',
    61: 'vertebrae_T12',
    62: 'vertebrae_C1',
    63: 'vertebrae_C2',
    64: 'vertebrae_C3',
    65: 'vertebrae_C4',
    66: 'vertebrae_C5',
    67: 'vertebrae_C6',
    68: 'vertebrae_C7',
    69: 'vertebrae_S1',
    70: 'rib_left_1',
    71: 'rib_left_2',
    72: 'rib_left_3',
    73: 'rib_left_4',
    74: 'rib_left_5',
    75: 'rib_left_6',
    76: 'rib_left_7',
    77: 'rib_left_8',
    78: 'rib_left_9',
    79: 'rib_left_10',
    80: 'rib_left_11',
    81: 'rib_left_12',
    82: 'rib_right_1',
    83: 'rib_right_2',
    84: 'rib_right_3',
    85: 'rib_right_4',
    86: 'rib_right_5',
    87: 'rib_right_6',
    88: 'rib_right_7',
    89: 'rib_right_8',
    90: 'rib_right_9',
    91: 'rib_right_10',
    92: 'rib_right_11',
    93: 'rib_right_12',
    94: 'trachea',
    95: 'iliac_artery_left',
    96: 'iliac_artery_right',
    97: 'iliac_vena_left',
    98: 'iliac_vena_right',
    99: 'humerus_left',
    100: 'humerus_right',
    101: 'scapula_left',
    102: 'scapula_right',
    103: 'clavicula_left',
    104: 'clavicula_right',
    105: 'hip_left',
    106: 'hip_right',
    107: 'sacrum',
    108: 'gluteus_maximus_left',
    109: 'gluteus_maximus_right',
    110: 'gluteus_medius_left',
    111: 'gluteus_medius_right',
    112: 'gluteus_minimus_left',
    113: 'gluteus_minimus_right',
    114: 'autochthon_left',
    115: 'autochthon_right',
    116: 'iliopsoas_left',
    117: 'iliopsoas_right',
    118: 'brachiocephalic_trunk',
    119: 'brachiocephalic_vein_left',
    120: 'brachiocephalic_vein_right',
    121: 'common_carotid_artery_left',
    122: 'common_carotid_artery_right',
    123: 'costal_cartilages',
    124: 'pulmonary_vein',
    125: 'subclavian_artery_left',
    126: 'subclavian_artery_right',
    127: 'superior_vena_cava',
    128: 'thyroid_gland',
    129: 'airway',
    130: 'skull',
    131: 'heart',
    132: 'brain',
    133: 'spinal_cord',
    134: 'sternum',
    135: 'atrial_appendage_left',
    136: 'liver_tumor',
    137: 'liver_cyst',
    138: 'liver_lesion',
    139: 'hepatic_tumor',
    140: 'pancreatic_tumor',
    141: 'pancreatic_cyst',
    142: 'pancreatic_lesion',
    143: 'pancreatic_pdac',
    144: 'pancreatic_pnet',
    145: 'colon_tumor',
    146: 'colon_cyst',
    147: 'colon_lesion',
    148: 'colon_cancer_primaries',
    149: 'kidney_tumor',
    150: 'kidney_cyst',
    151: 'kidney_lesion',
    152: 'lung_tumor',
    153: 'bone_lesion',
}

class_map_abdomenatlas_f = {
    1: 'aorta',
    2: 'adrenal_gland_left',
    3: 'adrenal_gland_right',
    4: 'common_bile_duct',
    5: 'celiac_aa',
    6: 'colon',
    7: 'duodenum',
    8: 'gall_bladder',
    9: 'postcava',
    10: 'kidney_left',
    11: 'kidney_right',
    12: 'liver',
    13: 'pancreas',
    14: 'pancreatic_duct',
    15: 'superior_mesenteric_artery',
    16: 'intestine',
    17: 'spleen',
    18: 'stomach',
    19: 'veins',
    20: 'renal_vein_left',
    21: 'renal_vein_right',
    22: 'cbd_stent',
    23: 'pancreatic_pdac',
    24: 'pancreatic_cyst',
    25: 'pancreatic_pnet'
}

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def rename_delete_tumor_mask(pid, args):

    class_list = glob.glob(os.path.join(args.source_datapath, pid, 'segmentations', '*.nii.gz'))
    class_list = [c.split('/')[-1][:-7] for c in class_list]
    tumor_list = [c for c in class_list if ('tumor' in c or 'lesion' in c or 'cyst' in c or 'pdac' in c or 'pnet' in c) and c[0] != '_']
    psuedo_list = [c for c in class_list if c[0] == '_']

    # if kidney_tumor and kidney_lesion both exist in the tumor_list, delete kidney_lesion.nii.gz from destination folder
    if 'kidney_tumor' in tumor_list and 'kidney_lesion' in tumor_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
    
    # if pid is greater than BDMAP_00005195 and kidney_tumor exists in the tumor_list, rename as kidney_lesion.nii.gz in destination folder
    if 'kidney_tumor' in tumor_list and int(pid.split('_')[-1]) > 5195 and '_A' not in pid and '_V' not in pid and '_O' not in pid:
        os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_tumor.nii.gz'), 
                                    os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_tumor.nii.gz'), 
                                          os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
    
    # if liver_tumor and liver_lesion both exist in the tumor_list, delete liver_tumor.nii.gz from destination folder
    if 'liver_tumor' in tumor_list and 'liver_lesion' in tumor_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz')))

    # if liver_tumor exists in the tumor_list, rename as liver_lesion.nii.gz in destination folder
    if 'liver_tumor' in tumor_list:
        os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz'), 
                                    os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz'), 
                                          os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_lesion.nii.gz')))

    if 'pancreas_tumor' in tumor_list:
        # remove pancreas_tumor.nii.gz from destination folder
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreas_tumor.nii.gz')))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreas_tumor.nii.gz')))
        
    # if pancreatic_tumor and pancreatic_lesion both exist in the tumor_list, delete pancreatic_tumor.nii.gz from destination folder
    if 'pancreatic_tumor' in tumor_list and 'pancreatic_lesion' in tumor_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz')))

    # if pancreatic_tumor exists in the tumor_list, rename as pancreatic_lesion.nii.gz in destination folder
    if 'pancreatic_tumor' in tumor_list and 'pancreatic_pdac' not in tumor_list:
        os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz'), 
                                    os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz'), 
                                          os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_lesion.nii.gz')))

    # replace _xxx_tumor.nii.gz with _xxx_lesion.nii.gz in the destination folder
    for class_name in psuedo_list:
        if 'tumor' in class_name:
            os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz'), 
                                        os.path.join(args.destination_datapath, pid, 'segmentations', class_name.replace('tumor', 'lesion') + '.nii.gz')))
            # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
            print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz'), 
                                                os.path.join(args.destination_datapath, pid, 'segmentations', class_name.replace('tumor', 'lesion') + '.nii.gz')))
    
    # if _xxx_tumor and xxx_tumor or xxx_lesion both exist in the tumor_list, delete _xxx_tumor.nii.gz from destination folder
    class_list = glob.glob(os.path.join(args.source_datapath, pid, 'segmentations', '*.nii.gz'))
    class_list = [c.split('/')[-1][:-7] for c in class_list]
    tumor_list = [c for c in class_list if ('tumor' in c or 'lesion' in c or 'cyst' in c or 'pdac' in c or 'pnet' in c) and c[0] != '_']
    psuedo_list = [c for c in class_list if c[0] == '_']
    for class_name in psuedo_list:
        if class_name[1:] in tumor_list:
            
            # delete the psuedo mask file in the destination folder
            os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz')))
            # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
            print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz')))
    
    # if _kidney_lesion exists in pseduo_list and kidney_tumor exists in tumor_list, delete _kidney_lesion.nii.gz from destination folder
    if 'kidney_tumor' in tumor_list and '_kidney_lesion' in psuedo_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', '_kidney_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', '_kidney_lesion.nii.gz')))

def count_num_slices(pid, datapath):

    # if there is a ct.nii.gz file
    if os.path.isfile(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz'))
    elif os.path.isfile(os.path.join(datapath, pid, 'ct.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'ct.nii.gz'))
    else:
        raise ValueError('No ct.nii.gz or liver.nii.gz file found in {}'.format(os.path.join(datapath, pid)))

    if (dim[0] == dim[1]) or \
    (dim[0] != dim[1] and dim[1] != dim[2]):
        return dim[-1]
    else:
        return dim[0]

def get_spacing(pid, datapath):

    # if there is a segmentations/liver.nii.gz file
    if os.path.isfile(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz')):
        nii = nib.load(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz'))
    elif os.path.isfile(os.path.join(datapath, pid, 'ct.nii.gz')):
        nii = nib.load(os.path.join(datapath, pid, 'ct.nii.gz'))
    else:
        raise ValueError('No ct.nii.gz or liver.nii.gz file found in {}'.format(os.path.join(datapath, pid)))
    
    spacing = nii.header['pixdim'][1:4]
    return spacing

def get_shape(pid, datapath):

    # if there is a ct.nii.gz file
    if os.path.isfile(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz'))
    elif os.path.isfile(os.path.join(datapath, pid, 'ct.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'ct.nii.gz'))
    else:
        raise ValueError('No ct.nii.gz or liver.nii.gz file found in {}'.format(os.path.join(datapath, pid)))
    
    return dim

def compute_centroid(mask):
    # Get the indices of the non-zero elements in the mask
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    # Compute the centroid by averaging the indices
    centroid = np.mean(indices, axis=0)
    return centroid

def compute_distance(centroid1, centroid2):

    # Compute the Euclidean distance between two centroids
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance

def generate_combined_labels(pid, datapath, class_maps):

    combined_labels, affine, header = load_mask(pid, 'liver', datapath)
    combined_labels.fill(0)
    for cid, class_name in class_maps.items():
        mask, _, _ = load_mask(pid, class_name, datapath)
        combined_labels[mask > 0.5] = cid
    nifti_path = os.path.join(datapath, pid, 'combined_labels.nii.gz')
    nib.save(nib.Nifti1Image(combined_labels.astype(np.uint8), affine=affine, header=header), nifti_path)

def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() > 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    else:
        return segmentation

def find_largest_subarray_bounds(arr, low_threshold, high_threshold):
    
    # Find the indices where the condition is True
    condition = (arr > low_threshold) & (arr < high_threshold)
    condition = getLargestCC(condition)
    x, y, z = np.where(condition)

    if not len(x):
        return (0,0,0), (0,0,0)  # No values above threshold

    # Find min and max indices along each dimension
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def crop_largest_subarray(arr, low_threshold, high_threshold, case_name=None):
    
    (min_x, min_y, min_z), (max_x, max_y, max_z) = find_largest_subarray_bounds(arr, low_threshold, high_threshold)
    # if max_x - min_x < 50 or max_y - min_y < 50 or max_z - min_z < 5:
    #     print('ERROR in {}'.format(case_name))
    
    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def standardization(original_ct_file, revised_ct_file, 
                    original_mask_file=None, revised_mask_file=None,
                    original_comb_file=None, revised_comb_file=None,
                    image_type=np.int16, mask_type=np.uint8,
                   ):
    
    img = nib.load(original_ct_file)
    data = np.array(img.dataobj)

    data[data > 1000] = 1000
    data[data < -1000] = -1000
    
    (min_x, min_y, min_z), (max_x, max_y, max_z) = crop_largest_subarray(arr=data, 
                                                                         low_threshold=-100, 
                                                                         high_threshold=100, 
                                                                         case_name=original_ct_file.split('/')[-2])
    data = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    data = nib.Nifti1Image(data, img.affine, img.header)
    data.set_data_dtype(image_type)
    data.get_data_dtype(finalize=True)
    
    nib.save(data, revised_ct_file)

    if original_comb_file is not None and revised_comb_file is not None:
        
        img = nib.load(original_comb_file)
        data = np.array(img.dataobj)
        mask = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

        mask = nib.Nifti1Image(mask, img.affine, img.header)
        mask.set_data_dtype(mask_type)
        mask.get_data_dtype(finalize=True)

        nib.save(mask, revised_comb_file)

    if original_mask_file is not None and revised_mask_file is not None:
        
        for original, revised in zip(original_mask_file, revised_mask_file):
            img = nib.load(original)
            data = np.array(img.dataobj)
            mask = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

            mask = nib.Nifti1Image(mask, img.affine, img.header)
            mask.set_data_dtype(mask_type)
            mask.get_data_dtype(finalize=True)

            nib.save(mask, revised)

def get_dim(nii_path):

    try:
        nii_image = nib.load(nii_path)
    except:
        print('{} is not a gzip file'.format(nii_path))
        sitk_img = sitk.ReadImage(nii_path)  # load with sitk
        sitk.WriteImage(sitk_img, nii_path)  # overwrite

    nii_image = nib.load(nii_path)
    
    return nii_image.shape

def load_ct(pid, datapath):

    ct_path = os.path.join(datapath, pid, 'ct.nii.gz')
    if os.path.isfile(ct_path):
        nii = nib.load(ct_path)
        ct = nii.get_fdata().astype(np.int16)
        return ct, nii.affine, nii.header
    else:
        None, None, None

def save_ct(data, affine, header, pid, datapath):
    
    if not os.path.exists(os.path.join(datapath, pid)):
        os.makedirs(os.path.join(datapath, pid))
    nifti_path = os.path.join(datapath, pid, 'ct.nii.gz')

    ct = nib.Nifti1Image(data, affine, header)
    ct.set_data_dtype(np.int16)
    ct.get_data_dtype(finalize=True)
    nib.save(ct, nifti_path)

def load_mask(pid, class_name, datapath, hiddenpath='/mnt/T9/AbdomenAtlasPro'):

    mask_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')
    mask_hidden_path = os.path.join(hiddenpath, pid, 'segmentations', class_name + '.nii.gz')
    if os.path.isfile(mask_path):
        nii = nib.load(mask_path)
        mask = nii.get_fdata().astype(np.uint8)
        return mask, nii.affine, nii.header
        
    elif os.path.isfile(mask_hidden_path):
        nii = nib.load(mask_hidden_path)
        mask = nii.get_fdata().astype(np.uint8)
        return mask, nii.affine, nii.header
        
    else:
        return None, None, None

def save_mask(data, affine, header, pid, class_name, datapath):
    
    if not os.path.exists(os.path.join(datapath, pid, 'segmentations')):
        os.makedirs(os.path.join(datapath, pid, 'segmentations'))
    nifti_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')

    mask = nib.Nifti1Image(data, affine, header)
    mask.set_data_dtype(np.uint8)
    mask.get_data_dtype(finalize=True)
    nib.save(mask, nifti_path)

def check_dim(list_of_array):

    dim = list_of_array[0].shape
    for i in range(len(list_of_array)):
        if dim != list_of_array[i].shape:
            return False
    return True

def plot_organ_projection(list_of_array, organ_name, pid, axis=2, pngpath=None):

    if axis == 2:
        projection = np.zeros((list_of_array[0][:,:,0].shape), dtype='float')
    else:
        raise
    for i in range(len(list_of_array)):
        organ_projection = np.sum(list_of_array[i], axis=axis) * 1.0
        organ_projection /= np.max(organ_projection)
        projection += organ_projection
    projection /= np.max(projection)
    projection *= 255.0
    projection = np.rot90(projection)

    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    cv2.imwrite(os.path.join(pngpath, pid + '.png'), projection)

def from_str_to_num_list(str_list):

    if str_list == '[]':
        return []
    else:
        num_list = ast.literal_eval(str_list)
        return num_list

def initialize_temporary_csv():

    # generate a string of random characters and numbers, lenth = 10
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    internal_csv_path = random_string + '.csv'

    # if CSV file already exists, remove it
    if os.path.isfile(internal_csv_path):
        os.remove(internal_csv_path)
    # create a CSV file to store the results
    with open(internal_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bdmap_id', 'prediction', 'logit', 'confidence', 'gt'])  # Write header

    return internal_csv_path

def get_AI_confidence(pid_result, args):

    conf = 1
    
    pancreas_volume_size = from_str_to_num_list(pid_result['pancreas_pr_volume_size'])
    if len(pancreas_volume_size) == 0 or max(pancreas_volume_size) < args.pancreas_volume_threshold:
        conf = 0
    else:
        pr = get_AI_prediction(pid_result, args)
        duct_volume_size = from_str_to_num_list(pid_result['duct_pr_volume_size'])
        if len(duct_volume_size) > 0 and max(duct_volume_size) > args.duct_volume_threshold and pr == 0:
            # If dilated duct, then this patient cannot be negative (healthy)
            conf = 0

    return conf

def get_AI_prediction(pid_result, args):

    PDAC_volume_size = from_str_to_num_list(pid_result['PDAC_pr_volume_size'])
    cyst_volume_size = from_str_to_num_list(pid_result['cyst_pr_volume_size'])
    PNET_volume_size = from_str_to_num_list(pid_result['PNET_pr_volume_size'])
    

    pr = 0  # Default prediction is negative
    for coi in args.coi:
        if coi == 'pdac':
            if len(PDAC_volume_size) > 0 and compute_diameter_from_volume(max(PDAC_volume_size)) > args.diameter_threshold:
                pr = 1
        elif coi == 'cyst':
            if len(cyst_volume_size) > 0 and compute_diameter_from_volume(max(cyst_volume_size)) > args.diameter_threshold:
                pr = 1
        elif coi == 'pnet': 
            if len(PNET_volume_size) > 0 and compute_diameter_from_volume(max(PNET_volume_size)) > args.diameter_threshold:
                pr = 1
    
    return pr

def compute_diameter_from_volume(volume):
    # Assuming volume is in mm^3, compute diameter in mm
    # Volume = (pi/6) * d^3 => d = (6*Volume/pi)^(1/3)

    diameter = ((6 * volume) / 3.141592653589793)**(1/3)
    # print(f"Computed diameter from volume {volume}: {diameter}\n")

    return diameter

def get_AI_logit(pid_result, args):

    PDAC_volume_size = from_str_to_num_list(pid_result['PDAC_pr_volume_size'])
    cyst_volume_size = from_str_to_num_list(pid_result['cyst_pr_volume_size'])
    PNET_volume_size = from_str_to_num_list(pid_result['PNET_pr_volume_size'])

    PDAC_largest_component_largest_logit = 0.0 if pid_result['PDAC_pr_largest_component_largest_logit'] == "" else float(pid_result['PDAC_pr_largest_component_largest_logit'])
    cyst_largest_component_largest_logit = 0.0 if pid_result['cyst_pr_largest_component_largest_logit'] == "" else float(pid_result['cyst_pr_largest_component_largest_logit'])
    PNET_largest_component_largest_logit = 0.0 if pid_result['PNET_pr_largest_component_largest_logit'] == "" else float(pid_result['PNET_pr_largest_component_largest_logit'])

    logit = 0  # Default logit is 0
    for coi in args.coi:
        if coi == 'pdac':
            if len(PDAC_volume_size) > 0 and compute_diameter_from_volume(max(PDAC_volume_size)) > args.diameter_threshold:
                logit = max(logit, PDAC_largest_component_largest_logit)
        elif coi == 'cyst':
            if len(cyst_volume_size) > 0 and compute_diameter_from_volume(max(cyst_volume_size)) > args.diameter_threshold:
                logit = max(logit, cyst_largest_component_largest_logit)
        elif coi == 'pnet':
            if len(PNET_volume_size) > 0 and compute_diameter_from_volume(max(PNET_volume_size)) > args.diameter_threshold:
                logit = max(logit, PNET_largest_component_largest_logit)
    return logit

def read_standardized_csv(gt, csv_files):

    raw_results = []
    gt_results = []
    for i, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        with open(csv_file, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            results = [row for row in reader]
            check_csv_standardization(results)
            raw_results.extend(results)

        if gt[i] == 'P':
            gt_results.extend(['1'] * len(results))
        elif gt[i] == 'N':
            gt_results.extend(['0'] * len(results))
    
    assert len(gt_results) == len(raw_results), "Ground truth results must match the number of predictions"

    return raw_results, gt_results

def check_csv_standardization(results):

    csv_header = [
        'bdmap_id', 'shape', 'spacing',
        'pancreas_pr', 'pancreas_pr_component_count', 'pancreas_pr_voxel_size', 'pancreas_pr_volume_size',
        'duct_pr', 'duct_pr_component_count', 'duct_pr_voxel_size', 'duct_pr_volume_size',
        'PDAC_pr', 'PDAC_pr_component_count', 'PDAC_pr_voxel_size', 'PDAC_pr_volume_size', 'PDAC_pr_largest_component_largest_logit',
        'cyst_pr', 'cyst_pr_component_count', 'cyst_pr_voxel_size', 'cyst_pr_volume_size', 'cyst_pr_largest_component_largest_logit',
        'PNET_pr', 'PNET_pr_component_count', 'PNET_pr_voxel_size', 'PNET_pr_volume_size', 'PNET_pr_largest_component_largest_logit'
    ]
    for header in csv_header:
        assert header in results[0], f"CSV file must contain '{header}' column"
        
# Zongwei's first implementation of reading temporary CSV and computing metrics
# def read_temporary_csv(args, remove_csv=True):

#     # Read the results from the internal CSV file
#     with open(args.internal_csv_path, mode='r') as file:
#         reader = csv.DictReader(file)
#         processed_results = [row for row in reader]
#     # Check if 'BDMAP_V' or 'BDMAP_A' is present in processed_results 'patient_id' key
#     if 'BDMAP_V' in processed_results[0]['bdmap_id'] or 'BDMAP_A' in processed_results[0]['bdmap_id']:
#         # print('>> Processing raw results to patient-level results')
#         processed_results = aggregate_patient_level_results(processed_results, coi=args.coi)

#     # Delete the internal CSV file after processing
#     if remove_csv:
#         if os.path.isfile(args.internal_csv_path):
#             os.remove(args.internal_csv_path)
            
#     return processed_results

# Revised by Wenxuan
def read_temporary_csv(args, remove_csv=True):

    # 1. Read the results from the internal CSV file
    with open(args.internal_csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        processed_results = [row for row in reader]
    
    # 2. Separate by ID pattern
    bdmap_av_results = [r for r in processed_results if "BDMAP_A" in r["bdmap_id"] or "BDMAP_V" in r["bdmap_id"]]
    other_results = [
        r for r in processed_results
        if not ("BDMAP_A" in r["bdmap_id"] or "BDMAP_V" in r["bdmap_id"])
    ]

    # 3. Aggregate A/V subset to patient-level if needed
    aggregated_results = []
    if bdmap_av_results:
        # print(">> Aggregating arterial/venous (BDMAP_A/V) results to patient-level")
        aggregated_results = aggregate_patient_level_results(bdmap_av_results, coi=args.coi)

    # 4. Combine both result types
    if aggregated_results and other_results:
        processed_results = aggregated_results + other_results
    elif aggregated_results:
        processed_results = aggregated_results
    else:
        processed_results = other_results

    # 5. Remove the temporary CSV file if required
    if remove_csv and os.path.isfile(args.internal_csv_path):
        os.remove(args.internal_csv_path)

    return processed_results

def compute_detection_metrics(gt, pr, conf=0.95, n_bootstraps=1000, seed=42):

    gt = np.array(gt)
    pr = np.array(pr)

    # Confusion matrix components
    TP = np.sum((pr == 1) & (gt == 1))
    FN = np.sum((pr == 0) & (gt == 1))
    TN = np.sum((pr == 0) & (gt == 0))
    FP = np.sum((pr == 1) & (gt == 0))
    total = TP + TN + FP + FN

    # Metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Confidence intervals (binomial, Wilson)
    sensitivity_ci = proportion_confint(TP, TP + FN, alpha=1-conf, method='wilson') if (TP + FN) > 0 else (0, 0)
    specificity_ci = proportion_confint(TN, TN + FP, alpha=1-conf, method='wilson') if (TN + FP) > 0 else (0, 0)
    accuracy_ci = proportion_confint(TP + TN, total, alpha=1-conf, method='wilson') if total > 0 else (0, 0)
    precision_ci = proportion_confint(TP, TP + FP, alpha=1-conf, method='wilson') if (TP + FP) > 0 else (0, 0)

    # Bootstrap F1 confidence interval
    rng = np.random.default_rng(seed)
    f1_scores = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(gt), size=len(gt), replace=True)
        gt_sample = gt[indices]
        pr_sample = pr[indices]
        f1_sample = f1_score(gt_sample, pr_sample, zero_division=0)
        f1_scores.append(f1_sample)
    
    lower = np.percentile(f1_scores, (1 - conf) / 2 * 100)
    upper = np.percentile(f1_scores, (1 + conf) / 2 * 100)
    f1_ci = (lower, upper)

    # Print metrics with 95% CI
    print(f"Sensitivity: {100*sensitivity:.1f}% ({TP}/{TP + FN}) \t(95% CI: {100*sensitivity_ci[0]:.1f} - {100*sensitivity_ci[1]:.1f}%)")
    print(f"Specificity: {100*specificity:.1f}% ({TN}/{TN + FP}) \t(95% CI: {100*specificity_ci[0]:.1f} - {100*specificity_ci[1]:.1f}%)")
    print(f"Accuracy:    {100*accuracy:.1f}% ({TP + TN}/{total}) \t(95% CI: {100*accuracy_ci[0]:.1f} - {100*accuracy_ci[1]:.1f}%)")
    print(f"Precision:   {100*precision:.1f}% ({TP}/{TP + FP}) \t(95% CI: {100*precision_ci[0]:.1f} - {100*precision_ci[1]:.1f}%)")
    print(f"F1 Score:    {100*f1:.1f}% ({TP}/{TP + FN + FP}) \t(95% CI: {100*f1_ci[0]:.1f} - {100*f1_ci[1]:.1f}%)")

    return {
        "sensitivity": (sensitivity, sensitivity_ci),
        "specificity": (specificity, specificity_ci),
        "accuracy": (accuracy, accuracy_ci),
        "precision": (precision, precision_ci),
        "f1_score": (f1, f1_ci),
    }

def compute_auc_metrics(gt, logit, conf=0.95, n_bootstraps=1000, seed=42, print_auc=True):
    gt = np.array(gt)
    logit = np.array(logit)

    # Compute AUC
    try:
        auc = roc_auc_score(gt, logit)
    except ValueError:
        print("AUC could not be computed (possibly only one class present).")
        return {"auc": (np.nan, (np.nan, np.nan))}

    # Bootstrapping to compute 95% CI
    rng = np.random.default_rng(seed)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(gt), size=len(gt), replace=True)
        if len(np.unique(gt[indices])) < 2:
            continue  # Skip iteration if only one class present
        score = roc_auc_score(gt[indices], logit[indices])
        bootstrapped_scores.append(score)

    # Compute confidence interval
    lower = np.percentile(bootstrapped_scores, (1 - conf) / 2 * 100)
    upper = np.percentile(bootstrapped_scores, (1 + conf) / 2 * 100)
    auc_ci = (lower, upper)

    # Print result
    if print_auc:
        print(f"AUC:         {auc:.3f} (95% CI: {auc_ci[0]:.3f} - {auc_ci[1]:.3f}; n={len(gt)})")

    return {
        "auc": (auc, auc_ci)
    }

def plot_roc_curve(GT, PR, saverocpath, roc_file_name=None,
                   axiswidth=5, linewidth=15, fontsize=80, figsize=(40, 40), operating_point=True, readers_performance=False,
                   alpha=0.0, zoomin=False, labelpad=50, legend_exp_name='', roc_color=[237/255, 16/255, 105/255]):
    """
    Plot ROC curve using ground truth and predicted scores.

    Parameters:
    - GT: Ground truth labels (binary)
    - PR: Prediction scores (probability or confidence)
    - args: An object with attribute `saverocpath` for saving the plot
    - axiswidth, linewidth, fontsize, figsize, alpha: Plot styling parameters
    - zoomin: If True, zoom into the high-specificity and high-sensitivity region
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(GT, PR)
    auc_metric = compute_auc_metrics(GT, PR, print_auc=False)

    # Find the point closest to (0,1)
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    op_idx = np.argmin(distances)
    op_fpr, op_tpr = fpr[op_idx], tpr[op_idx]
    op_threshold = thresholds[op_idx]

    # Styling
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = axiswidth
    plt.rcParams['axes.spines.right'] = zoomin
    plt.rcParams['axes.spines.top'] = zoomin

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    if readers_performance:
        legend_label = legend_exp_name + ' \nAUC={:.3f} (95% CI {:.3f}-{:.3f})'.format(auc_metric['auc'][0], auc_metric['auc'][1][0], auc_metric['auc'][1][1])
        plt.plot(fpr, tpr, color=roc_color, 
                label=legend_label,
                linewidth=linewidth)
    else:
        legend_label = legend_exp_name + ' (n={})\nAUC={:.3f} (95% CI {:.3f}-{:.3f})'.format(len(GT), auc_metric['auc'][0], auc_metric['auc'][1][0], auc_metric['auc'][1][1])
        plt.plot(fpr, tpr, color=roc_color, 
                label=legend_label,
                linewidth=linewidth)

    # Fill narrow band under ROC curve
    n_shades = 30
    for i in range(n_shades):
        alpha_val = 0.15 * (1 - i / n_shades)  # fade out gradually
        offset = i * 0.005  # controls thickness
        lower = np.maximum(tpr - offset, 0)
        ax.fill_between(fpr, lower, tpr - (offset - 0.005), color=roc_color, alpha=alpha_val, linewidth=0)

    if operating_point:
        # Plot optimal operating point
        ax.plot(op_fpr, op_tpr, 'o', markersize=3*linewidth,
                markeredgewidth=linewidth,
                markeredgecolor='black', markerfacecolor='white',
                label=f'ePAI operating point (thresh={op_threshold:.2f})')
    if readers_performance:
        # Add human reader performance points with distinct shapes
        human_points = [
            {
                "label": "Specialist (n=3)",
                "sensitivity": 0.298,
                "specificity": 0.963,
                "color": "white",
                "marker": "D"  # Diamond
            },
            {
                "label": "General Radiologist (n=12)",
                "sensitivity": 0.318,
                "specificity": 0.953,
                "color": "white",
                "marker": "^"  # Triangle up
            },
            {
                "label": "Resident (n=15)",
                "sensitivity": 0.294,
                "specificity": 0.942,
                "color": "white",
                "marker": "s"  # Square (rectangle)
            }
        ]
        for pt in human_points:
            ax.plot(
                1 - pt["specificity"], pt["sensitivity"],
                marker=pt["marker"],
                markersize=3*linewidth,
                markeredgewidth=linewidth,
                markeredgecolor='black',
                markerfacecolor=pt["color"],
                linestyle='None',
                label=pt["label"],
                zorder=10
            )

        # To avoid duplicate legend entries, handle legend after plotting all points
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend([h for h, l in unique], [l for h, l in unique], loc='lower right', frameon=False)
    
    ax.legend(loc='lower right', frameon=False)

    plt.axis('square')

    # Tick settings
    ax.tick_params(axis='both', which='major', length=axiswidth*8, width=axiswidth, pad=labelpad//2)

    if zoomin:
        plt.xlim(0, 0.3)
        plt.xticks([0.0, 0.1, 0.2, 0.3])
        plt.ylim(0.85, 1.0)
        plt.yticks([0.9, 0.95, 1.0])
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        plt.xlim(-0.02, 1.0)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylim(0., 1.02)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel('1-Specificity', labelpad=labelpad)
        ax.set_ylabel('Sensitivity', labelpad=labelpad)

    plt.grid(alpha=alpha, linewidth=linewidth)

    # Save plots
    if not os.path.exists(saverocpath):
        os.makedirs(saverocpath)
    basename = roc_file_name+'_zoomin' if zoomin else roc_file_name
    fig.savefig(os.path.join(saverocpath, f'{basename}.png'), 
                bbox_inches='tight', pad_inches=0.0, dpi=200)
 

def plot_two_roc_curves(GT1, PR1, GT2, PR2, saverocpath, roc_file_name=None,
                        legend_exp_name1='', legend_exp_name2='',
                        axiswidth=5, linewidth=15, fontsize=80, figsize=(40, 40),
                        alpha=0.0, zoomin=False, labelpad=50,
                        readers_performance=False,
                        color1=[160/255, 210/255, 240/255],
                        color2=[237/255, 16/255, 105/255]):
    """Plot two ROC curves on the same figure with full styling and features."""
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = axiswidth
    plt.rcParams['axes.spines.right'] = zoomin
    plt.rcParams['axes.spines.top'] = zoomin

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # ---- First ROC ----
    fpr1, tpr1, thresholds1 = roc_curve(GT1, PR1)
    auc1 = compute_auc_metrics(GT1, PR1, print_auc=False)['auc']

    # Operating point (closest to (0,1))
    dist1 = np.sqrt(fpr1**2 + (1 - tpr1)**2)
    op_idx1 = np.argmin(dist1)
    op_fpr1, op_tpr1, op_thr1 = fpr1[op_idx1], tpr1[op_idx1], thresholds1[op_idx1]

    label1 = f"{legend_exp_name1} (n={len(GT1)})\nAUC={auc1[0]:.3f} (95% CI {auc1[1][0]:.3f}-{auc1[1][1]:.3f})"
    ax.plot(fpr1, tpr1, color=color1, linewidth=linewidth, label=label1)

    # Shaded band under ROC 1
    n_shades = 30
    for i in range(n_shades):
        alpha_val = 0.15 * (1 - i / n_shades)
        offset = i * 0.005
        lower = np.maximum(tpr1 - offset, 0)
        ax.fill_between(fpr1, lower, tpr1 - (offset - 0.005), color=color1, alpha=alpha_val, linewidth=0)

    # ---- Second ROC ----
    fpr2, tpr2, thresholds2 = roc_curve(GT2, PR2)
    auc2 = compute_auc_metrics(GT2, PR2, print_auc=False)['auc']

    dist2 = np.sqrt(fpr2**2 + (1 - tpr2)**2)
    op_idx2 = np.argmin(dist2)
    op_fpr2, op_tpr2, op_thr2 = fpr2[op_idx2], tpr2[op_idx2], thresholds2[op_idx2]

    label2 = f"{legend_exp_name2} (n={len(GT2)})\nAUC={auc2[0]:.3f} (95% CI {auc2[1][0]:.3f}-{auc2[1][1]:.3f})"
    ax.plot(fpr2, tpr2, color=color2, linewidth=linewidth, label=label2)

    # Shaded band under ROC 2
    for i in range(n_shades):
        alpha_val = 0.15 * (1 - i / n_shades)
        offset = i * 0.005
        lower = np.maximum(tpr2 - offset, 0)
        ax.fill_between(fpr2, lower, tpr2 - (offset - 0.005), color=color2, alpha=alpha_val, linewidth=0)

    # ---- Mark Operating Points ----
    ax.plot(op_fpr1, op_tpr1, 'o', markersize=3*linewidth,
            markeredgewidth=linewidth, markeredgecolor='black',
            markerfacecolor='white',
            label=f'ePAI operating point' #(thresh={op_thr1:.2f})
            )

    ax.plot(op_fpr2, op_tpr2, 'o', markersize=3*linewidth,
            markeredgewidth=linewidth, markeredgecolor='black',
            markerfacecolor='white',
            # label=f'ePAI operating point' #(thresh={op_thr2:.2f})
            )

    # ---- Optional: Human Reader Performance ----
    if readers_performance:
        human_points = [
            {
                "label": "Specialist (n=3)",
                "sensitivity": 0.298,
                "specificity": 0.963,
                "color": "white",
                "marker": "D"  # Diamond
            },
            {
                "label": "General Radiologist (n=12)",
                "sensitivity": 0.318,
                "specificity": 0.953,
                "color": "white",
                "marker": "^"  # Triangle
            },
            {
                "label": "Resident (n=15)",
                "sensitivity": 0.294,
                "specificity": 0.942,
                "color": "white",
                "marker": "s"  # Square
            }
        ]
        for pt in human_points:
            ax.plot(
                1 - pt["specificity"], pt["sensitivity"],
                marker=pt["marker"], markersize=3*linewidth,
                markeredgewidth=linewidth, markeredgecolor='black',
                markerfacecolor=pt["color"], linestyle='None',
                label=pt["label"], zorder=10
            )

    # ---- Legend ----
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, l in unique], [l for h, l in unique], loc='lower right', frameon=False)

    # ---- Axes and Labels ----
    plt.axis('square')
    ax.tick_params(axis='both', which='major', length=axiswidth*8, width=axiswidth, pad=labelpad//2)

    # if zoomin:
    #     plt.xlim(0, 0.3)
    #     plt.ylim(0.85, 1.0)
    #     plt.xticks([0.0, 0.1, 0.2, 0.3])
    #     plt.yticks([0.9, 0.95, 1.0])
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
    #     show_legend = False
    if zoomin:
        # Define square zoom region
        plt.xlim(-0.0000001, 0.12)
        plt.ylim(0.86, 0.98)   # Extend a bit lower to make aspect visually square
        plt.xticks([0.02, 0.06, 0.1])
        plt.yticks([0.88, 0.92, 0.96])
        
        # Show only bottom and left borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Maintain square aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Remove axis labels for a clean inset look
        ax.set_xlabel('')
        ax.set_ylabel('')

        show_legend = False
    else:
        plt.xlim(-0.02, 1.0)
        plt.ylim(0., 1.02)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel('1-Specificity', labelpad=labelpad)
        ax.set_ylabel('Sensitivity', labelpad=labelpad)
        show_legend = True

    plt.grid(alpha=alpha, linewidth=linewidth)

    # ---- Save Figure ----
    os.makedirs(saverocpath, exist_ok=True)
    basename = roc_file_name + ('_zoomin' if zoomin else '')
    if show_legend:
        # deduplicate legend entries before saving
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend([h for h, l in unique], [l for h, l in unique],
                  loc='lower right', frameon=False)
    else:
        # remove legend if any exists from previous calls
        leg = ax.get_legend()
        if leg:
            leg.remove()
    
    fig.savefig(os.path.join(saverocpath, f'{basename}.png'),
                bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close(fig)
    
 
def aggregate_patient_level_results(processed_results, coi):

    # Build a dictionary to organize results by patient_id
    patient_data = defaultdict(dict)

    for row in processed_results:
        bdmap_id = row['bdmap_id']
        prediction = float(row['prediction'])
        logit = float(row['logit'])
        confidence = float(row['confidence'])
        gt = int(row['gt'])

        phase = bdmap_id.split('_')[1][0]  # 'A' or 'V'
        patient_id = 'BDMAP_' + bdmap_id.split('_')[1][1:]

        patient_data[patient_id][phase] = {
            'prediction': prediction,
            'logit': logit,
            'confidence': confidence,
            'gt': gt
        }

    # Choose the appropriate phase per coi
    aggregated_results = []
    preferred_phase = 'V' if 'pdac' in coi else 'A'
    fallback_phase = 'A' if preferred_phase == 'V' else 'V'

    for patient_id, phases in patient_data.items():
        if preferred_phase in phases:
            data = phases[preferred_phase]
        elif fallback_phase in phases:
            data = phases[fallback_phase]
        else:
            continue  # skip if neither phase is available

        aggregated_results.append({
            'patient_id': patient_id,
            'prediction': data['prediction'],
            'logit': data['logit'],
            'confidence': data['confidence'],
            'gt': data['gt']
        })
    
    return aggregated_results

def plot_scaling_result(auc_data, axiswidth=5, linewidth=15, fontsize=80, figsize=(40, 40),
                        capsize=0, marker_size=50, marker_color=[237/255, 16/255, 105/255],
                        labelpad=50, legend_exp_name='', x_values=[0, 25, 50, 75, 100],
                        savepath='result/scaling', file_name='scaling_result'):

    # Styling
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = axiswidth
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Extract AUC values and confidence intervals
    auc_values = [item['auc'][0] for item in auc_data]
    conf_intervals = [item['auc'][1] for item in auc_data]
    lower_bounds = [auc - ci[0] for auc, ci in zip(auc_values, conf_intervals)]
    upper_bounds = [ci[1] - auc for auc, ci in zip(auc_values, conf_intervals)]

    # Plotting
    ax.errorbar(
        x=x_values,
        y=auc_values,
        yerr=[lower_bounds, upper_bounds],
        fmt='x',
        markersize=marker_size,
        markeredgewidth=linewidth,
        capsize=capsize,
        color=marker_color,
        ecolor=marker_color,
        elinewidth=axiswidth,
        label=legend_exp_name
    )

    if min(auc_values) > 0.95:
        y_values = [0.92, 0.94, 0.96, 0.98, 1.0]
        plt.ylim(0.92, 1.0)
    elif min(auc_values) > 0.85:
        y_values = [0.8, 0.85, 0.9, 0.95, 1.0]
        plt.ylim(0.8, 1.0)
    elif min(auc_values) > 0.65:
        y_values = [0.6, 0.7, 0.8, 0.9, 1.0]
        plt.ylim(0.6, 1.0)
    else:
        y_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        plt.ylim(0.2, 1.0)

    # Set ticks
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)

    # Set x and y limits
    plt.xlim(0, 105)
    
    # Labels
    ax.set_xlabel('Proportion of data used in training (%)', labelpad=labelpad)
    ax.set_ylabel('AUC', labelpad=labelpad)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=axiswidth*8, width=axiswidth, pad=labelpad//2)

    ax.legend(loc='lower right', frameon=False)

    # Layout and save
    plt.tight_layout()
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(os.path.join(savepath, f'{file_name}.png'), 
                bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close(fig)

def plot_scaling_ood_result(auc_data, axiswidth=5, linewidth=15, fontsize=80, figsize=(40, 40),
                        capsize=0, marker_size=50, marker_color=[237/255, 16/255, 105/255],
                        labelpad=50, legend_exp_name='', x_values=[0, 25, 50, 75, 100],
                        savepath='result/scaling', file_name='scaling_result'):

    # Styling
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = axiswidth
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Extract AUC values and confidence intervals
    auc_values = [item['auc'][0] for item in auc_data]
    conf_intervals = [item['auc'][1] for item in auc_data]
    lower_bounds = [auc - ci[0] for auc, ci in zip(auc_values, conf_intervals)]
    upper_bounds = [ci[1] - auc for auc, ci in zip(auc_values, conf_intervals)]

    # Plotting
    ax.errorbar(
        x=x_values,
        y=auc_values,
        yerr=[lower_bounds, upper_bounds],
        fmt='x',
        markersize=marker_size,
        markeredgewidth=linewidth,
        capsize=capsize,
        color=marker_color,
        ecolor=marker_color,
        elinewidth=axiswidth,
        label=legend_exp_name
    )

    if min(auc_values) > 0.95:
        y_values = [0.92, 0.94, 0.96, 0.98, 1.0]
        plt.ylim(0.92, 1.0)
    elif min(auc_values) > 0.85:
        y_values = [0.8, 0.85, 0.9, 0.95, 1.0]
        plt.ylim(0.8, 1.0)
    elif min(auc_values) > 0.65:
        y_values = [0.6, 0.7, 0.8, 0.9, 1.0]
        plt.ylim(0.6, 1.0)
    else:
        y_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        plt.ylim(0.2, 1.0)

    # Set ticks
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)

    # Set x and y limits
    plt.xlim(0, 205)
    
    # Labels
    ax.set_xlabel('Proportion of data used in training (%)', labelpad=labelpad)
    ax.set_ylabel('AUC', labelpad=labelpad)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=axiswidth*8, width=axiswidth, pad=labelpad//2)

    ax.legend(loc='lower right', frameon=False)

    # Layout and save
    plt.tight_layout()
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(os.path.join(savepath, f'{file_name}.png'), 
                bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close(fig)