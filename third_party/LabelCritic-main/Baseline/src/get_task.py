# Author: Qilong Wu
# Institute: JHU CCVL, NUS
# Description: Use this to get json tasks(Atlas, JHH) planner.
# Use case: python get_task.py

#############################################################################
import os, json
from tqdm import tqdm

# The logic here is to collect all ct cases by targets or labels in a json
# For 2D, I collect by whether ori,y1,y2 figures all present in the folder by case.
# For 3D, I assume that they correspond to 2D cases in the same preprocessing way.
def get_2d_cases(proj_path, save_path):
    cases = {}
    weirdo = {} # save cases with missing figures
    for organ in tqdm(os.listdir(proj_path)):
        cases[organ] = []
        weirdo[organ] = {}
        for case in os.listdir(os.path.join(proj_path, organ)):
            case_id = "BDMAP_" + case.split("BDMAP_")[1].split("_")[0]
            # judge whether all 8 figures exist
            if case_id not in weirdo[organ]:
                weirdo[organ][case_id] = 1
            else:
                weirdo[organ][case_id] += 1
                if weirdo[organ][case_id] == 8:
                    cases[organ].append(case_id)
                    del weirdo[organ][case_id]
        if weirdo[organ] == {}:
            del weirdo[organ]
        cases[organ] = sorted(cases[organ])
    if weirdo != {}:
        print("Weirdo cases exists! As follow: ", weirdo)
    # print(cases.keys())
    with open(save_path, "w") as f:
        json.dump(cases, f, indent=4)

# 1. for abdomenatlas y1 and y2 (task 4 and 5)
"""
    2D: 
        Figures: /mnt/sdg/pedro/data/projections_AtlasBench_beta_pro
        Zip: /ccvl/net/ccvl15/pedro/projections_AtlasBench_beta_pro.tar.gz
        Copy: /mnt/ccvl15/qwu59/project/error_detect/AnnotationVLM/data/projections_AtlasBench_beta_pro
        **Note: Copy in ccvl23, Figures in ccvl19, Zip in ccvl15
    3D: 
        (1) y1
        Image & Mask: /mnt/sdh/pedro/AbdomenAtlasBeta
        (2) y2
        Image & Mask: /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0Mini/AbdomenAtlas1.0Mini
        **Note: ccvl23, for both y1 and y2
"""
atlas_2d = "/mnt/ccvl15/qwu59/project/error_detect/AnnotationVLM/data/projections_AtlasBench_beta_pro"
atlas_2d_save = "../tasks/AbdomenAtlas.json"
get_2d_cases(atlas_2d, atlas_2d_save)

# 2. for jhh y1 and y2 (task 6 and 7)
""" 
    2D: 
        Figures: /mnt/sdh/pedro/projections_JHHBench_nnUnet_JHH
        **Note: ccvl23, for both y1 and y2
    3D: 
        (1) y1
        Image: /mnt/T9/AbdomenAtlasPro
        Mask: /mnt/sdc/pedro/JHH/nnUnetResults
        (2) y2
        Image: /mnt/T9/AbdomenAtlasPro
        Mask: /mnt/T8/AbdomenAtlasPre
        **Note: ccvl23, for both y1 and y2
"""
jhh_2d = "/mnt/sdh/pedro/projections_JHHBench_nnUnet_JHH"
jhh_2d_save = "../tasks/JHH.json"
get_2d_cases(jhh_2d, jhh_2d_save)