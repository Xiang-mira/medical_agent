# Author: Qilong Wu
# Institute: JHU CCVL, NUS
# Description: Use this to run error detection on the baseline M3D model.
# Use case: CUDA_VISIBLE_DEVICES=0 python run_m3d.py --task 4 --organs liver kidneys

#############################################################################
import argparse
import numpy as np
import json, csv
from tqdm import tqdm
import torch, warnings, os
import nibabel as nib
from transformers import AutoTokenizer, AutoModelForCausalLM
# import custom_tf as ctf
import opti_tf as ctf
from importlib import reload
reload(ctf)

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_CACHE'] = './HFCache'
os.environ['HF_HOME'] = './HFCache'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32
model_path = '/mnt/sdh/qwu59/ckpts/m3d/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

tasks = [
    # task1, old bad 1
    {
        "file": "../tasks/bad_labels_AbdomenAtlasBeta.json",
        "part": "errors_beta_full",
        "subpart": "y1",
        "path": "/mnt/sdh/pedro/AbdomenAtlasBeta/",
        "label2": "Incorrect", 
    },
    # task2, old bad 2
    {
        "file": "../tasks/bad_labels_nnUnet.json",
        "part": "errors_nnUnet_full",
        "subpart": "y1",
        "path": "/mnt/sdc/pedro/JHH/nnUnetResults",
        "path_": "/mnt/sdc/pedro/ErrorDetection/cropped_nnunet_results_250Epch_liver",
        "label2": "Incorrect",
    },
    # task3, old good
    {
        "file": "../tasks/good_labels_AbdomenAtlasBeta.json",
        "part": "good_labels_beta_full",
        "subpart": "y1",
        "path": "/mnt/sdc/pedro/AbdomenAtlasBeta/",
        "label2": "Correct",
    },
    # task4, new atlas y1, label2 good or bad?
    {
        "file": "../tasks/AbdomenAtlas.json",
        "part": "AbdomenAtlas",
        "subpart": "y1",
        "path": "/mnt/sdh/pedro/AbdomenAtlasBeta",
        "label2": "Uncertain",
    },
    # task5, new atlas y2, label2 good or bad?
    {
        "file": "../tasks/AbdomenAtlas.json",
        "part": "AbdomenAtlas",
        "subpart": "y2",
        "path": "/mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0Mini/AbdomenAtlas1.0Mini",
        "label2": "Uncertain",
    },
    # task6, new jhh y1, label2 good or bad?
    {
        "file": "../tasks/JHH.json",
        "part": "JHH",
        "subpart": "y1",
        "path": "/mnt/T9/AbdomenAtlasPro",
        "mask_path": "/mnt/sdc/pedro/JHH/nnUnetResults",
        "label2": "Uncertain",
    },
    # task7, new jhh y2, label2 good or bad?
    {
        "file": "../tasks/JHH.json",
        "part": "JHH",
        "subpart": "y2",
        "path": "/mnt/T9/AbdomenAtlasPro",
        "mask_path": "/mnt/T8/AbdomenAtlasPre",
        "label2": "Uncertain",
    },
]

def load_model(model_path, choose, dtype):
    if choose == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={'': 'cuda'},
            trust_remote_code=True
        )
        print("Load all the model to GPU.")
    elif choose == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        print("Offload part of the model to CPU.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    # model = model.to(device=device)
    return model, tokenizer

def inference(model, tokenizer, question, ct_pro, seg_enable=False, branch='ct'):
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(
        input_txt,
        return_tensors="pt"
    )['input_ids'].to(device=device)
    attention_mask = tokenizer(
        input_txt,
        return_tensors="pt", 
        padding=True
    )['attention_mask'].to(device=device)

    if branch == 'ct':
        image_pt = ct_pro.transformed["image"].unsqueeze(0).to(device=device).type(dtype)
    elif branch == 'ctmin':
        image_pt = ct_pro.ctmin_transformed["image"].unsqueeze(0).to(device=device).type(dtype)

    try:
        outputs = model.generate(
            image_pt,
            input_id,
            attention_mask=attention_mask,
            seg_enable=seg_enable,
            max_new_tokens=32,
            do_sample=True,
            top_p=0.9,
            temperature=1.0
        )
        if seg_enable:
            generation, seg_logit = outputs
            seg_mask = ((torch.sigmoid(seg_logit) > 0.5) * 1.0).squeeze(0)
            return tokenizer.batch_decode(generation, skip_special_tokens=True)[0], seg_mask
        else:
            generation = outputs
            return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    except:
        return "The image is damaged or the model is not able to generate the answer."

def append_dict_to_csv(dict_data, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=dict_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(dict_data)
        
def step_1_q(organ):
    return "Does the bone ct image contain the {}? Answer yes or no.".format(organ)

def step_2_q(organ):
    return (
        "The lowest intensity black area within the body in this bone ct image"
        "is the {} mask annotation. ".format(organ) +
        "What do you think of this? Is it correct? Only answer yes or no."
    )
    
if __name__ == "__main__":
    # Organs: ['postcava', 'kidney_right', 'liver', 'pancreas', 'stomach', 'kidneys', 'kidney_left', 'spleen', 'gall_bladder', 'aorta']
    # ***** Example: *****
    # CUDA_VISIBLE_DEVICES=0 python run_m3d.py --task 4 --organs liver kidneys

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--organs", nargs='+', default=["liver", "kidney", "spleen"])
    parser.add_argument("--choose", type=str, default="cuda") # "cuda" or "auto"
    args = parser.parse_args()
    
    result_path = "../results/m3d/"
    model, tokenizer = load_model(model_path, args.choose, dtype)

    for i, j in enumerate(tasks):
        if args.task == i + 1:
            task = j
            break

    # load the json file for task
    with open(task["file"]) as f:
        task_data = json.load(f)

    for i, organ in enumerate(tqdm(task_data)):
        print("Organ:", organ)
        if organ not in args.organs:
            continue
        question1 = step_1_q(organ)
        question2 = step_2_q(organ)
        for j, case in enumerate(tqdm(task_data[organ])):
            # check whether the case exists in the final csv
            check_table = os.path.join(result_path, "final", f"{task['part']}_{task['subpart']}.csv")
            skip_sign = False
            if os.path.exists(check_table):
                with open(check_table, mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["sample"] == case and row["organ"] == organ:
                            skip_sign = True
                            break
            if skip_sign:
                continue
            try:
                mask_path = False
                if "mask_path" in task:
                    mask_path = os.path.join(task["mask_path"], case)
                case_path = os.path.join(task["path"], case)
                # print("Path:", case_path, mask_path)
                ct_pro = ctf.CTImageProcessor(case_path, ct_name="ct", mask_name=organ, mask_path=mask_path)
            except Exception as e:
                if "path_" in task:
                    case_path = os.path.join(task["path_"], case)
                    ct_pro = ctf.CTImageProcessor(case_path, ct_name="ct", mask_name=organ, mask_path=mask_path)
                else:
                    print("Case not found:", case, "Error:", e)
                    pass
            # Now I change the new loader so when init requires the mask path optional
            # this case is used for those image & mask are in the different folders
            # eg: ct_pro = ctf.CTImageProcessor(case_path, ct_name="ct", mask_name=organ, mask_path=mask_path)
            text1 = inference(model, tokenizer, question1, ct_pro, branch='ct')
            text2 = inference(model, tokenizer, question2, ct_pro, branch='ctmin')
            task_raw = {
                "sample": case,
                "organ": organ,
                "part": task["part"] + "_" + task["subpart"],
                "question1": question1,
                "answer1": text1,
                "question2": question2,
                "answer2": text2
            }
            task_single = {
                "sample": case,
                "organ": organ,
                "part": task["part"] + "_" + task["subpart"],
                "result step 1": "present" if "yes" in text1.lower() else "no",
                "label step 1": "present" if ct_pro.mask_present else "no",
                "result step 2": "Correct" if "yes" in text2.lower() else "Incorrect",
                "label step 2": task["label2"],
            }
            print(task_single)
            append_dict_to_csv(task_raw, os.path.join(result_path, "raw", f"{task['part']}_{task['subpart']}.csv"))
            append_dict_to_csv(task_single, os.path.join(result_path, "final", f"{task['part']}_{task['subpart']}.csv"))
