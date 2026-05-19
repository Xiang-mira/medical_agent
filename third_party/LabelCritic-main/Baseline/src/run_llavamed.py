# Author: Qilong Wu
# Institute: JHU CCVL, NUS
# Description: Use this to run error detection on the baseline LLaVA-Med model.
# Use case: CUDA_VISIBLE_DEVICES=0 python run_llavamed.py --task 4 --organs liver kidneys

#############################################################################
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
from platform import processor
import torch, os, csv, json
from tqdm import tqdm
import nibabel as nib
import numpy as np
from PIL import Image

tasks = [
    # task1, old bad 1
    {
        "file": "../tasks/bad_labels_AbdomenAtlasBeta.json",
        "part": "errors_beta_full",
        "subpart": "y1",
        "path": "/mnt/sdc/pedro/ErrorDetection/errors_beta_full",
        "label2": "Incorrect",
    },
    # task2, old bad 2
    {
        "file": "../tasks/bad_labels_nnUnet.json",
        "part": "errors_nnUnet_full",
        "subpart": "y1",
        "path": "/mnt/sdc/pedro/ErrorDetection/errors_nnUnet_full",
        "label2": "Incorrect",
    },
    # task3, old good
    {
        "file": "../tasks/good_labels_AbdomenAtlasBeta.json",
        "part": "good_labels_beta_full",
        "subpart": "y1",
        "path": "/mnt/sdc/pedro/ErrorDetection/good_labels_beta_full",
        "label2": "Correct",
    },
    # task4, new atlas y1, label2 good or bad?
    {
        "file": "../tasks/AbdomenAtlas.json",
        "part": "AbdomenAtlas",
        "subpart": "y1",
        "path": "/mnt/ccvl15/qwu59/project/Error_Detection/AnnotationVLM/data/projections_AtlasBench_beta_pro",
        "label2": "Uncertain",
    },
    # task5, new atlas y2, label2 good or bad?
    {
        "file": "../tasks/AbdomenAtlas.json",
        "part": "AbdomenAtlas",
        "subpart": "y2",
        "path": "/mnt/ccvl15/qwu59/project/Error_Detection/AnnotationVLM/data/projections_AtlasBench_beta_pro",
        "label2": "Uncertain",
    },
    # task6, new jhh y1, label2 good or bad?
    {
        "file": "../tasks/JHH.json",
        "part": "JHH",
        "subpart": "y1",
        "path": "/mnt/sdh/pedro/projections_JHHBench_nnUnet_JHH",
        "label2": "Uncertain",
    },
    # task7, new jhh y2, label2 good or bad?
    {
        "file": "../tasks/JHH.json",
        "part": "JHH",
        "subpart": "y2",
        "path": "/mnt/sdh/pedro/projections_JHHBench_nnUnet_JHH",
        "label2": "Uncertain",
    }
]

def step_1_q(organ):
    return (
        "The image I am sending is frontal projections of one CT scan, focusing on showing the bone. "
        "Look at it carefully, and answer the questions below:\n\n"

        f"Is the {organ} present within this image limits? Answer Yes or No."
    )

def step_2_q(organ):
    return (
        "The image is a frontal projection of a CT scan. "
        "The left side of the image represents the right side of the human body. "
        f"Do you think the red overlay in the image is {organ}? Answer yes or no."
    )

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
def inference(image_path, question, device, processor, mute=True):
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda().to(device)
    image = Image.open(image_path)
    image_tensor = process_images([image], processor, model.config)[0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            # if args.temperature > 0 else False,
            temperature=0.8,
            # top_p=args.top_p,
            num_beams=3,
            max_new_tokens=256,
            use_cache=True)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    answer = output
    
    if not mute:
        print("*" * 80)
        print("Question:")
        print(question)
        print("=" * 50)
        print("Answer:")
        print(answer)
        print("*" * 80)
    return answer

def check_step1(answer):
    try:
        # judge = answer.split("Q4")[1].lower()
        return "present" if "yes" in answer.lower() else "no"
    except:
        return "no"

def check_step2(answer):
    try:
        # judge = answer.split("Annotation")[1].lower()
        return "Correct" if "yes" in answer.lower() else "Incorrect"
    except:
        return "Incorrect"
    
def check_step1_label(case, organ, path="/mnt/T8/AbdomenAtlasPre"):
    if organ == "kidneys":
        temp1 = nib.load(os.path.join(path, case, "segmentations", "kidney_left.nii.gz")).get_fdata()
        temp2 = nib.load(os.path.join(path, case, "segmentations", "kidney_right.nii.gz")).get_fdata()
        temp = temp1 + temp2
        # temp = np.maximum(temp1, temp2)
    else:
        temp = nib.load(os.path.join(path, case, "segmentations", f"{organ}.nii.gz")).get_fdata()
    # check whether is all zero
    return "no" if np.all(temp == 0) else "present"

def append_dict_to_csv(dict_data, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=dict_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(dict_data)

def get_one_result(task, case, organ):
    question1 = step_1_q(organ)
    question2 = step_2_q(organ)
    # try:
    image1_path = os.path.join(task["path"], organ, f"{case}_ct_window_bone_axis_1.png")
    answer1 = inference(image1_path, question1, device, processor)
    # except:
    #     image1_path = os.path.join(task["path"], organ, f"{case}_ct_window_bone_axis_1_{organ}.png")
    #     answer1 = inference(image1_path, question1, device, processor)
    image2_path = os.path.join(task["path"], organ, f"{case}_overlay_window_bone_axis_1_{organ}_{task['subpart']}.png")
    answer2 = inference(image2_path, question2, device, processor)
    judge1 = check_step1(answer1)
    judge2 = check_step2(answer2)
    label1 = check_step1_label(case, organ) # "BDMAP_00000055"
    label2 = task["label2"]
    
    task_raw = {
        "sample": case,
        "organ": organ,
        "part": task["part"] + "_" + task["subpart"],
        "question1": question1,
        "answer1": answer1,
        "question2": question2,
        "answer2": answer2,
    }
    task_single = {
        "sample": case,
        "organ": organ,
        "part": task["part"] + "_" + task["subpart"],
        "result step 1": judge1,
        "label step 1": label1,
        "result step 2": judge2,
        "label step 2": label2,
    }
    return task_raw, task_single

if __name__ == "__main__":
    # Organs: ['postcava', 'kidney_right', 'liver', 'pancreas', 'stomach', 'kidneys', 'kidney_left', 'spleen', 'gall_bladder', 'aorta']
    # ***** Example: *****
    # CUDA_VISIBLE_DEVICES=0 python run_llavamed.py --task 4 --organs liver kidneys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda") # cuda:1
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--organs", nargs='+', default=["liver", "kidney", "spleen"])
    args = parser.parse_args()
    
    device = args.device
    model_path = '/mnt/sdh/qwu59/ckpts/llava-med-v1.5-mistral-7b'
    result_path = "../results/llava-med/"
    
    # processor = LlavaNextProcessor.from_pretrained(model_path)
    # model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    from llava.model.builder import load_pretrained_model
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b',
        device=device,
    )
    # model = model.to(device)
    
    for i, j in enumerate(tasks):
        if args.task == i + 1:
            task = j
            break
        
    with open(task["file"]) as f:
        task_data = json.load(f)
        
    for organ in tqdm(task_data):
        for case in tqdm(task_data[organ]):
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
            
            # inference new case
            task_raw, task_single = get_one_result(task, case, organ)
            print(task_single)
            append_dict_to_csv(task_raw, os.path.join(result_path, "raw", f"{task['part']}_{task['subpart']}.csv"))
            append_dict_to_csv(task_single, os.path.join(result_path, "final", f"{task['part']}_{task['subpart']}.csv"))
    