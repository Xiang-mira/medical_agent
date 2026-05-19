#Example usage: python3 RunErrorDetection.py --path /mnt/sdg/pedro/compose_BetaGT_ProGT/ --port 9000 --organ_list all --dice_check > compose_BetaGT_ProGT_Error_Detection.log 2>&1

import argparse
import os
import ErrorDetector as ed
import importlib
import re

# Reload the module in case it has been updated
importlib.reload(ed)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run SystematicComparisonLMDeploySepFigures with the specified path.')
parser.add_argument('--path', help='Path to the annotations')
parser.add_argument('--port', help='VLLM port to use', default='8000')
parser.add_argument('--file_structure', help='file_structure: dual (good and bad annotations in one folder, y2 assumed to be the good one) or all_good (only takes y2 samples) or dual_bad (assumes y1 and y2 to be bad) or pick_bad_only (picks only y1) or pick_good_only (picks only y2)', default='dual')
parser.add_argument('--organ_list', help='List of organs to process', default='auto')
parser.add_argument('--size', help='Size of the images', default=512)
parser.add_argument('--dice_check', action='store_true',
                    help='Whether to check for dice similarity', default=False)
parser.add_argument('--dice_threshold', help='Threshold for dice similarity', default=0.5)
parser.add_argument('--examples', help='Number of examples to use for in-context learning.', default='0')
parser.add_argument('--limit', help='Maximum number of images analyzed', default='10000')
parser.add_argument('--skip_good', action='store_true', default=False)
parser.add_argument('--skip_bad', action='store_true', default=False)
parser.add_argument('--continuing', action='store_true', default=False)
parser.add_argument('--good_examples_pth',  default=None)
parser.add_argument('--bad_examples_pth',  default=None)
parser.add_argument('--dice_list',  default=None)
parser.add_argument('--csv_path',  default=None)
parser.add_argument('--shapeless',  action='store_true', default=False, help='Ignores shape of gallbladder, stomach and pancreas')
parser.add_argument('--simple_prompt_ablation', action='store_true', default=False)

all_organs=['aorta','liver','kidneys','spleen','pancreas','postcava','stomach','gall_bladder']

# Parse the arguments
args = parser.parse_args()

if '.csv' in args.csv_path:
    args.csv_path=args.csv_path[:-4]

# Extract the organ from the path
path = args.path
if path[-1] != '/':
    path += '/'

if args.organ_list == 'auto':
    organs = [os.path.basename(os.path.normpath(path))]
    if organs[0] not in all_organs:
        organs = [file for file in os.listdir(path) if (('right' not in file) and ('left' not in file))]
elif args.organ_list == 'all':
    organs = [file for file in os.listdir(path) if (('right' not in file) and ('left' not in file))]
else:
    organs = re.findall(r'\w+', args.organ_list)

if args.file_structure == 'auto':
    if 'good' in path.lower() or 'correct' in path.lower():
        args.file_structure = 'all_good'
    elif 'bad' in path.lower() or 'error' in path.lower():
        args.file_structure = 'all_bad'
    else:
        args.file_structure = 'dual'

print('file structure:', args.file_structure)

base_url = 'http://0.0.0.0:8000/v1'.replace('8000', args.port)

for organ in organs:
    if 'right' in organ or 'left' in organ:
        continue
    if organ not in all_organs:
        continue

    print('PROCESSING ORGAN: ', organ)
    if organ not in path:
        pth = os.path.join(path, organ)
    else:
        pth = path
    if args.dice_list is not None:
        dice_list = os.path.join(args.dice_list, 'DSC'+organ+'.csv')
    else:
        dice_list = None
    # Call the function with the extracted organ and provided path
    print(pth)
    print(os.listdir(pth)[:10])
    if args.examples=='0':
        print('Zero-shot')
        ed.ZeroShotErrorDetectionSystematicEvalLMDeploy(
            pth=pth,
            size=int(args.size),
            organ=organ,
            save_memory=True,
            solid_overlay='auto',
            base_url=base_url,
            file_structure=args.file_structure,
            dice_check=args.dice_check,
            dice_threshold=float(args.dice_threshold),
            limit=int(args.limit),
            skip_bad=args.skip_bad,skip_good=args.skip_good,
            csv_file=args.csv_path+organ+'.csv',
            dice_list=dice_list,
            shapeless=args.shapeless,
            simple_prompt_ablation=args.simple_prompt_ablation
            )
    else:
        if args.simple_prompt_ablation:
            raise ValueError('Simplified prompt ablation is not compatible with few-shot error detection.')
        print('Few-shot')
        ed.FewShotErrorDetectionSystematicEvalLMDeploy(
            n=int(args.examples),
            pth=pth,
            size=int(args.size),
            organ=organ,
            save_memory=True,
            solid_overlay='auto',
            base_url=base_url,
            file_structure=args.file_structure,
            dice_check=args.dice_check,
            dice_threshold=float(args.dice_threshold),
            limit=int(args.limit),
            skip_bad=args.skip_bad,skip_good=args.skip_good,
            good_examples_path=args.good_examples_pth,
            bad_examples_path=args.bad_examples_pth,
            csv_file=args.csv_path+organ+'.csv',
            restart=(not args.continuing),
            dice_list=dice_list,
            shapeless=args.shapeless
            )