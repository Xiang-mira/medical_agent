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
parser.add_argument('--dice_th', help='VLLM port to use', default='0.5')
parser.add_argument('--dice_th_max', help='VLLM port to use', default='0.8')
parser.add_argument('--organ_list', help='List of organs to process', default='auto')
parser.add_argument('--csv_path', help='path of csv to save results', default=None)
parser.add_argument('--continuing', action='store_true', help="Continues from interrupted run.")
parser.add_argument('--dice_list', help='path of csvs with dice scores', default=None)
parser.add_argument('--examples', help='number of examples for in-context learning', default=0,type=int)
parser.add_argument('--shapeless',  action='store_true', default=False, help='Ignores shape of gallbladder, stomach and pancreas')
parser.add_argument('--simple_prompt_ablation', action='store_true', default=False)

# Parse the arguments
args = parser.parse_args()

all_organs=['aorta','liver','kidneys','spleen','pancreas','postcava','stomach','gall_bladder']

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
elif args.organ_list == 'all_shapeless':
    organs = ['pancreas','stomach','gall_bladder']
else:
    organs = re.findall(r'\w+', args.organ_list)


# Extract the organ from the path
path = args.path
if path[-1] != '/':
    path += '/'
#organ = os.path.basename(os.path.normpath(path))

base_url = 'http://0.0.0.0:8000/v1'.replace('8000', args.port)

if '.csv' in args.csv_path:
    args.csv_path=args.csv_path[:-4]


for organ in organs:
    print('PROCESSING ORGAN: ', organ)
    # Call the function with the extracted organ and provided path
    if args.dice_list is not None:
        dice_list = os.path.join(args.dice_list, 'DSC'+organ+'.csv')
    else:
        dice_list = None
    ed.SystematicComparisonLMDeploySepFigures(
        pth=os.path.join(path,organ),
        size=512,
        organ=organ,
        dice_check=True,
        save_memory=True,
        solid_overlay='auto',
        multi_image_prompt_2='auto',
        dual_confirmation=True,
        conservative_dual=False,
        dice_th=float(args.dice_th),
        base_url=base_url,
        csv_file=args.csv_path+organ+'.csv',
        restart=(not args.continuing),
        dice_list=dice_list,
        examples=args.examples,
        shapeless=args.shapeless,
        simple_prompt_ablation=args.simple_prompt_ablation,
        dice_threshold_max=float(args.dice_th_max)
    )