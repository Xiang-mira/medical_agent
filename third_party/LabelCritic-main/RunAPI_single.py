import argparse
import os
import ErrorDetector as ed
import importlib
import re
import sys

# Reload the module in case it has been updated
importlib.reload(ed)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Compare two projected CT/mask pairs using LLM for label evaluation.')
parser.add_argument('--path', help='Path to the annotations')

#parser.add_argument('--good_proj', required=True,
#                    help='Path to projection of ground truth sample')
#parser.add_argument('--bad_proj', required=True,
#                    help='Path to projection of predicted sample')
parser.add_argument('--organ', required=True,
                    help='Name of the organ being evaluated')
parser.add_argument('--port', default='8000',
                    help='VLLM port to use (default: 8000)')
parser.add_argument('--dice_th', default='0.5',
                    help='Minimum dice threshold to consider')
parser.add_argument('--dice_th_max', default='0.8',
                    help='Maximum dice threshold to consider')
parser.add_argument('--csv_path', default=None,
                    help='Path to CSV for saving results')
parser.add_argument('--continuing', action='store_true',
                    help='Continue from previous run (don\'t overwrite)')
parser.add_argument('--dice_list', default=None,
                    help='Path to CSVs with precomputed dice scores')
parser.add_argument('--examples', type=int, default=0,
                    help='Number of in-context examples')
parser.add_argument('--shapeless', action='store_true', default=False,
                    help='Ignore shape evaluation for pancreas/stomach/gallbladder')
parser.add_argument('--simple_prompt_ablation', action='store_true', default=False,
                    help='Use simplified prompt for ablation study')
parser.add_argument('--no_dice_check', action='store_true', default=False,
                    help='Do not skip comparisons when 2D projection Dice is above threshold')
parser.add_argument('--no_dual_confirmation', action='store_true', default=False,
                    help='Use the original single-order prompt path instead of dual confirmation')
parser.add_argument('--conservative_dual', action='store_true', default=False,
                    help='Require both dual-confirmation answers to be decisive and order-consistent')
parser.add_argument('--skip_organ_presence_gate', action='store_true', default=False,
                    help='Diagnostic only: skip the initial organ-present-in-projection gate')
parser.add_argument('--strict_choice_prompt', action='store_true', default=False,
                    help='Diagnostic only: use a short forced-choice overlay comparison prompt')
parser.add_argument(
    '--base_url',
    type=str,
    default='http://udc-ba02-35',
    help='Base API host, without port'
)
# Parse arguments
args = parser.parse_args()

# Build base API URL
base_url = f'{args.base_url}:{args.port}/v1'
# Determine organ name
organ = args.organ
path = args.path
if path[-1] != '/':
    path += '/'
# Set CSV output file
if args.csv_path:
    if args.csv_path.endswith('.csv'):
        csv_file_path = args.csv_path
    else:
        csv_file_path = args.csv_path + f'_{organ}.csv'
else:
    csv_file_path = None

# Set dice score CSV if available
if args.dice_list:
    dice_list = os.path.join(args.dice_list, f'DSC{organ}.csv')
else:
    dice_list = None

# Call comparison function
print(f'Processing organ: {organ}')
ed.SystematicComparisonLMDeploySepFigures(
    pth=os.path.join(path,organ),
    #good_pth=args.good_proj,
    size=512,
    organ=organ,
    save_memory=True,
    solid_overlay='auto',
    multi_image_prompt_2='auto',
    dual_confirmation=not args.no_dual_confirmation,
    conservative_dual=args.conservative_dual,
    dice_check=not args.no_dice_check,
    dice_th=float(args.dice_th),
    base_url=base_url,
    csv_file=csv_file_path,
    restart=(not args.continuing),
    dice_list=dice_list,
    examples=args.examples,
    shapeless=args.shapeless,
    simple_prompt_ablation=args.simple_prompt_ablation,
    skip_organ_presence_gate=args.skip_organ_presence_gate,
    strict_choice_prompt=args.strict_choice_prompt,
    dice_threshold_max=float(args.dice_th_max)
)

'''
python RunAPI_single.py \
  --good_proj ./comparison_results/good_projection/pancreas \
  --bad_proj ./comparison_results/bad_projection/pancreas \
  --organ pancreas \
  --csv_path results/pancreas_eval.csv \
  --port 8000
'''
