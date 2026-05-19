'''
See eval.sh for usage.
'''

import os
import argparse
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def event(pid_result, gt, args):

    pid = pid_result['bdmap_id']
    # print(pid_result)

    conf = get_AI_confidence(pid_result, args)
    pr = get_AI_prediction(pid_result, args)
    logit = get_AI_logit(pid_result, args)

    # write pid, pr, conf to a CSV file
    with open(args.internal_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pid, pr, logit, conf, gt])

def main(args):
    
    auc = []
    for pct in tqdm(args.pct, total=len(args.pct), ncols=60):

        args.internal_csv_path = initialize_temporary_csv()
        csv_files = [os.path.join(args.csv_path, pct+'pct', csv_file+'.csv') for csv_file in args.csv_files]
        raw_results, gt_results = read_standardized_csv(args.gt, csv_files)
        for pid_result, gt in zip(raw_results, gt_results):
            event(pid_result, gt, args)
        
        processed_results = read_temporary_csv(args, remove_csv=True)
        if args.confAI:
            auc_metric = compute_auc_metrics(gt = [int(x['gt']) for x in processed_results if int(x['confidence']) >= 0.5],
                                             logit = [float(x['logit']) for x in processed_results if int(x['confidence']) >= 0.5],
                                             print_auc=False,
                                             )
        else:
            auc_metric = compute_auc_metrics(gt = [int(x['gt']) for x in processed_results],
                                             logit = [float(x['logit']) for x in processed_results],
                                             print_auc=False,
                                             )
        auc.append(auc_metric)
        # print(f'>> AUC metric for {pct} is {auc_metric}')
    
    plot_scaling_result(auc, legend_exp_name=args.scaling_legend+' (n={})'.format(len(gt_results)),
                        file_name=args.experiment_name,
                        x_values=[int(x) for x in args.pct],
                        )
    
if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Evaluate Tumor Detection Results")

    # Add arguments to the parser
    parser.add_argument('--csv_path', type=str, default='result/raw/scaling', help='Path to the CSV file containing results')
    parser.add_argument('--csv_files', type=str, default='JHH-Test-PDAC<2cm,JHH-Test-Normal', help='Comma-separated list of CSV files to process')
    parser.add_argument('--gt', type=str, default='P,N', help='Ground truth label to evaluate against')
    parser.add_argument('--pct', type=str, default='10,25,50,75,100', help='Use of training set percentage')
    parser.add_argument('--confAI', action='store_true', help='Use AI confidence for evaluation')
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='number of CPU core needed for this process',
                       )
    parser.add_argument('--pancreas_volume_threshold', type=float, default=1000, help='Pancreas volume threshold for tumor detection in mm^3')
    parser.add_argument('--duct_volume_threshold', type=float, default=1000, help='Duct volume threshold for tumor detection in mm^3')
    parser.add_argument('--coi', type=str, default='pdac', help='Comma-separated list of conditions of interest')
    parser.add_argument('--diameter_threshold', type=float, default=1, help='Diameter threshold for tumor detection in mm')
    parser.add_argument('--scaling_legend', type=str, default='', help='Suffix for the experiment name')
    parser.add_argument('--experiment_name', type=str, default='', help='Experiment name for saving results')

    # Parse the arguments
    args = parser.parse_args()

    # Convert the pct argument to a list of integers
    args.pct = [x.strip() for x in args.pct.split(',')]

    if args.gt is not None:
        args.gt = [x.strip() for x in args.gt.split(',')]
    
    if args.coi is not None:
        args.coi = [x.strip() for x in args.coi.split(',')]

    # Convert the csv_files argument to a list of file names
    args.csv_files = [x.strip() for x in args.csv_files.split(',')]
    
    # Check if the CSV directory exists
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Directory {args.csv_path} does not exist. Please check the path.")
    
    main(args)