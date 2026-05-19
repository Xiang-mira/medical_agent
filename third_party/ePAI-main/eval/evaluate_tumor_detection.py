'''
See eval.sh for usage.
'''

import os
import random 
import string
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

    # if CSV file already exists, remove it
    if os.path.isfile(args.internal_csv_path):
        os.remove(args.internal_csv_path)
    # create a CSV file to store the results
    with open(args.internal_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bdmap_id', 'prediction', 'logit', 'confidence', 'gt'])  # Write header
    
    # Read the CSV files
    raw_results, args.gt_results = read_standardized_csv(args.gt, args.csv_path)
    
    if args.debug:
        for pid_result, gt in tqdm(zip(raw_results, args.gt_results), total=len(raw_results), ncols=80):
            event(pid_result, gt, args)
            # try:
            #     event(pid_result, gt, args)
            # except Exception as e:
            #     print(f"Error processing {pid_result['bdmap_id']}: {e}")

    else:
        if args.num_core > 0:
            num_core = args.num_core
        else:
            num_core = int(cpu_count())
        # print('>> {} CPU cores are secured.'.format(num_core))
        with ProcessPoolExecutor(max_workers=num_core) as executor:
            futures = {executor.submit(event, pid_result, gt, args): pid_result
                    for pid_result, gt in zip(raw_results, args.gt_results)}
            
            for future in tqdm(as_completed(futures), total=len(futures), ncols=60):
                folder = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {folder}: {e}")

    # Read the results from the internal CSV file
    processed_results = read_temporary_csv(args)

    # Optional: process second dataset if provided
    processed_results2 = None
    if args.csv_path2 and args.gt2:
        print("\n>> Processing second ROC dataset...")
        random_string_2 = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        args.internal_csv_path = random_string_2 + '.csv'
        if os.path.isfile(args.internal_csv_path):
            os.remove(args.internal_csv_path)
        with open(args.internal_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['bdmap_id', 'prediction', 'logit', 'confidence', 'gt'])

        raw_results2, gt_results2 = read_standardized_csv(args.gt2, args.csv_path2)
        num_core = args.num_core if args.num_core > 0 else int(cpu_count())
        with ProcessPoolExecutor(max_workers=num_core) as executor:
            futures = {executor.submit(event, pid_result, gt, args): pid_result
                       for pid_result, gt in zip(raw_results2, gt_results2)}
            for future in tqdm(as_completed(futures), total=len(futures), ncols=60):
                folder = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {folder}: {e}")

        processed_results2 = read_temporary_csv(args)
        
    # print("Total predictions found: {}, {} are positive and {} are negative".format(len(processed_results), [int(x['gt']) for x in processed_results].count(1), [int(x['gt']) for x in processed_results].count(0)))

    if processed_results2 is not None:
        # Two ROC curves
        print("\n>> Plotting two ROC curves on the same figure...")
        plot_two_roc_curves(
            GT1=[int(x['gt']) for x in processed_results],
            PR1=[float(x['logit']) for x in processed_results],
            GT2=[int(x['gt']) for x in processed_results2],
            PR2=[float(x['logit']) for x in processed_results2],
            saverocpath=args.saverocpath,
            roc_file_name=args.roc_file_name,
            legend_exp_name1=args.roc_legend,
            legend_exp_name2=args.roc_legend2,
            axiswidth=8, linewidth=12, fontsize=80, figsize=(40, 40),
            readers_performance=args.readers_performance,
            alpha=0.0, zoomin=args.zoomin,
        )
    else:
        if args.confAI: 
            compute_detection_metrics(gt = [int(x['gt']) for x in processed_results if int(x['confidence']) >= 0.5],
                                    pr = [int(x['prediction']) for x in processed_results if int(x['confidence']) >= 0.5],
                                    )
            compute_auc_metrics(gt = [int(x['gt']) for x in processed_results if int(x['confidence']) >= 0.5],
                                logit = [float(x['logit']) for x in processed_results if int(x['confidence']) >= 0.5],
                                )
            plot_roc_curve(GT = [int(x['gt']) for x in processed_results if int(x['confidence']) >= 0.5],
                        PR = [float(x['logit']) for x in processed_results if int(x['confidence']) >= 0.5],
                        saverocpath = args.saverocpath,
                        roc_file_name=args.roc_file_name,
                        legend_exp_name=args.roc_legend,
                        axiswidth=8, linewidth=12, fontsize=80, figsize=(40, 40), 
                        readers_performance=args.readers_performance,
                        alpha=0.0, zoomin=False,
                        )
        else:
            compute_detection_metrics(gt = [int(x['gt']) for x in processed_results],
                                    pr = [int(x['prediction']) for x in processed_results],
                                    )
            compute_auc_metrics(gt = [int(x['gt']) for x in processed_results],
                                logit = [float(x['logit']) for x in processed_results],
                                )
            plot_roc_curve(GT = [int(x['gt']) for x in processed_results],
                        PR = [float(x['logit']) for x in processed_results],
                        saverocpath = args.saverocpath,
                        roc_file_name=args.roc_file_name,
                        legend_exp_name=args.roc_legend,
                        axiswidth=8, linewidth=12, fontsize=80, figsize=(40, 40), 
                        readers_performance=args.readers_performance,
                        alpha=0.0, zoomin=False,
                        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='result/raw/prediagnostic.csv', help='Path to the CSV file containing predictions')
    parser.add_argument('--coi', type=str, default=None, help='Comma-separated list of conditions of interest')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth label to evaluate against')
    parser.add_argument('--diameter_threshold', type=float, default=2, help='Diameter threshold for tumor detection in mm')
    parser.add_argument('--pancreas_volume_threshold', type=float, default=1000, help='Pancreas volume threshold for tumor detection in mm^3')
    parser.add_argument('--duct_volume_threshold', type=float, default=1000, help='Duct volume threshold for tumor detection in mm^3')
    parser.add_argument('--saverocpath', type=str, default='data_demo/roc_curve', help='Path to save the ROC curve plot')
    parser.add_argument('--roc_file_name', type=str, default='roc_curve', help='File name for the ROC curve plot')
    parser.add_argument('--confAI', action='store_true', help='Use AI confidence for evaluation')
    parser.add_argument('--debug', action='store_true', help='Turn on debug mode')
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='number of CPU core needed for this process',
                       )
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment for logging purposes')
    parser.add_argument('--roc_legend', type=str, default='', help='Suffix for the experiment name')
    parser.add_argument('--readers_performance', action='store_true', help='Use human reader points for evaluation')
    parser.add_argument('--csv_path2', type=str, default=None, help='Optional path to the second CSV file')
    parser.add_argument('--gt2', type=str, default=None, help='Optional ground truth label for second ROC')
    parser.add_argument('--roc_legend2', type=str, default='Exp 2', help='Suffix for the second experiment name')
    parser.add_argument('--zoomin', action='store_true', help='Zoom in on ROC curve plot')

    args = parser.parse_args()

    # translate coi to a list
    if args.coi is not None:
        args.coi = [x.strip() for x in args.coi.split(',')]
        # print(args.coi)

    if args.csv_path is not None:
        args.csv_path = [x.strip() for x in args.csv_path.split(',')]
        # print(args.csv_path)

    if args.gt is not None:
        args.gt = [x.strip() for x in args.gt.split(',')]
        # print(args.gt)
        
    if args.csv_path2 is not None:
        args.csv_path2 = [x.strip() for x in args.csv_path2.split(',')]
        
    if args.gt2 is not None:
        args.gt2 = [x.strip() for x in args.gt2.split(',')]

    # generate a string of random characters and numbers, lenth = 10
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    args.internal_csv_path = random_string + '.csv'
    
    if args.experiment_name is not None:
        print('\n>> {}'.format(args.experiment_name))

    main(args)
