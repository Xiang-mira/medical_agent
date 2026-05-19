#!/bin/bash

# Define input variables
organ=""
annotation_vlm_root=""
error_detection_root=""
port=""

# Parsing input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --organ) organ="$2"; shift ;;
        --annotation_vlm_root) annotation_vlm_root="$2"; shift ;;
        --error_detection_root) error_detection_root="$2"; shift ;;
        --port) port="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Ensure required arguments are provided
if [[ -z "$organ" || -z "$annotation_vlm_root" || -z "$error_detection_root" || -z "$port" ]]; then
    echo "Usage: $0 --organ <organ_name> --annotation_vlm_root <path_to_annotation_vlm> --error_detection_root <path_to_error_detection> --port <port_number>"
    exit 1
fi

# Navigate to the AnnotationVLM root directory
cd "$annotation_vlm_root"

# Reset log and run the first command
> ED_"$organ"_BadBetaZeroShot.log
python3 RunErrorDetection.py --path "$error_detection_root/errors_beta_full/" --port "$port" --organ "$organ" \
--file_structure auto --examples 0 > ED_"$organ"_BadBetaZeroShot.log 2>&1

# Reset log and run the second command
> ED_"$organ"_BadnnUnetZeroShot.log
python3 RunErrorDetection.py --path "$error_detection_root/errors_nnUnet_full/" --port "$port" --organ "$organ" \
--file_structure auto --examples 0 > ED_"$organ"_BadnnUnetZeroShot.log 2>&1

# Reset log and run the third command
> ED_"$organ"_GoodBetaZeroShot.log
python3 RunErrorDetection.py --path "$error_detection_root/good_labels_beta_full/" --port "$port" --organ "$organ" \
--file_structure auto --examples 0 > ED_"$organ"_GoodBetaZeroShot.log 2>&1

# Reset log and run the fourth command
> ED_"$organ"_BadBeta2Shot.log
python3 RunErrorDetection.py --path "$error_detection_root/errors_beta_full/" --port "$port" --organ "$organ" \
--examples 2 --good_examples_pth "$error_detection_root/good_labels_beta_full/$organ/" \
--bad_examples_pth "$error_detection_root/errors_beta_full/$organ/" \
--file_structure auto > ED_"$organ"_BadBeta2Shot.log 2>&1

# Reset log and run the fifth command
> ED_"$organ"_BadnnUnet2Shot.log
python3 RunErrorDetection.py --path "$error_detection_root/errors_nnUnet_full/" --port "$port" --organ "$organ" \
--examples 2 --good_examples_pth "$error_detection_root/good_labels_beta_full/$organ/" \
--bad_examples_pth "$error_detection_root/errors_nnUnet_full/$organ/" \
--file_structure auto > ED_"$organ"_BadnnUnet2Shot.log 2>&1

# Reset log and run the sixth command
> ED_"$organ"_GoodBeta2Shot.log

# Check if the errors_nnUnet_full directory for the organ is empty
if [ -z "$(ls -A "$error_detection_root/errors_nnUnet_full/$organ/")" ]; then
    bad_examples_pth="$error_detection_root/errors_beta_full/$organ/"
else
    bad_examples_pth="$error_detection_root/errors_nnUnet_full/$organ/"
fi

python3 RunErrorDetection.py --path "$error_detection_root/good_labels_beta_full/" --port "$port" --organ "$organ" \
--examples 2 --good_examples_pth "$error_detection_root/good_labels_beta_full/$organ/" \
--bad_examples_pth "$bad_examples_pth" \
--file_structure auto > ED_"$organ"_GoodBeta2Shot.log 2>&1