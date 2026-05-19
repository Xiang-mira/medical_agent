CSV_PATH=result/raw # Path to the standardized CSV files.
ROC_PATH=result/roc # Path to save the ROC curves.
DIAMETER_THRESHOLD=1 # Hyperparameter for diameter threshold in mm.

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\<2cm.csv,$CSV_PATH/JHH-Test-PDAC\>2cm.csv,$CSV_PATH/JHH-Test-nonPDAC\<2cm.csv,$CSV_PATH/JHH-Test-nonPDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac,pnet,cyst --gt P,P,P,P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-lesion --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-lesion --roc_legend 'Internal test cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-StageI.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-StageI --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-StageI --roc_legend 'Internal test Stage I cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-StageII.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-StageII --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-StageII --roc_legend 'Internal test Stage II cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-StageIII.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-StageIII --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-StageIII --roc_legend 'Internal test Stage III cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-StageIV.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-StageIV --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-StageIV --roc_legend 'Internal test Stage IV cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-StageIII.csv,$CSV_PATH/JHH-Test-StageIV.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-StageIII-IV --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-StageIII-IV --roc_legend 'Internal test Stage III,IV cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\<2cm.csv,$CSV_PATH/JHH-Test-PDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-PDAC --roc_legend 'Internal test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\<2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-PDAC\<2cm --roc_legend 'Internal test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-PDAC\>2cm --roc_legend 'Internal test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\<2cm.csv,$CSV_PATH/JHH-Test-PDAC\>2cm.csv,$CSV_PATH/JHH-Test-nonPDAC\<2cm.csv,$CSV_PATH/JHH-Test-nonPDAC\>2cm.csv --coi pdac --gt P,P,N,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-classification --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-classification --roc_legend 'Internal test PDAC/nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-nonPDAC\<2cm.csv,$CSV_PATH/JHH-Test-nonPDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi cyst,pnet --gt P,P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-nonPDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-nonPDAC --roc_legend 'Internal test nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-nonPDAC\<2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-nonPDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-nonPDAC\<2cm --roc_legend 'Internal test small nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-nonPDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name JHH-Test-nonPDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHH-Test-nonPDAC\>2cm --roc_legend 'Internal test large nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-Normal.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name JHHOUT-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHHOUT-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHHOUT-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHHOUT-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name JHHOUT-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name JHHOUT-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

### Northwestern cohort evaluation
python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/NW-Test-PDAC\<2cm.csv,$CSV_PATH/NW-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-Normal.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name NW-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name NW-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/NW-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name NW-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name NW-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/NW-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name NW-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name NW-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/Heidelberg-Test-PDAC.csv,$CSV_PATH/Heidelberg-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name Heidelberg-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name Heidelberg-Test-PDAC  --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name PANORAMA-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name PANORAMA-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name PANORAMA-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name PANORAMA-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name PANORAMA-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name PANORAMA-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-Prediagnostic-Latest.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-Prediagnostic --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-Prediagnostic --roc_legend 'External test prediagnostic cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-Prediagnostic.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac,cyst --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-Prediagnostic --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-Prediagnostic --roc_legend 'External test prediagnostic cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-nonPDAC\<2cm.csv,$CSV_PATH/UCSF-Test-nonPDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-nonPDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-nonPDAC --roc_legend 'External test nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-nonPDAC\<2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-nonPDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-nonPDAC\<2cm --roc_legend 'External test small nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-nonPDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-nonPDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-nonPDAC\>2cm --roc_legend 'External test large nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/ULS-Test-PDAC\<2cm.csv,$CSV_PATH/ULS-Test-PDAC\>2cm.csv,$CSV_PATH/ULS-Test-Normal.csv --coi pdac,cyst,pnet --gt P,P,N --saverocpath $ROC_PATH --roc_file_name ULS-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name ULS-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/ULS-Test-PDAC\<2cm.csv,$CSV_PATH/ULS-Test-Normal.csv --coi pdac,cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name ULS-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name ULS-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/ULS-Test-PDAC\>2cm.csv,$CSV_PATH/ULS-Test-Normal.csv --coi pdac,cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name ULS-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name ULS-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-PDAC\>2cm.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name CoH-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name CoH-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name CoH-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name CoH-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/CoH-Test-PDAC\>2cm.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name CoH-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name CoH-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/CoH-Test-NonPDAC\<2cm.csv,$CSV_PATH/CoH-Test-NonPDAC\>2cm.csv,$CSV_PATH/CoH-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,P,N --saverocpath $ROC_PATH --roc_file_name CoH-Test-NonPDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name CoH-Test-NonPDAC --roc_legend 'External test nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/CoH-Test-NonPDAC\<2cm.csv,$CSV_PATH/CoH-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name CoH-Test-NonPDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name CoH-Test-NonPDAC\<2cm --roc_legend 'External test small nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/CoH-Test-NonPDAC\>2cm.csv,$CSV_PATH/CoH-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name CoH-Test-NonPDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name CoH-Test-NonPDAC\>2cm --roc_legend 'External test large nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name SDU-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name SDU-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name SDU-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name SDU-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name SDU-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name SDU-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/SDU-Test-nonPDAC.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name SDU-Test-nonPDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name SDU-Test-nonPDAC --roc_legend 'External test nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name UW-Test-lesion --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UW-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI --debug

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UW-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UW-Test-PDAC\<2cm --roc_legend 'External test small PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UW-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UW-Test-PDAC\>2cm --roc_legend 'External test large PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UW-Test-nonPDAC\<2cm.csv,$CSV_PATH/UW-Test-nonPDAC\>2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi cyst,pnet --gt P,P,N --saverocpath $ROC_PATH --roc_file_name UW-Test-nonPDAC-lesion --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UW-Test-nonPDAC --roc_legend 'External test nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UW-Test-nonPDAC\<2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name UW-Test-nonPDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UW-Test-nonPDAC\<2cm --roc_legend 'External test small nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UW-Test-nonPDAC\>2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name UW-Test-nonPDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UW-Test-nonPDAC\>2cm --roc_legend 'External test large nonPDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/MSD-Pancreas\<2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi pdac,cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name MSD-Pancreas-Lesion\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name MSD-Pancreas-Lesion\<2cm --roc_legend 'External test small lesion cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/MSD-Pancreas\>2cm.csv,$CSV_PATH/UW-Test-Normal.csv --coi pdac,cyst,pnet --gt P,N --saverocpath $ROC_PATH --roc_file_name MSD-Pancreas-Lesion\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name MSD-Pancreas-Lesion\>2cm --roc_legend 'External test large lesion cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/ULS-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/Heidelberg-Test-Normal.csv,$CSV_PATH/ULS-Test-Normal.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv --coi pdac --gt P,P,P,P,P,N,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC <2cm cohort' --confAI

############# Overall evaluation for AUC
# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,P,P,P,P,P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC <2cm cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\>2cm --roc_legend 'External test PDAC >2cm cohort' --confAI


############# Overall evaluation updated
# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv --coi pdac --gt P,P,P,P,P,P,P,P,P,P,P,P,N,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv --coi pdac --gt P,P,P,P,P,P,N,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC <2cm cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/CoH-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv --coi pdac --gt P,P,P,P,P,P,N,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\>2cm --roc_legend 'External test PDAC >2cm cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-nonPDAC\<2cm.csv,$CSV_PATH/UCSF-Test-nonPDAC\>2cm.csv --coi pdac --gt P,P,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-classification --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-classification --roc_legend 'External test PDAC/nonPDAC cohort' --confAI

############# Internal vs External evaluation
# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\<2cm.csv,$CSV_PATH/JHH-Test-PDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --gt P,P,N --csv_path2 $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt2 P,P,P,P,P,P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name Internal-VS-External-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name Internal-VS-External-Test-PDAC --roc_legend 'Internal test PDAC cohort' --roc_legend2 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/JHH-Test-PDAC\<2cm.csv,$CSV_PATH/JHH-Test-PDAC\>2cm.csv,$CSV_PATH/JHH-Test-Normal.csv --gt P,P,N --csv_path2 $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt2 P,P,P,P,P,P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name Internal-VS-External-Test-PDAC --diameter_threshold $DIAMETER_THRESHOLD --experiment_name Internal-VS-External-Test-PDAC --roc_legend 'Internal test PDAC cohort' --roc_legend2 'External test PDAC cohort' --confAI --zoomin

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/UW-Test-PDAC\<2cm.csv,$CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC <2cm cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-PDAC\>2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\>2cm.csv,$CSV_PATH/UW-Test-PDAC\>2cm.csv,$CSV_PATH/SDU-Test-PDAC\>2cm.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/SDU-Test-Normal.csv --coi pdac --gt P,P,P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\>2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\>2cm --roc_legend 'External test PDAC >2cm cohort' --confAI

############# stage-wise evaluation
# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-StageI.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-StageI --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-StageI --roc_legend 'External test Stage I cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-StageII.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-StageII --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-StageII --roc_legend 'External test Stage II cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-StageIII.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-StageIII --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-StageIII --roc_legend 'External test Stage III cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-StageIV.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-StageIV --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-StageIV --roc_legend 'External test Stage IV cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Test-StageIII.csv,$CSV_PATH/UCSF-Test-StageIV.csv,$CSV_PATH/UCSF-Test-Normal-AllOrgans.csv --coi pdac --gt P,P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Test-StageIII-IV --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Test-StageIII-IV --roc_legend 'External test Stage III,IV cohort' --confAI

# External validation in Big Paper
# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/SDU-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/Heidelberg-Test-Normal.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/ULS-Test-Normal.csv --coi pdac --gt P,P,P,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/MSD-Pancreas\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/ULS-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/Heidelberg-Test-Normal.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/ULS-Test-Normal.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv,$CSV_PATH/CoH-Test-Normal-AllOrgans.csv --coi pdac --gt P,P,P,P,P,P,N,N,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/Heidelberg-Test-Normal.csv,$CSV_PATH/UW-Test-Normal.csv,$CSV_PATH/ULS-Test-Normal.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv,$CSV_PATH/CoH-Test-Normal-AllOrgans.csv --coi pdac --gt P,P,P,P,N,N,N,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/PANORAMA-Test-PDAC\<2cm.csv,$CSV_PATH/JHHOUT-Test-PDAC\<2cm.csv,$CSV_PATH/UCSF-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-PDAC\<2cm.csv,$CSV_PATH/CoH-Test-Normal-Pancreas.csv,$CSV_PATH/CoH-Test-Normal-AllOrgans.csv --coi pdac --gt P,P,P,P,N,N --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC cohort' --confAI

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/MSD-Pancreas\<2cm.csv,$CSV_PATH/ULS-Test-PDAC\<2cm.csv --coi pdac,cyst,pnet --gt P,P --saverocpath $ROC_PATH --roc_file_name External-Test-PDAC\<2cm --diameter_threshold $DIAMETER_THRESHOLD --experiment_name External-Test-PDAC\<2cm --roc_legend 'External test PDAC cohort' --confAI

############# reader study

# python -W ignore evaluate_tumor_detection.py --csv_path $CSV_PATH/UCSF-Reader-Study-Early-Stage-PDAC.csv,$CSV_PATH/UCSF-Reader-Study-Normal.csv --coi pdac --gt P,N --saverocpath $ROC_PATH --roc_file_name UCSF-Reader-Study --diameter_threshold $DIAMETER_THRESHOLD --experiment_name UCSF-Reader-Study --roc_legend 'ePAI performance' --confAI --readers_performance 

############# scaling

# python -W ignore scaling_result.py --csv_files JHH-Test-PDAC\<2cm,JHH-Test-PDAC\>2cm,JHH-Test-Normal --coi pdac --gt P,P,N --experiment_name JHH-Test-PDAC --scaling_legend 'Internal test PDAC cohort' --confAI

# python -W ignore scaling_result.py --csv_files JHH-Test-nonPDAC\<2cm,JHH-Test-nonPDAC\>2cm,JHH-Test-Normal --coi pnet,cyst --gt P,P,N --experiment_name JHH-Test-nonPDAC --scaling_legend 'Internal test nonPDAC cohort' --confAI

# python -W ignore scaling_result.py --csv_files JHH-Test-PDAC\<2cm,JHH-Test-PDAC\>2cm,JHH-Test-nonPDAC\<2cm,JHH-Test-nonPDAC\>2cm,JHH-Test-Normal --coi pdac,pnet,cyst --gt P,P,P,P,N --experiment_name JHH-Test-lesion --scaling_legend 'Internal test cohort' --confAI

# python -W ignore scaling_result.py --csv_files JHH-Test-PDAC\<2cm,JHH-Test-Normal --coi pdac --gt P,N --experiment_name JHH-Test-PDAC\<2cm --scaling_legend 'Internal test small PDAC cohort' --confAI

# python -W ignore scaling_result.py --csv_files JHH-Test-PDAC\>2cm,JHH-Test-Normal --coi pdac --gt P,N --experiment_name JHH-Test-PDAC\>2cm --scaling_legend 'Internal test large PDAC cohort' --confAI

# python -W ignore scaling_result.py --csv_files JHH-Test-nonPDAC\<2cm,JHH-Test-Normal --coi pnet,cyst --gt P,N --experiment_name JHH-Test-nonPDAC\<2cm --scaling_legend 'Internal test small nonPDAC cohort' --confAI

# python -W ignore scaling_result.py --csv_files JHH-Test-nonPDAC\>2cm,JHH-Test-Normal --coi pnet,cyst --gt P,N --experiment_name JHH-Test-nonPDAC\>2cm --scaling_legend 'Internal test large nonPDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files Heidelberg-Test-PDAC,Heidelberg-Test-Normal --coi pdac --gt P,N --experiment_name Heidelberg-Test-PDAC --scaling_legend 'Heidelberg test PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files JHHOUT-Test-PDAC\<2cm,JHH-Test-Normal --coi pdac --gt P,N --experiment_name JHHOUT-Test-PDAC\<2cm --scaling_legend 'JHHOUT test small PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files JHHOUT-Test-PDAC\>2cm,JHH-Test-Normal --coi pdac --gt P,N --experiment_name JHHOUT-Test-PDAC\>2cm --scaling_legend 'JHHOUT test large PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files PANORAMA-Test-PDAC\<2cm,PANORAMA-Test-Normal --coi pdac --gt P,N --experiment_name PANORAMA-Test-PDAC\<2cm --scaling_legend 'PANORAMA test small PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files PANORAMA-Test-PDAC\>2cm,PANORAMA-Test-Normal --coi pdac --gt P,N --experiment_name PANORAMA-Test-PDAC\>2cm --scaling_legend 'PANORAMA test large PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files SDU-Test-PDAC\<2cm,SDU-Test-Normal --coi pdac --gt P,N --experiment_name SDU-Test-PDAC\<2cm --scaling_legend 'SDU test small PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files SDU-Test-PDAC\>2cm,SDU-Test-Normal --coi pdac --gt P,N --experiment_name SDU-Test-PDAC\>2cm --scaling_legend 'SDU test large PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files SDU-Test-nonPDAC,SDU-Test-Normal --coi pnet,cyst --gt P,N --experiment_name SDU-Test-nonPDAC --scaling_legend 'SDU test nonPDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files UCSF-Test-PDAC\<2cm,UCSF-Test-Normal-AllOrgans --coi pdac --gt P,N --experiment_name UCSF-Test-PDAC\<2cm --scaling_legend 'UCSF test small PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files UCSF-Test-PDAC\>2cm,UCSF-Test-Normal-AllOrgans --coi pdac --gt P,N --experiment_name UCSF-Test-PDAC\>2cm --scaling_legend 'UCSF test large PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files UCSF-Test-nonPDAC\<2cm,UCSF-Test-Normal-AllOrgans --coi pnet,cyst --gt P,N --experiment_name UCSF-Test-nonPDAC\<2cm --scaling_legend 'UCSF test small nonPDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files UCSF-Test-nonPDAC\>2cm,UCSF-Test-Normal-AllOrgans --coi pnet,cyst --gt P,N --experiment_name UCSF-Test-nonPDAC\>2cm --scaling_legend 'UCSF test large nonPDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files ULS-Test-PDAC\<2cm,ULS-Test-Normal --coi pdac --gt P,N --experiment_name ULS-Test-PDAC\<2cm --scaling_legend 'ULS test small PDAC cohort' --confAI

# python -W ignore scaling_result_ood.py --csv_files ULS-Test-PDAC\>2cm,ULS-Test-Normal --coi pdac --gt P,N --experiment_name ULS-Test-PDAC\>2cm --scaling_legend 'ULS test large PDAC cohort' --confAI