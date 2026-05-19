<div align="center">

<img src="document/fig_epai_logo.png" alt="logo" width="260"/>

</div>

## 0. Create a virtual environment *[Only for First-Time User]*

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```

</div>
</details>



```bash
conda create -n ePAI python=3.10 -y
```

## 1. Installation *[Only for First-Time User]*

```bash
source activate ePAI
git clone https://github.com/BodyMaps/ePAI.git
cd ePAI/train/
pip install --upgrade setuptools packaging
pip install nnunetv2
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

<details>
<summary style="margin-left: 25px;">[Optional] If the above installation does not work for you, try this!</summary>
<div style="margin-left: 25px;">

_Delete the current ePAI environment. Then, do:_

```bash
conda create -n ePAI python=3.11 -y
source activate ePAI
git clone https://github.com/BodyMaps/ePAI.git
cd ePAI/train/
pip install --upgrade setuptools packaging
pip install nnunetv2==2.6.0
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

</div>
</details>



## 2. Train ePAI

#### 2.1 Generate json file in the path of raw data *[Only for First-Time User]*

```bash
export nnUNet_N_proc_DA=36 # number of CPU cores
export nnUNet_raw="/mnt/bodymaps/ePAI/nnUNet/raw"
export nnUNet_preprocessed="/mnt/bodymaps/ePAI/nnUNet/preprocessed"

ROOT_PATH="/mnt/T9/project/ePAI/train"
cd $ROOT_PATH
python -W ignore generate_json.py --raw_dir /mnt/bodymaps/ePAI/nnUNet/raw --dataset_name Dataset1013_ePAI_3MM 
# raw_dir: The directory saving raw data
# dataset_name: The folder name in the raw directory

nnUNetv2_plan_and_preprocess -d 1013 -npfp 64 -np 64 -c 3d_fullres
# npfp and np are CPU cores used for preprocessing
# This step will generate a nnUNetPlans.json in the $nnUNet_preprocessed/Dataset1013_ePAI_3MM directory.
```

#### 2.2 Tune training parameters *[Not Necessary If You Don't Know How to Do This]*

Modify `vi $ROOT_PATH/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

```python
### Some hyperparameters for you to fiddle with
self.initial_lr = 1e-2 
# If you increase the batch size, learning rate will have to increase as well.
self.weight_decay = 3e-5
self.oversample_foreground_percent = 0.33
self.probabilistic_oversampling = False
self.num_iterations_per_epoch = 250
# If you increase the batch size, num_iterations_per_epoch needs to be decreased, otherwise it takes too long for one epoch.
self.num_val_iterations_per_epoch = 50
self.num_epochs = 1000
self.current_epoch = 0
self.enable_deep_supervision = True
```

Modify `$nnUNet_preprocessed/Dataset1013_ePAI_3MM/nnUNetPlans.json`

```python
"batch_size": 16, # 47,276 MiB
```

#### 2.3 Now Train It!

```bash
export nnUNet_N_proc_DA=36 # number of CPU cores
export nnUNet_raw="/mnt/bodymaps/ePAI/nnUNet/raw"
export nnUNet_preprocessed="/mnt/bodymaps/ePAI/nnUNet/preprocessed"
export nnUNet_results="./runsv2"

ROOT_PATH="/mnt/T9/project/ePAI/train"
cd $ROOT_PATH
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1013 3d_fullres all
```

## 3. Inference ePAI

#### 3.1 Convert dataset (CT scans) to nnUNet format [Only for First-Time User]

Assume the original CT data is under BDMAP format:

```bash
path_to_bdmap_format_data/
│── BDMAP_0000001/
│    └── ct.nii.gz
│── BDMAP_0000002/
│    └── ct.nii.gz
└── ...
```

Then, to convert it into a nnUNet input folder, run

```bash
INPUT_BDMAP_PATH="/mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro"
OUTPUT_BDMAP_PATH="/mnt/bodymaps/ePAI/nnUNet/eval"
python -W ignore softlink2nnUNet.py --input_bdmap_path $INPUT_BDMAP_PATH --output_nnunet_path $OUTPUT_BDMAP_PATH
```

This will create softlinks of the BDMAP format data into nnUNet format:

```bash
$OUTPUT_BDMAP_PATH/
│── BDMAP_0000001_0000.nii.gz
│── BDMAP_0000002_0000.nii.gz
└── ...
```

> [!NOTE]
>
> Here, the `OUTPUT_BDMAP_PATH` will be used in the `DATA_PATH` below (Section 3.3).

Use this nnUNet format data folder for the inference process (Section 3.3).

#### 3.2 Convert dataset (ground truth segmentation masks) to nnUNet format [Only for First-Time User]

Assume the original ground truth segmentation masks are under BDMAP format:

```bash
path_to_bdmap_format_gt/
│── BDMAP_0000001/
│    └── segmentations
│        └── pancreas.nii.gz
│        └── pancreatic_duct.nii.gz
│        └── pancreatic_pdac.nii.gz
│        └── pancreatic_cyst.nii.gz
│        └── pancreatic_pnet.nii.gz
│        └── ...
│── BDMAP_0000002/
│    └── segmentations
│        └── ...
└── ...
```

where each NIfTI file is a binary (i.e., 0 means background and 1 means target class) segmentation mask of the corresponding class. 

> [!WARNING]
> Note that `pancreas`, `pancreatic_duct`, `pancreatic_pdac`, `pancreatic_cyst,` and `pancreatic_pnet` segmentation masks are **REQUIRED**, other class segmentation masks are optional.

To convert it into a nnUNet format combined segmentation mask folder, first open `bdmapLabels2nnunetLabels.py` and modify the `label_mapping` to fit your needs. 

Then, run

```bash
INPUT_BDMAP_PATH="/mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro"
INPUT_BDMAP_GT_PATH="/mnt/bodymaps/mask_only/AbdomenAtlasPro/AbdomenAtlasPro"
OUTPUT_BDMAP_GT_PATH="/mnt/bodymaps/ePAI/nnUNet/labelsTs"
python -W ignore bdmapLabels2nnunetLabels.py --input_bdmap_path $INPUT_BDMAP_PATH --input_bdmap_gt_path $INPUT_BDMAP_GT_PATH --output_nnunet_gt_path $OUTPUT_BDMAP_GT_PATH
```

This will create combined segmentation masks of the BDMAP format ground truth into nnUNet format:

```bash
$OUTPUT_BDMAP_GT_PATH/
│── BDMAP_0000001.nii.gz
│── BDMAP_0000002.nii.gz
└── ...
```

where each NIfTI file is a combined segmentation mask of each CT (e.g., in the combined segmentation mask: 0 for background, 13 for pancreas, 23 for pancreatic PDAC, 24 for pancreatic Cyst, etc.). 

> [!NOTE]
> 1. The `label_mapping` order doesn't matter, we only need to provide 5 pancreas-related class mappings in the `LABEL_IDS` below (Sec 3.3).
>
> 2. Here, the `OUTPUT_BDMAP_GT_PATH` will be used in the `LABEL_PATH` below (Section 3.3).

Use this nnUNet format segmentation mask folder for the inference process (Section 3.3).

#### 3.3 Run the inference process

**[Optional, but Important] Download the best checkpoint**

```bash
wget http://www.cs.jhu.edu/~zongwei/model/qchen76_2025_0421.tar.gz
tar -xzvf qchen76_2025_0421.tar.gz

wget http://www.cs.jhu.edu/~zongwei/model/qchen76_2025_0404.tar.gz
tar -xzvf qchen76_2025_0404.tar.gz

wget http://www.cs.jhu.edu/~zongwei/model/wli131_2024_1115.tar.gz
tar -xzvf wli131_2024_1115.tar.gz
```

##### 3.3.1 Run the inference process (NO ground truth in the output CSV file)
```bash
export nnUNet_N_proc_DA=36 # number of CPU cores for training, not important
export nnUNet_raw="/mnt/bodymaps/ePAI/nnUNet/raw" # Placefolder, not important
export nnUNet_preprocessed="/mnt/bodymaps/ePAI/nnUNet/preprocessed" # Placefolder, not important
export nnUNet_results="./runsv2" # Placefolder, not important

ROOT_PATH="/mnt/T9/project/ePAI/train"
DATA_PATH="/mnt/bodymaps/ePAI/nnUNet/eval"
SAVE_PATH="${ROOT_PATH}/out"

CKPT_PATH="${ROOT_PATH}/runsv2/Dataset1013_ePAI_3MM/nnUNetTrainer__nnUNetPlans__3d_fullres"
# Best checkpoint
# CKPT_PATH="/mnt/bodymaps/ePAI/model/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres"
# Second best checkpoint
# CKPT_PATH="/mnt/bodymaps/ePAI/model/qchen76_2025_0404/nnUNetTrainer__nnUNetPlans__3d_fullres"
# CKPT_PATH="/mnt/bodymaps/ePAI/model/wli131_2024_1115/nnUNetTrainer__nnUNetPlans__3d_fullres"

cd $ROOT_PATH
PATIENT_ID="JHH-Test-nonPDAC<2cm"
INPUT_CSV_PATH="input_csv/${PATIENT_ID}.csv"
OUTPUT_CSV_PATH="output_csv/${PATIENT_ID}.csv"

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict_from_modelfolder \
  -i $DATA_PATH \
  -o $SAVE_PATH \
  -m $CKPT_PATH \
  -f all \
  --input_csv $INPUT_CSV_PATH \
  --output_csv $OUTPUT_CSV_PATH \
  --continue_prediction \
  --save_probabilities \
  -npp 3 \
  -nps 3 \
  -num_parts 1 \
  -part_id 0 \
  -chk checkpoint_final.pth
```

The output contains two types:

1. The segmentation predictions under `$SAVE_PATH`
2. The CSV output under `OUTPUT_CSV_PATH`

##### 3.3.2 Run the inference process (INCLUDE ground truth in the output CSV file)

<details>
<summary style="margin-left: 25px;">[Optional] If you already have the inference and output CSV file ready, and only want to add ground truth into current output CSV file</summary>
<div style="margin-left: 25px;">
    
1. Make sure your output CSV file has the same column names as the ones in ePAI/train/csv_header.csv (e.g., your output CSV file should contain `pancreas_pr`).
2. Create a folder named `output_csv` inside the `ePAI/train/` folder.
3. Copy your output CSV files into the `ePAI/train/output_csv/` folder.

```bash
cd ePAI/train/
mkdir output_csv/
cp /path/to/your/csv/*.csv output_csv/
```
</div>
</details>

```bash
export nnUNet_N_proc_DA=36 # number of CPU cores for training, not important
export nnUNet_raw="/mnt/bodymaps/ePAI/nnUNet/raw" # Placefolder, not important
export nnUNet_preprocessed="/mnt/bodymaps/ePAI/nnUNet/preprocessed" # Placefolder, not important
export nnUNet_results="./runsv2" # Placefolder, not important

ROOT_PATH="/mnt/T9/project/ePAI/train"
DATA_PATH="/mnt/bodymaps/ePAI/nnUNet/eval"
SAVE_PATH="${ROOT_PATH}/out"

LABEL_PATH="/mnt/bodymaps/ePAI/nnUNet/raw/Dataset1013_ePAI_3MM/labelsTs"  # path to combined labels (nnUNet format)
LABEL_IDS=(13 14 23 24 25)  # the label ID of pancreas, duct, pdac, cyst, pnet (must contain 5 numbers and the order of 5 numbers MATTERS!!!)
# Note: if you do not have the ground truth label for pancreas/duct/pdac/cyst/pnet, please put -1 as the label ID.
# For example, if your dataset only has pdac which labeled as 1 in the step 3.2, then you can set LABEL_IDS=(-1 -1 1 -1 -1). 

CKPT_PATH="${ROOT_PATH}/runsv2/Dataset1013_ePAI_3MM/nnUNetTrainer__nnUNetPlans__3d_fullres"
# Best checkpoint
# CKPT_PATH="/mnt/bodymaps/ePAI/model/qchen76_2025_0404/nnUNetTrainer__nnUNetPlans__3d_fullres"
# Second best checkpoint
# CKPT_PATH="/mnt/bodymaps/ePAI/model/wli131_2024_1115/nnUNetTrainer__nnUNetPlans__3d_fullres"

cd $ROOT_PATH
PATIENT_ID="JHH-Test-nonPDAC<2cm"
INPUT_CSV_PATH="input_csv/${PATIENT_ID}.csv"
OUTPUT_CSV_PATH="output_csv/${PATIENT_ID}.csv"

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict_from_modelfolder \
  -i $DATA_PATH \
  -o $SAVE_PATH \
  -m $CKPT_PATH \
  -l $LABEL_PATH \
  --panc_duct_pdac_cyst_pnet "${LABEL_IDS[@]}" \
  --add_gt_info_to_csv \
  -f all \
  --input_csv $INPUT_CSV_PATH \
  --output_csv $OUTPUT_CSV_PATH \
  --continue_prediction \
  --save_probabilities \
  -npp 3 \
  -nps 3 \
  -num_parts 1 \
  -part_id 0 \
  -chk checkpoint_final.pth
```

The output contains two types:

1. The segmentation predictions under `$SAVE_PATH`
2. The CSV output under `OUTPUT_CSV_PATH`
