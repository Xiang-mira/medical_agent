# A Hybrid Paradigm Integrate Self-attention and Convolution on 3D Medical Image Segmentation

[[`Paper`](#)] [[`Dataset`](https://amos22.grand-challenge.org/)] [[`BibTeX`](#)]

![Variable-Shape design](assets/fig01.jpg?raw=true)


The **VSmTrans** is a hybrid Transformer that tightly integrates self-attention and convolution into one paradigm, which enjoy the benefits of a large receptive field and strong inductive bias from both sides.

## Installation

The code requires `python>=3.9`, as well as `pytorch>1.12`. (Skip the following) [Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.]

Install nnUNet:
```
cd nnUNet
pip install -e .
pip install einops timm monai
```

Our model relies on the nnU-Net framework, and its needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to set a few environment variables. Please follow the instructions

```
export nnUNet_raw="raw data path"  (Used during training)
export nnUNet_preprocessed="preprocessed data path"  (Used during training)
export nnUNet_results="result path"  (necessarily)
```
so please set the nnUNet_results variable first. The reference command is:
```
export nnUNet_results="/path/to/nnunet_results"
```
We have provided this directory in our file, you can easily find it.

**IMPORTANT** : If you only want to test our checkpoint, you can directly jump to the __Test Our Checkpoint__ section.

## Dataset Format
Datasets must be located in the nnUNet_raw folder (which you either define when installing nnU-Net or export/set every time you intend to run nnU-Net commands!). Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit integer, and a dataset name (which you can freely choose): For example, Dataset005_Prostate has 'Prostate' as dataset name and the dataset id is 5. Datasets are stored in the nnUNet_raw folder like this:
```
nnUNet_raw/
├── Dataset001_BrainTumour
├── Dataset002_Heart
├── Dataset003_Liver
├── Dataset004_Hippocampus
├── Dataset005_Prostate
├── ...
```
Within each dataset folder, the following structure is expected:
```
Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
├── imagesTs  # optional
└── labelsTr
```
Json format details please refer to [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). 

**Note that the naming of each data is required to specify the input modal type**, e.g. BraTS. This dataset hat four input channels: FLAIR (0000), T1w (0001), T1gd (0002) and T2w (0003).
```
nnUNet_raw/Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── BRATS_001_0002.nii.gz
│   ├── BRATS_001_0003.nii.gz
│   ├── BRATS_002_0000.nii.gz
│   ├── BRATS_002_0001.nii.gz
│   ├── BRATS_002_0002.nii.gz
│   ├── BRATS_002_0003.nii.gz
│   ├── ...
├── imagesTs
│   ├── BRATS_485_0000.nii.gz
│   ├── BRATS_485_0001.nii.gz
│   ├── BRATS_485_0002.nii.gz
│   ├── BRATS_485_0003.nii.gz
│   ├── BRATS_486_0000.nii.gz
│   ├── BRATS_486_0001.nii.gz
│   ├── BRATS_486_0002.nii.gz
│   ├── BRATS_486_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BRATS_001.nii.gz
    ├── BRATS_002.nii.gz
    ├── ...
```
Here is another example of the second dataset of the MSD, which has only one input channel:
```
nnUNet_raw/Dataset002_Heart/
├── dataset.json
├── imagesTr
│   ├── la_003_0000.nii.gz
│   ├── la_004_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── la_001_0000.nii.gz
│   ├── la_002_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── la_003.nii.gz
    ├── la_004.nii.gz
    ├── ...
```
Remember: For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also make sure that all your data is co-registered!




self.network = VSmixTUnet(
    in_channels=1,
    out_channels=16,
    feature_size=24,
    split_size=[1, 3, 5, 7],
    window_size=6,
    num_heads=[3, 6, 12, 24],
    img_size=[96, 96, 96],
    depths=[2, 2, 2, 2],
    patch_size=(2, 2, 2),
    do_ds=True
)

## Experiment planning and preprocessing
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 3d_fullres
```
Where DATASET_ID is the dataset id (duh). it is recommended that you use the --verify_dataset_integrity command. This will check for some of the most common error sources! For more information about all the options available to you please run nnUNetv2_plan_and_preprocess -h.

## Training
```
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [additional options, see -h]
```
DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validation is trained. More details please refer [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

## Inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_fullres -f FOLD [--save_probabilities]
```

Note that per default, inference will be done with all 5 folds from the cross-validation as an ensemble. We very strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference.

If you wish to make predictions with a single model, train the all fold and specify it in `nnUNetv2_predict` with `-f all`

You can also run it directly, for the default configuration see [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework.

## Model
Our model exists in the directory: `nnUNet/nnunetv2/training/nnUNetTrainer/VSmTrans.py`. You can easily to use it in own framework as follows:
```
self.network = VSmixTUnet(
    in_channels=1,
    out_channels=16,
    feature_size=48,
    split_size=[1, 2, 3, 4],
    window_size=6,
    num_heads=[3, 6, 12, 24],
    img_size=[96, 96, 96],
    depths=[2, 2, 2, 2],
    patch_size=(2, 2, 2),
    do_ds=enable_deep_supervision
)
```

## Test Our Checkpoint
Our checkpoint exists in the directory: `nnUNet_results/Dataset001_BDMAP/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`.

And our test script is in the directory: `nnUNet/predict_script.py`.

You can directly put the test data into this directory: `nnUNet_results/Dataset001_BDMAP/nnUNetTrainer__nnUNetPlans__3d_fullres/AbdomenAtlasTutorial/AbdomenAtlasTest`. Then run the following command:

```
python predict_script.py
```

The output will be saved in the directory: `nnUNet_results/Dataset001_BDMAP/nnUNetTrainer__nnUNetPlans__3d_fullres/AbdomenAtlasTutorial/AbdomenAtlasPredict`.

If you want to modify the relevant path or train our model, please refer to the content above and code for specific details.

If you have any further questions, please feel free to contact us.


ps: If you encounter this error during the running process:
```
Traceback (most recent call last):
  ...
  File "/home/kemove/miniconda3/envs/VSmTrans/lib/python3.9/site-packages/acvl_utils/cropping_and_padding/bounding_boxes.py", line 5, in <module>
    import blosc2
ModuleNotFoundError: No module named 'blosc2'
```
The reason may be a version issue with the code. You can replace it with the file we provided, which is located in this directory: `nnUNet/bounding_boxes.py`.  

Another solution: Downgrading acvl_utils to version 0.2 gets rid of the initial error. 
```
pip install --upgrade acvl_utils==0.2
```

## Citation
```
@article{liu2024vsmtrans,
  title={VSmTrans: A hybrid paradigm integrating self-attention and convolution for 3D medical image segmentation},
  author={Liu, Tiange and Bai, Qingze and Torigian, Drew A and Tong, Yubing and Udupa, Jayaram K},
  journal={Medical Image Analysis},
  volume={98},
  pages={103295},
  year={2024},
  publisher={Elsevier}
}
```


