# The input should look like this:
# $nnUNet_eval/input/
# │── BDMAP_0000001_0000.nii.gz
# │── BDMAP_0000002_0000.nii.gz
# └── ...

export nnUNet_results="nnUNetPersonal/nnUNet_results"
export nnUNet_raw="nnUNetPersonal/nnUNet_raw"
export nnUNet_preprocessed="nnUNetPersonal/nnUNet_preprocessed"
export nnUNet_predictions="nnUNetPersonal/nnUNet_predictions"
export nnUNet_eval="nnUNetPersonal/nnUNet_eval"

GPU_ID=0 
DATASET=1339

TRAINER=nnUNetTrainer 
CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_predict -d $DATASET -i $nnUNet_eval/input/ -o $nnUNet_predictions/output/ -tr $TRAINER -d $DATASET -c 3d_fullres -f all -p nnUNetPlans --continue_prediction 
