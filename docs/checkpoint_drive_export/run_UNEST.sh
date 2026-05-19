# cd renalStructures_UNEST_segmentat
# Please put your CT files in ./dataset, and it should look like this:
# $./dataset/
# │── BDMAP_0000001_0000.nii.gz
# │── BDMAP_0000002_0000.nii.gz
# └── ...
# remember change the "bundle_root" in this file: ./configs/inference.json

export PYTHONPATH=$PYTHONPATH:"'/projects/bodymaps/Xinze/bundles/renalStructures_UNEST_segmentation/scripts'" # please change this path to your own path

python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
