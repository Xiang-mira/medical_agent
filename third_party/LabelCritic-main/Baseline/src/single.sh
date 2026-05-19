#!/bin/bash
#
#SBATCH --job-name=llava_inference
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=qwu59@jh.edu
#SBATCH --partition=intern
#SBATCH --nodelist=ccvl37

source /etc/profile.d/modules.sh
module purge
module load conda
conda activate m3d
/bin/hostname
echo "Successfully activated the conda environment for m3d."
# pip show transformers
# ================== Separate line ==================
DEVICE_ID=$1
TASK_ID=$2
ORGAN=$3
echo "DEVICE_ID: $DEVICE_ID, TASK_ID: $TASK_ID, ORGAN: $ORGAN"

# # run the inference
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run_llava_slurm.py --task $TASK_ID --organ $ORGAN
# bash -c "$COMMAND" >> $LOG 2>&1

# sbatch single.sh 0 4 kidneys & \
# sbatch single.sh 0 5 kidneys & \
# sbatch single.sh 0 4 stomach & \
# sbatch single.sh 0 5 stomach & \
# sbatch single.sh 0 6 stomach & \
# sbatch single.sh 0 7 stomach & \
# sbatch single.sh 0 6 aorta & \
# sbatch single.sh 0 7 aorta