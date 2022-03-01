#!/bin/bash

#SBATCH --job-name=iivi_torch
#SBATCH --output=outfiles/iivi_torch.out.%j
#SBATCH --error=outfiles/iivi_torch.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

DAVIS_FOLDERS=("blackswan")

srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS;"
srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS_ANNOS;"
srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS_MASKS;"
srun bash -c "./msrsync -p 4 /fs/vulcan-projects/action_augment_hao/gnerv/data/DAVIS/JPEGImages/480p/ /scratch0/mgwillia/DAVIS/;"
srun bash -c "./msrsync -p 4 /fs/vulcan-projects/action_augment_hao/gnerv/data/DAVIS/Annotations/480p/ /scratch0/mgwillia/DAVIS_ANNOS/;"
srun bash -c "ls /scratch0/mgwillia;"
srun bash -c "ls /scratch0/mgwillia/DAVIS;"

srun bash -c "hostname;"
for FOLDER in ${DAVIS_FOLDERS[@]}; do
    srun bash -c "mkdir -p /vulcanscratch/mgwillia/implicit-inpainting-pytorch/results/${FOLDER}_fixed;"
    srun bash -c "mkdir -p /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed;"
    srun bash -c "python preprocess_mask.py --annotation-path /scratch0/mgwillia/DAVIS_ANNOS/$FOLDER \
                    --mask-path /scratch0/mgwillia/DAVIS_MASKS/${FOLDER} --generate-fix-mask;"
    srun bash -c "python train.py --chkpt-dir /vulcanscratch/mgwillia/implicit-inpainting-pytorch/checkpoints/${FOLDER}_fixed \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed;"
    srun bash -c "python test.py --test-dir /vulcanscratch/mgwillia/implicit-inpainting-pytorch/results/${FOLDER}_fixed \
                    --dir-video /scratch0/mgwillia/DAVIS/$FOLDER --dir-mask /scratch0/mgwillia/DAVIS_MASKS/${FOLDER}_fixed \
                    --model-restore /vulcanscratch/mgwillia/implicit-inpainting-pytorch/checkpoints/$FOLDER/checkpoint_final.index;"
done
