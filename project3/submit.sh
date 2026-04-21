#!/bin/bash
#BSUB -q gpuv100
#BSUB -J pilot_02501
#BSUB -n 16
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o hpc_outputs/cfg_%J.out
#BSUB -e hpc_outputs/cfg_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.8.0

source ~/miniconda3/bin/activate computer_vison

cd ~/work3/s224225/ADLCV/project3/

python3 test_imagenet.py \
    --model-names facebook/dinov2-small-imagenet1k-1-layer google/vit-base-patch16-224 google/siglip-so400m-patch14-384\
    --rotations 0 60 120 \
    --scales 0.5 1.0 2.0 \
    --max-workers 16
