#!/bin/bash
#BSUB -q gpua10
#BSUB -J adlcv
#BSUB -n 16
#BSUB -R "rusage[mem=5GB]"
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

# cd /work3/s224225/ADLCV/project3/

python3 feature_analysis.py \
    --angles {0..360..10} \