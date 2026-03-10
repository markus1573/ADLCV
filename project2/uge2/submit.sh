#!/bin/bash
#BSUB -q gpua100
#BSUB -J week5_cfg/02501
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o hpc_outputs/cfg_%J.out
#BSUB -e hpc_outputs/cfg_%J.err
#BSUB -B
#BSUB -N

module load python3/3.12.4
module load cuda/12.8.0

source /work3/s214643/venvs/ex22_02501/bin/activate

cd ~/Desktop/ADLCV/project2/uge2/

python3 ddpm_train.py --cfg