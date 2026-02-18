#!/bin/bash
#BSUB -q gpuv100
#BSUB -J week2_simple/02501
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o hpc_outputs/simple_%J.out
#BSUB -e hpc_outputs/simple_%J.err
#BSUB -B
#BSUB -N

module load python3/3.11.9
module load cuda/12.1

source /work3/s224384/ADLCV/.venv/bin/activate

cd /work3/s224384/ADLCV/project1/uge2/

python3 ./image_classification.py