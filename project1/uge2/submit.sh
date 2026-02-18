#!/bin/bash
#BSUB -q gpuv100
#BSUB -J week2_head2/02501
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:15
#BSUB -o hpc_outputs/head2_%J.out
#BSUB -e hpc_outputs/head2_%J.err
#BSUB -B
#BSUB -N

module load python3/3.12.4
module load cuda/12.8.0

source /work3/s214643/venvs/ex12_02501/bin/activate

cd ~/Desktop/ADLCV/project1/uge2/

python3 ./imageclassification.py --output-dir models/head2.pth --embed 128 --heads 2 --layers 4
python3 ./attention_visualize.py --output-dir ~/Desktop/project1/uge2/attention_viz/head2