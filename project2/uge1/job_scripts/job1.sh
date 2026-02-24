#!/bin/bash
#BSUB -J adlcv_project2              # Job name
#BSUB -o job_scripts/output.%J.out            # Standard output file (%J = Job ID)
#BSUB -e job_scripts/error.%J.err             # Standard error file
#BSUB -n 4
#BSUB -R "span[hosts=1]"                  # Number of CPU cores
#BSUB -R "rusage[mem=4GB]"       # Memory per core in MB
#BSUB -W 5:00                    # Walltime (HH:MM)
#BSUB -q gpuv100                   # Queue name
#BSUB -gpu "num=1:mode=exclusive_process"

source /zhome/5f/a/186998/miniconda3/bin/activate computer_vison

cd ~/Documents/ADLCV/project2/uge1

python ddpm_train.py

