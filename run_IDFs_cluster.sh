#!/bin/bash
#SBATCH --time=71:59:0
#SBATCH --account=def-rscholes #def-agiang01 #
#SBATCH --ntasks-per-node=64 #48 # 
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3948M #10624M #
#SBATCH --array=1 #,3,5,6
#SBATCH --job-name='run_wateryear'
python run_IDFs3.py $SLURM_ARRAY_TASK_ID
#python run_IDFs1.py