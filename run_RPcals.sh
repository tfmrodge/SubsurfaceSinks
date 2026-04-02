#!/bin/bash
#SBATCH --time=02:59:0
#SBATCH --account=def-rscholes #def-agiang01 #
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=7500M
#SBATCH --array=1-9
#SBATCH --job-name='run_RPcals'
# Log files (one per array task)
#SBATCH --output=/home/tfmrodge/scratch/RPColumns/outlogs/%x_%A_%a.out
#SBATCH --error=/home/tfmrodge/scratch/RPColumns/outlogs/%x_%A_%a.err
#salloc --time=3:0:0 --mem-per-cpu=10G --nodes=1 --ntasks=10 --account=def-rscholes
module load python/3.12.4 scipy-stack/2025a
source ~/bcenv/bin/activate
python /home/tfmrodge/projects/def-rscholes/SubsurfaceSinks/calibrate_RPcolumns.py $SLURM_ARRAY_TASK_ID
#/home/tfmrodge/scratch/RPColumns/outlogs