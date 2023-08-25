#!/bin/bash
#SBATCH --time=23:59:0
#SBATCH --account=def-rscholes #def-agiang01 #
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name='run_IDFs'
python run_IDFs.py