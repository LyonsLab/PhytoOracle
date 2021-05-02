#!/bin/bash -l
#SBATCH --account=dukepauli
#SBATCH --partition=standard
#SBATCH --job-name="po_test_batch"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=94
#SBATCH --time=12:00:00
#module load python/3.8
WORK_DIR=${1}
cd ${WORK_DIR}

./run_pipeline.py --season 10 --sensor rgb
