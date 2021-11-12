#!/bin/bash -l
#SBATCH --account=dukepauli
#SBATCH --partition=standard
#SBATCH --job-name="po_batch"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=94
#SBATCH --time=168:00:00
WORK_DIR=${1}
cd ${WORK_DIR}

./pipeline.py -r --season 10 --sensor scanner3DTop
