#!/bin/bash
#SBATCH --account=lyons-lab
#SBATCH --partition=standard
#SBATCH --job-name="po_batch"
#SBATCH --mem=470GB
#SBATCH --ntasks=94
#SBATCH --ntasks-per-node=94
#SBATCH --nodes=1
#SBATCH --time=168:00:00
WORK_DIR=${1}
cd ${WORK_DIR}

./run_pipeline.py --season 11 --sensor stereoTop
