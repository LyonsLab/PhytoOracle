#!/bin/bash
#SBATCH --account=lyons-lab
#SBATCH --partition=standard
#SBATCH --job-name="po_batch"
#SBATCH --nodes=1
#SBATCH --mem=470GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=94
#SBATCH --time=168:00:00
#SBATCH --mpi=pmi2
WORK_DIR=${1}
cd ${WORK_DIR}

./run_pipeline.py --season 11 --sensor stereoTop
