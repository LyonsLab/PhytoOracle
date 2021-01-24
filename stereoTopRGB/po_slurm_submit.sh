#!/bin/bash -l
#SBATCH --account=dukepauli
#SBATCH --partition=standard
#SBATCH --job-name="po_test_batch"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=94
#SBATCH --time=12:00:00
module load python/3.8
SCAN_DATE=${1%/}
WORK_DIR=${2}

#cd /xdisk/ericlyons/big_data/egonzalez/phyto_training/PhytoOracle/stereoTopRGB
cd ${WORK_DIR}
./run.sh ${SCAN_DATE}
