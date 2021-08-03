#!/bin/bash -l
#SBATCH --account=lyons-lab
#SBATCH --partition=standard
#SBATCH --job-name="po_worker_emm"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#module load singularity
#module load python/3.8

export CCTOOLS_HOME=${HOME}/cctools-7.1.12-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.12-x86_64-centos7/bin/work_queue_factory -T slurm -M phytooracle_s11 -B "--account=lyons-lab --partition=standard --job-name=po_worker_emm"  -w 500 -W 1500 --workers-per-cycle 0 --cores=1 -t 300
