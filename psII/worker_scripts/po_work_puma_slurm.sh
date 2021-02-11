#!/bin/bash -l
#SBATCH --account=<account>
#SBATCH --partition=standard
#SBATCH --job-name="po_worker"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#module load singularity
module load python/3.8

export CCTOOLS_HOME=${HOME}/cctools-7.1.12-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.12-x86_64-centos7/bin/work_queue_factory -T slurm -B "--account=<account> --partition=standard" -M phytooracle -w <min_workers> -W <max_workers> --workers-per-cycle 0 --cores=1 -t 300
