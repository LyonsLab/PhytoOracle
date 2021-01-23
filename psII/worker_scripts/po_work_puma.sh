#!/bin/bash -l
#SBATCH --account=dukepauli
#SBATCH --partition=standard
#SBATCH --job-name="po_worker"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=22
#SBATCH --time=12:00:00
#module load singularity
module load python/3.8

export CCTOOLS_HOME=${HOME}/cctools-7.1.6-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.6-x86_64-centos7/bin/work_queue_factory -T local -M phytooracle -w 20 -W 20 --workers-per-cycle 10 --cores=1 -t 900
