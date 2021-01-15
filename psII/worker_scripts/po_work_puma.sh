#!/bin/bash -l
#SBATCH --account=<account>
#SBATCH --partition=standard
#SBATCH --job-name="<name>"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=94
#SBATCH --time=12:00:00
#module load singularity
module load python/3.8

export CCTOOLS_HOME=${HOME}/cctools-7.1.6-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.6-x86_64-centos7/bin/work_queue_factory -T local -M phytooracle -w 60 -W 90 --workers-per-cycle 10 --cores=1 -t 900
