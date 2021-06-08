#!/bin/bash
#SBATCH --account=dukepauli
#SBATCH --partition=standard
#SBATCH --job-name="po_worker"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00

export CCTOOLS_HOME=${HOME}/cctools-7.1.12-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.12-x86_64-centos7/bin/work_queue_factory -T slurm -B "--account=dukepauli --partition=standard --job-name=po_worker --time=6:00:00" -M phytooracle -w 15 -W 25 --workers-per-cycle 0 --cores=32 -t 300
