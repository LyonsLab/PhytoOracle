#!/bin/bash -l
#SBATCH --account=frost_lab
#SBATCH --partition=standard
#SBATCH --job-name="po_worker"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

export CCTOOLS_HOME=${HOME}/cctools-7.1.12-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.12-x86_64-centos7/bin/work_queue_factory -T slurm -B "--account=frost_lab --partition=standard --job-name=po_worker" -M phytooracle_3d -w 10 -W 20 --workers-per-cycle 0 --cores=16 -t 300
