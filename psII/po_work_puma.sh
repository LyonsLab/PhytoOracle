#!/bin/bash 
#SBATCH --account=dukepauli --partition=standard
#SBATCH --job-name="phytooracle"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=94
#SBATCH --time=18:00:00
#module load singularity
module load python/3.8

export CCTOOLS_HOME=/home/u31/emmanuelgonzalez/cctools-7.1.6-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH
#cd /xdisk/ericlyons/big_data/egonzalez/PhytoOracle/psII
#cd /home/u31/emmanuelgonzalez/
#RGB
#singularity pull docker://agpipeline/cleanmetadata:2.2
#singularity pull docker://agpipeline/bin2tif:2.0
#singularity pull docker://zhxu73/gistools:latest
#singularity pull docker://agpipeline/plotclip:3.1

#3D
#singularity pull docker://agpipeline/cleanmetadata:2.0
#singularity pull docker://agpipeline/ply2las:2.1
#singularity pull docker://agpipeline/plotclip:3.0

#PSII
#singularity pull docker://agpipeline/cleanmetadata:2.0

#cd /xdisk/ericlyons/big_data/egonzalez/PhytoOracle/psII/
#cd /xdisk/ericlyons/big_data/egonzalez/PhytoOracle/scanner3DTop/
/home/u31/emmanuelgonzalez/cctools-7.1.6-x86_64-centos7/bin/work_queue_factory -T local -M phyto_oracle-atmo -w 40 -W 90 --workers-per-cycle 10 --cores=1 -t 900
