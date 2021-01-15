#!/bin/bash
#PBS -q windfall
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -W group_list=lyons-lab
#PBS -l place=pack:shared
#PBS -l walltime=48:00:00
#PBS -l cput=152:00:00
module load singularity
module load unsupported
module load ferng/glibc
export CCTOOLS_HOME=/rsgrps/ericlyons/phytooracle/cctools-release-7.1.2
export PATH=${CCTOOLS_HOME}/bin:$PATH
cd /rsgrps/ericlyons/phytooracle/PSII/1set/ps2_geotif
singularity pull docker://acicarizona/ps2top-bin2png:1.0
singularity pull docker://acicarizona/ps2top-img_segmentation:1.0
singularity pull docker://acicarizona/ps2top-fluorescence_aggregation:1.0
/rsgrps/ericlyons/phytooracle/cctools-release-7.1.2/bin/resource_monitor -O log-16c64g_1set -i 2 -- makeflow --jx ps2_Tif_workflow.jx --jx-args args.json
