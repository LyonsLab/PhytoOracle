#!/bin/bash

HPC_PATH="/xdisk/ericlyons/data/emmanuelgonzalez/testing/season_12/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}

################
#Pre-processing#
################
singularity run ${SIMG_PATH}3d_preprocessing.simg -l ${SIMG_PATH}season12_all_bucket_gcps.txt -o preprocessing_out -i ${HPC_PATH}${RAW_DATA_PATH}
