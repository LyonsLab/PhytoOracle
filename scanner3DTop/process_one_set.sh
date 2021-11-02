#!/bin/bash

HPC_PATH="/xdisk/dukepauli/emmanuelgonzalez/testing/season_10/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}

################
#Pre-processing#
################
singularity run ${SIMG_PATH}3d_preprocessing.simg -l ${SIMG_PATH}gcp_season_10.txt -o preprocessing_out -i ${HPC_PATH}${RAW_DATA_PATH}
