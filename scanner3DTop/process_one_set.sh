#!/bin/bash

# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID, $DATA_BASE_URL
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",

BETYDB_URL="http://128.196.65.186:8000/bety/"
BETYDB_KEY="wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD"
HPC_PATH="/xdisk/ericlyons/big_data/cosi/scanner3DTop/"
SIMG_PATH='/xdisk/ericlyons/big_data/singularity_images/'

#3D_DIR="3D_out/"
#FIELDMOSAIC_DIR="fieldmosaic_out/"


METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
#3D_OUT_DIR=${HPC_PATH}${3D_DIR}
#MEANTEMP_WK=${HPC_PATH}${MEANTEMP_DIR}${UUID}"/"
#MOSAIC_LIST_FILE=${HPC_PATH}${FIELDMOSAIC_DIR}"filelist.txt"

singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_mergeply.simg -m ${METADATA} -o ${HPC_PATH}/3D_out ${HPC_PATH}${RAW_DATA_PATH}
