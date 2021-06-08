#!/bin/bash


BETYDB_URL="http://128.196.65.186:8000/bety/"
BETYDB_KEY="wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD"
HPC_PATH="/xdisk/ericlyons/big_data/cosi/PhytoOracle_flirIr_S11/"
SIMG_PATH='/xdisk/ericlyons/big_data/singularity_images/'

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="flir2tif_out/"
MEANTEMP_DIR="meantemp_out/" 
PLOTCLIP_DIR="plotclip_out/"
#FIELDMOSAIC_DIR="fieldmosaic_out/"


METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
IR_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_ir.bin"
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
IN_TIF=${TIFS_DIR}${UUID}"_ir.tif"
#MEANTEMP_WK=${HPC_PATH}${MEANTEMP_DIR}${UUID}"/"
#MOSAIC_LIST_FILE=${HPC_PATH}${FIELDMOSAIC_DIR}"filelist.txt"

HTTP_USER="mcosi"
HTTP_PASSWORD="CoGe"
set -e


# Make a cleaned copy of the metadata
SENSOR="flirIrCamera"
METADATA=${METADATA}
WORKING_SPACE=${CLEANED_META_DIR}
USERID=""

ls "cached_betydb/bety_experiments.json"

# Convert  bin/RGB image to TIFF format
IR_BIN=${IR_BIN}
WORKING_SPACE=${TIFS_DIR}

mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt  ${HPC_PATH}flir2tif_S11.simg -m ${METADATA} ${IR_BIN}


