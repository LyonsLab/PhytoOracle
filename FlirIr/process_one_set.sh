#!/bin/bash

HPC_PATH="/scratch/PhytoOracle/FlirIr/"
SIMG_PATH='/scratch/singularity_images/'

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="flir2tif_out/"
MEANTEMP_DIR="meantemp_out/" 
PLOTCLIP_DIR="plotclip_out/"

METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
IR_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_ir.bin"
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
IN_TIF=${TIFS_DIR}${UUID}"_ir.tif"

# Make a cleaned copy of the metadata
SENSOR="flirIrCamera"
METADATA=${METADATA}
WORKING_SPACE=${CLEANED_META_DIR}
USERID=""

# Convert  bin/RGB image to TIFF format
IR_BIN=${IR_BIN}
WORKING_SPACE=${TIFS_DIR}

mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt  ${SIMG_PATH}po_flir2tif_s10_latest.simg -m ${METADATA} ${IR_BIN}
