#!/bin/bash

# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID, $DATA_BASE_URL
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",

BETYDB_URL="http://128.196.65.186:8000/bety/"
BETYDB_KEY="wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD"
HPC_PATH="/xdisk/ericlyons/big_data/egonzalez/PhytoOracle/stereoTopRGB/"
SIMG_PATH="/xdisk/ericlyons/big_data/singularity_images/"
#HPC_PATH="/tmp/"

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="bin2tif_out/"
SOILMASK_DIR="soil_mask_out/"
FIELDMOSAIC_DIR="fieldmosaic_out/"
PLOTCLIP_DIR="plotclip_out/"
GPSCORRECT_DIR="gpscorrect_out/"

# Inputs 
METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
LEFT_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_left.bin"
RIGHT_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_right.bin"

# Outputs 
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
LEFT_TIF=${TIFS_DIR}${UUID}"_left.tif"
RIGHT_TIF=${TIFS_DIR}${UUID}"_right.tif"
#LEFT_SOILMASK=${SOILMASK_DIR}${UUID}"_left_mask.tif"
#RIGHT_SOILMASK=${SOILMASK_DIR}${UUID}"_right_mask.tif"
MOSAIC_LIST_FILE=${FIELDMOSAIC_DIR}"filelist.txt"
LEFT_CLIP=${PLOTCLIP_DIR}${UUID}"_left.tif"
RIGHT_CLIP=${PLOTCLIP_DIR}${UUID}"_right.tif"
LEFT_TIF_CORRECT=${GPSCORRECT_DIR}${UUID}"_left_corrected.tif"
RIGHT_TIF_CORRECT=${GPSCORRECT_DIR}${UUID}"_right_corrected.tif"
GPS_CSV=${HPC_PATH}"2020-02-18_coordinates_CORRECTED.csv"
GEOJ=${HPC_PATH}"season10_lettuce_multi.geojson"
GPS_UNCOR=${HPC_PATH}""

HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
DATA_BASE_URL="128.196.142.19/"
set -e

# Stage the data from HTTP server
#mkdir -p ${RAW_DATA_PATH}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${METADATA} -O ${METADATA}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${LEFT_BIN} -O ${LEFT_BIN}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${RIGHT_BIN} -O ${RIGHT_BIN}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${GPS_CSV} -O ${GPS_CSV}

# Convert LEFT bin/RGB image to TIFF format
LEFT_BIN=${LEFT_BIN}
METADATA=${METADATA}
WORKING_SPACE=${TIFS_DIR}

ls ${LEFT_BIN}
mkdir -p ${WORKING_SPACE}
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}bin2tif_2_2.simg --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_BIN}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}rgb_bin2tif.simg -m ${METADATA} ${LEFT_BIN}
ls ${LEFT_TIF}
