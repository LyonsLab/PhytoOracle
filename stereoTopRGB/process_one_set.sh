#!/bin/bash

# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID, $DATA_BASE_URL
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",

HPC_PATH="/xdisk/cjfrost/egonzalez/po_season11/PhytoOracle/stereoTopRGB/"
SIMG_PATH="/xdisk/ericlyons/big_data/singularity_images/"

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
LEFT_TIF=${TIFS_DIR}${UUID}"_left.tif"
RIGHT_TIF=${TIFS_DIR}${UUID}"_right.tif"
LEFT_CLIP=${PLOTCLIP_DIR}${UUID}"_left.tif"
RIGHT_CLIP=${PLOTCLIP_DIR}${UUID}"_right.tif"
LEFT_TIF_CORRECT=${GPSCORRECT_DIR}${UUID}"_left_corrected.tif"
RIGHT_TIF_CORRECT=${GPSCORRECT_DIR}${UUID}"_right_corrected.tif"
#GPS_CSV=${HPC_PATH}"2020-02-18_coordinates_CORRECTED.csv"
#GEOJ=${HPC_PATH}"season10_lettuce_multi.geojson"
#GPS_UNCOR=${HPC_PATH}"/xdisk/cjfrost/egonzalez/po_season11/PhytoOracle/stereoTopRGB/"

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
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}rgb_bin2tif.simg -m ${METADATA} ${LEFT_BIN}
ls ${LEFT_TIF}
