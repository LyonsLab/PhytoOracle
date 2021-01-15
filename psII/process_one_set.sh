#!/bin/bash

# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID, $DATA_BASE_URL
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",
#XDG_RUNTIME_DIR="/home/u31/emmanuelgonzalez"
BETYDB_URL="http://128.196.65.186:8000/bety/"
BETYDB_KEY="wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD"
DATA_PATH="/tmp/"
HPC_PATH="/xdisk/ericlyons/big_data/egonzalez/PhytoOracle/psII/"
SIMG_PATH="/xdisk/ericlyons/big_data/singularity_images/"
XDG_RUNTIME_DIR=${HPC_PATH}
GEOJ=${HPC_PATH}"season11_multi_latlon_geno.geojson"
SENSOR="psiiTop"
CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="bin2tif_out/"
TIF_RESIZE="tifresize_out/"
SOILMASK_DIR="soil_mask_out/"
FIELDMOSAIC_DIR="fieldmosaic_out/"
PLOTCLIP_DIR="plotclip_out/"
GPSCORRECT_DIR="gpscorrect_out/"
SEG_DIR="psii_segmentation_out/"

METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
LEFT_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_left.bin"
RIGHT_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_right.bin"
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
GPS_CSV=${HPC_PATH}"2020-01-08_coordinates_CORRECTED_5-1-2020.csv"
PSII_BIN=${RAW_DATA_PATH}${UUID}"_rawData0046.bin"
PSII_TIF=${TIFS_DIR}${UUID}"_rawData0046.tif"

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

# Make a cleaned copy of the metadata
#SENSOR="ps2Top"
#METADATA=${METADATA}
#WORKING_SPACE=${CLEANED_META_DIR}
#USERID=""

##ls ${RAW_DATA_PATH}
##ls ${METADATA}
#ls "cached_betydb/bety_experiments.json"
#mkdir -p ${WORKING_SPACE}
##BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:2.2 --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}

#BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}cleanmetadata_2.2.simg --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}
##ls ${CLEANED_META_DIR}
##ls ${METADATA_CLEANED}

# Convert bins to PNG/TIF
METADATA=${METADATA}
TIFS_DIR=${TIFS_DIR}
WORKING_SPACE=${TIFS_DIR}
PSII_BIN=${PSII_BIN}

ls ${METADATA}
mkdir -p ${WORKING_SPACE}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/bin2tif_psii:latest -m ${METADATA} -o ${WORKING_SPACE} ${HPC_PATH}${RAW_DATA_PATH}*.bin
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}psii_bin_to_tif_up.simg -m ${METADATA} -o ${WORKING_SPACE} ${HPC_PATH}${RAW_DATA_PATH}*.bin

# Resize TIFs
#TIFS_DIR=${TIFS_DIR}
#TIF_RESIZE=${TIF_RESIZE}
#WORKING_SPACE=${TIF_RESIZE}

#ls ${TIFS_DIR}
#mkdir -p ${WORKING_SPACE}
##singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/resize_tif:latest ${TIFS_DIR}*.tif
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}resizetif_psii.simg ${TIFS_DIR}*.tif

# Clip to plot 
SENSOR=${SENSOR}
TIFS_DIR=${TIFS_DIR}
PLOTCLIP_DIR=${PLOTCLIP_DIR}
WORKING_SPACE=${PLOTCLIP_DIR}

ls ${TIFS_DIR}
mkdir -p ${WORKING_SPACE}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/plotclip_shp:latest -sen ${SENSOR} -o ${PLOTCLIP_DIR} -shp ${GEOJ} ${TIF_RESIZE}*.tif
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}rgb_flir_plot_clip_geojson.simg -sen ${SENSOR} -o ${PLOTCLIP_DIR} -shp ${GEOJ} ${TIFS_DIR}

# Image segmentation
SEG_DIR=${SEG_DIR}
PLOTCLIP_DIR=${PLOTCLIP_DIR}
WORKING_SPACE=${SEG_DIR}

ls ${PLOTCLIP_DIR}
mkdir -p ${SEG_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/psii_segmentation:latest -o ${SEG_DIR} ${PLOTCLIP_DIR}*
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}segmentation_psii.simg -o ${SEG_DIR} ${PLOTCLIP_DIR}*

# create tarball of plotclip result
tar -cvf ${UUID}_segmentation.tar ${SEG_DIR}
#tar -cvf ${UUID}_plotclip.tar ${PLOTCLIP_DIR}

