#!/bin/bash

#DATA_PATH="/tmp/"
HPC_PATH="/home/emmanuelgonzalez/PhytoOracle/psII/"
SIMG_PATH="/scratch/singularity_images/"
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

# Convert bins to PNG/TIF
METADATA=${METADATA}
TIFS_DIR=${TIFS_DIR}
WORKING_SPACE=${TIFS_DIR}
PSII_BIN=${PSII_BIN}

ls ${METADATA}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}psii_bin_to_tif_up.simg -m ${METADATA} -o ${WORKING_SPACE} ${HPC_PATH}${RAW_DATA_PATH}*.bin

# Clip to plot 
SENSOR=${SENSOR}
TIFS_DIR=${TIFS_DIR}
PLOTCLIP_DIR=${PLOTCLIP_DIR}
WORKING_SPACE=${PLOTCLIP_DIR}

ls ${TIFS_DIR}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}rgb_flir_plot_clip_geojson.simg -sen ${SENSOR} -o ${PLOTCLIP_DIR} -shp ${GEOJ} ${TIFS_DIR}

# Image segmentation
SEG_DIR=${SEG_DIR}
PLOTCLIP_DIR=${PLOTCLIP_DIR}
WORKING_SPACE=${SEG_DIR}

ls ${PLOTCLIP_DIR}
mkdir -p ${SEG_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}segmentation_psii.simg -o ${SEG_DIR} ${PLOTCLIP_DIR}*

# create tarball of plotclip result
tar -cvf ${UUID}_segmentation.tar ${SEG_DIR}

