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
DIR_PATH=${RAW_DATA_PATH}
DATE=$(dirname "$DIR_PATH")

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="bin2tif_out/"
SOILMASK_DIR="soil_mask_out/"
FIELDMOSAIC_DIR="fieldmosaic_out/"
PLOTCLIP_DIR="plotclip_out/"
GPSCORRECT_DIR="gpscorrect_out/"
ORTHO_OUT="ortho_out/"
# Inputs 
METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
LEFT_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_left.bin"
RIGHT_BIN=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_right.bin"

# Outputs 
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
LEFT_TIF=${HPC_PATH}${TIFS_DIR}${UUID}"_left.tif"
RIGHT_TIF=${TIFS_DIR}${UUID}"_right.tif"
MOSAIC_LIST_FILE=${FIELDMOSAIC_DIR}"filelist.txt"
LEFT_CLIP=${PLOTCLIP_DIR}${UUID}"_left.tif"
RIGHT_CLIP=${PLOTCLIP_DIR}${UUID}"_right.tif"
LEFT_TIF_CORRECT=${GPSCORRECT_DIR}${UUID}"_left_corrected.tif"
RIGHT_TIF_CORRECT=${GPSCORRECT_DIR}${UUID}"_right_corrected.tif"
#GPS_CSV=${HPC_PATH}${ORTHO_OUT}/2020-01-20/2020-01-20_coordinates_CORRECTED.csv"
GPS_CSV=${HPC_PATH}${ORTHO_OUT}${DATE}"/"${DATE}"_coordinates_CORRECTED.csv"
GEOJ=${HPC_PATH}"season10_multi_latlon_geno.geojson"
GPS_UNCOR=${HPC_PATH}""

HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
DATA_BASE_URL="128.196.142.19/"
set -e

# Correct GPS
#RIGHT_TIF=${RIGHT_TIF}
GPSCORRECT_DIR=${GPSCORRECT_DIR}
WORKING_SPACE=${GPSCORRECT_DIR}
GPS_CSV=${GPS_CSV}
#TIFS_DIR=${HPC_PATH}${TIFS_DIR}

#ls ${TIFS_DIR}
ls ${LEFT_TIF}
#ls ${RIGHT_TIF}
ls ${GPS_CSV}

mkdir -p ${WORKING_SPACE}
##singularity run -B $(pwd):/mnt --pwd /mnt docker://acicarizona/gistools --csv ${GPS_CSV} -o ${WORKING_SPACE} ${TIFS_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/edit_gps:latest --csv ${GPS_CSV} ${LEFT_TIF}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}edit_gps.simg --csv ${GPS_CSV} ${LEFT_TIF}
ls ${LEFT_TIF_CORRECT}
#ls ${RIGHT_TIF_CORRECT}

# plotclip-shp > NEW
# left
PLOTCLIP_DIR=${PLOTCLIP_DIR}
WORKING_SPACE=${PLOTCLIP_DIR}
##EPSG="32612"
SENSOR="stereoTop"
GEOJ=${GEOJ}
##GPSCORRECT_DIR=${GPSCORRECT_DIR}
LEFT_TIF_CORRECT=${LEFT_TIF_CORRECT}

#mkdir -p ${WORKING_SPACE}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/plotclip_geo:latest --sensor ${SENSOR} --shape ${GEOJ} ${LEFT_TIF_CORRECT}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}plotclip_shp.simg --sensor ${SENSOR} --shape ${GEOJ} ${LEFT_TIF_CORRECT}
#ls ${LEFT_CLIP}

# create tarball of plotclip result
#
tar -cvf ${UUID}_plotclip.tar ${PLOTCLIP_DIR}

