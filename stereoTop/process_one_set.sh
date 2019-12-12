#!/bin/bash



# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="bin2tif_out/"
SOILMASK_DIR="soil_mask_out/"
FIELDMOSAIC_DIR="fieldmosaic_out/"

METADATA=${RAW_DATA_PATH}${UUID}"_metadata.json"
LEFT_BIN=${RAW_DATA_PATH}${UUID}"_left.bin"
RIGHT_BIN=${RAW_DATA_PATH}${UUID}"_right.bin"
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
LEFT_TIF=${TIFS_DIR}${UUID}"_left.tif"
RIGHT_TIF=${TIFS_DIR}${UUID}"_right.tif"
LEFT_SOILMASK=${SOILMASK_DIR}${UUID}"_left_mask.tif"
RIGHT_SOILMASK=${SOILMASK_DIR}${UUID}"_right_mask.tif"
MOSAIC_LIST_FILE=${FIELDMOSAIC_DIR}"filelist.txt"

set -e

# Make a cleaned copy of the metadata
SENSOR="stereoTop"
METADATA=${METADATA}
WORKING_SPACE=${CLEANED_META_DIR}
USERID=""

ls ${RAW_DATA_PATH}
ls ${METADATA}
ls "cached_betydb/bety_experiments.json"
mkdir -p ${WORKING_SPACE}
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:latest --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}
ls ${CLEANED_META_DIR}
ls ${METADATA_CLEANED}

# Convert LEFT bin/RGB image to TIFF format
LEFT_BIN=${LEFT_BIN}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${TIFS_DIR}

ls ${TIFS_DIR}
ls ${LEFT_BIN}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_BIN}
ls ${LEFT_TIF}

# Convert RIGHT bin/RGB image to TIFF format
RIGHT_BIN=${RIGHT_BIN}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${TIFS_DIR}

ls ${TIFS_DIR}
ls ${RIGHT_BIN}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${RIGHT_BIN}
ls ${RIGHT_TIF}

# Generate soil mask from LEFT TIFF image
LEFT_TIF=${LEFT_TIF}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${SOILMASK_DIR}

ls ${SOILMASK_DIR}
ls ${LEFT_TIF}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/soilmask:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_TIF}
ls ${LEFT_SOILMASK}

# Generate soil mask from RIGHT TIFF image
RIGHT_TIF=${RIGHT_TIF}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${SOILMASK_DIR}

ls ${SOILMASK_DIR}
ls ${RIGHT_TIF}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/soilmask:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${RIGHT_TIF}
ls ${RIGHT_SOILMASK}
