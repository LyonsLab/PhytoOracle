#!/bin/bash



# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID, $DATA_BASE_URL
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",

BETYDB_URL="http://128.196.65.186:8000/bety/"
BETYDB_KEY="wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD"

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="bin2tif_out/"
SOILMASK_DIR="soil_mask_out/"
FIELDMOSAIC_DIR="fieldmosaic_out/"
PLOTCLIP_DIR="plotclip_out/"

METADATA=${RAW_DATA_PATH}${UUID}"_metadata.json"
LEFT_BIN=${RAW_DATA_PATH}${UUID}"_left.bin"
RIGHT_BIN=${RAW_DATA_PATH}${UUID}"_right.bin"
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
LEFT_TIF=${TIFS_DIR}${UUID}"_left.tif"
RIGHT_TIF=${TIFS_DIR}${UUID}"_right.tif"
#LEFT_SOILMASK=${SOILMASK_DIR}${UUID}"_left_mask.tif"
#RIGHT_SOILMASK=${SOILMASK_DIR}${UUID}"_right_mask.tif"
MOSAIC_LIST_FILE=${FIELDMOSAIC_DIR}"filelist.txt"
LEFT_CLIP=${PLOTCLIP_DIR}${UUID}"_left.tif"
RIGHT_CLIP=${PLOTCLIP_DIR}${UUID}"_right.tif"

HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
DATA_BASE_URL="128.196.142.38/"
set -e

# Stage the data from HTTP server
mkdir -p ${RAW_DATA_PATH}
wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${METADATA} -O ${METADATA}
wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${LEFT_BIN} -O ${LEFT_BIN}
wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${RIGHT_BIN} -O ${RIGHT_BIN}


# Make a cleaned copy of the metadata
SENSOR="stereoTop"
METADATA=${METADATA}
WORKING_SPACE=${CLEANED_META_DIR}
USERID=""

ls ${RAW_DATA_PATH}
ls ${METADATA}
ls "cached_betydb/bety_experiments.json"
mkdir -p ${WORKING_SPACE}
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:2.2 --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}
ls ${CLEANED_META_DIR}
ls ${METADATA_CLEANED}

# Convert LEFT bin/RGB image to TIFF format
LEFT_BIN=${LEFT_BIN}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${TIFS_DIR}

ls ${LEFT_BIN}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:2.0 --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_BIN}
ls ${LEFT_TIF}

# Convert RIGHT bin/RGB image to TIFF format
RIGHT_BIN=${RIGHT_BIN}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${TIFS_DIR}

ls ${RIGHT_BIN}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/bin2tif:2.0 --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${RIGHT_BIN}
ls ${RIGHT_TIF}

# plotclip
# left
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/
METADATA=${METADATA_CLEANED}
PLOTCLIP_DIR="plotclip_out/"
WORKING_SPACE=${PLOTCLIP_DIR}
EPSG="32612"
SENSOR="stereoTop"
#LEFT_TIF=${TIFS_DIR}${UUID}"_left.tif"
LEFT_TIF=${LEFT_TIF}

mkdir -p ${WORKING_SPACE}
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://chrisatua/development:plotclip_test --working_space "/mnt/${WORKING_SPACE}" --metadata "/mnt/${METADATA}" --epsg ${EPSG} ${SENSOR} "/mnt/${LEFT_TIF}"
#ls ${LEFT_CLIP}
mv plotclip_out/result.json plotclip_out/${UUID}.json

# plotclip
# right
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/
METADATA=${METADATA_CLEANED}
PLOTCLIP_DIR="plotclip_out/"
WORKING_SPACE=${PLOTCLIP_DIR}
EPSG="32612"
SENSOR="stereoTop"
#RIGHT_TIF=${TIFS_DIR}${UUID}"_right.tif"
RIGHT_TIF=${RIGHT_TIF}

mkdir -p ${WORKING_SPACE}
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://chrisatua/development:plotclip_test --working_space "/mnt/${WORKING_SPACE}" --metadata "/mnt/${METADATA}" --epsg ${EPSG} ${SENSOR} "/mnt/${RIGHT_TIF}"
#ls ${RIGHT_CLIP}
mv plotclip_out/result.json plotclip_out/${UUID}.json

# create tarball of plotclip result
#
tar -cvf ${UUID}_plotclip.tar ${PLOTCLIP_DIR}

# Generate soil mask from LEFT TIFF image
#LEFT_TIF=${LEFT_TIF}
#METADATA=${METADATA_CLEANED}
#WORKING_SPACE=${SOILMASK_DIR}

#ls ${LEFT_TIF}
#ls ${METADATA_CLEANED}
#mkdir -p ${WORKING_SPACE}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/soilmask:2.0 --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${LEFT_TIF}
#ls ${LEFT_SOILMASK}

# Generate soil mask from RIGHT TIFF image
#RIGHT_TIF=${RIGHT_TIF}
#METADATA=${METADATA_CLEANED}
#WORKING_SPACE=${SOILMASK_DIR}

#ls ${RIGHT_TIF}
#ls ${METADATA_CLEANED}
#mkdir -p ${WORKING_SPACE}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/soilmask:2.0 --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${RIGHT_TIF}
#ls ${RIGHT_SOILMASK}
