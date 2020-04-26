#!/bin/bash



# Variable needed to be pass in by external
# $RAW_DATA_PATH, $UUID, $DATA_BASE_URL
#
# e.g.
#"RAW_DATA_PATH": "2018-05-15/2018-05-15__12-04-43-833",
#"UUID": "5716a146-8d3d-4d80-99b9-6cbf95cfedfb",

CLEANED_META_DIR="cleanmetadata_out/"
TIFS_DIR="flir2tif_out/"
MEANTEMP_DIR="meantemp_out/" 
#PLOTCLIP_DIR="plotclip_out/"
#FIELDMOSAIC_DIR="fieldmosaic_out/"

MEANTEMP_WK=${MEANTEMP_DIR}${UUID}"/"

METADATA=${RAW_DATA_PATH}${UUID}"_metadata.json"
IR_BIN=${RAW_DATA_PATH}${UUID}"_ir.bin"
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json"
IN_TIF=${TIFS_DIR}${UUID}"_ir.tif"
#MOSAIC_LIST_FILE=${FIELDMOSAIC_DIR}"filelist.txt"

HTTP_USER="mcosi"
HTTP_PASSWORD="CoGe"
set -e

# Stage the data from HTTP server
mkdir -p ${RAW_DATA_PATH}
wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${METADATA} -O ${METADATA}
wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${IR_BIN} -O ${IR_BIN}


# Make a cleaned copy of the metadata
SENSOR="flirIrCamera"
METADATA=${METADATA}
WORKING_SPACE=${CLEANED_META_DIR}
USERID=""

ls ${RAW_DATA_PATH}
ls ${METADATA}
ls "cached_betydb/bety_experiments.json"
mkdir -p ${WORKING_SPACE}
BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:2.0 --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}
ls ${CLEANED_META_DIR}
ls ${METADATA_CLEANED}

# Convert  bin/RGB image to TIFF format
IR_BIN=${IR_BIN}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${TIFS_DIR}

ls ${IR_BIN}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/flir2tif:2.2 --result print --working_space ${WORKING_SPACE} --metadata ${METADATA} ${IR_BIN}
ls ${IN_TIF}

# Extract meantemp data
IN_TIF=${IN_TIF} 
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${MEANTEMP_WK}
TITLE="title"
YEAR="year"
AUTHOR="author"

ls ${IN_TIF}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}

BETYDB_URL=http://128.196.65.186:8000/bety/ BETYDB_KEY=YUKPF38ZxMB0UOkP6etB9bNOjTjIeWFj0RbNGIg5 singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/meantemp:3.0 --result print --working_space ${WORKING_SPACE} --metadata ${METADATA} --citation_author $AUTHOR --citation_title $TITLE --citation_year ${YEAR} ${IN_TIF}

#docker run --rm --mount "src=`pwd`,target=/mnt,type=bind" -e "BETYDB_URL=http://128.196.65.186:8000/bety/" -e "BETYDB_KEY=YUKPF38ZxMB0UOkP6etB9bNOjTjIeWFj0RbNGIg5" agpipeline/meantemp:3.0 --working_space "/mnt" --metadata "/mnt/cleanmetadata_out/46bbef81-0858-4195-b921-05cd8700af47_metadata_cleaned.json" --citation_author "Me Myself" --citation_title "Something that's warm" --citation_year "2020" "/mnt/flir2tif_out/46bbef81-0858-4195-b921-05cd8700af47_ir.tif"
