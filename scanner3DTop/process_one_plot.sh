#!/bin/bash

#Pass down by the main_workflow
#PLOT_NAME="MAC Field Scanner Season 4 Range 21 Column 10"
#PLOT_DIR="plotclip_out/MAC Field Scanner Season 4 Range 21 Column 10/"
#METADATA="cleanmetadata_out/233895ec-7b93-4523-829c-c59d2da9778a_metadata_cleaned.json"
#CANOPY_HEIGHT_DIR="canopy_height_out/"

# plotmerge
#
MERGE_FILENAME=${PLOT_NAME}
LAS_FILE_LIST=$(ls -d "${PLOT_DIR}"*.las | xargs -I {} echo "\"/mnt/{}\"")

BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/plotmerge:3.0 --result print --working_space /mnt/plotmerge_out --metadata /mnt/${METADATA} --merge_filename "${MERGE_FILENAME}" scanner3DTop ${LAS_FILE_LIST}

# insert plot name
#
METADATA_OUT="plotmerge_out/""${PLOT_NAME}""_metadata_cleaned.json"
echo ${METADATA_OUT}
./insert_plot_name.py ${METADATA} "${METADATA_OUT}" "${PLOT_NAME}"

# canopy height
#
WORKING_SPACE=${CANOPY_HEIGHT_DIR}
METADATA=${METADATA_OUT}
MERGED_LAS_FILE="plotmerge_out/"${PLOT_NAME}_merged.las

mkdir -p ${WORKING_SPACE}
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/canopy_height:2.1 --working_space ${WORKING_SPACE} --result print --metadata "/mnt/${METADATA}" "/mnt/${MERGED_LAS_FILE}"

