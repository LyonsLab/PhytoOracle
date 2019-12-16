#!/bin/bash

# Pass down by the main_workflow
#LEVEL_0_PATH="small_test_set/PNG/2017-06-21__00-00-26-364/",
#LEVEL_1_PATH="small_test_set/PLY/2017-06-21__00-00-26-364/",
#UUID="b5246694-65d8-44b9-a99c-3d010c92ec64",

BETYDB_URL="https://terraref.ncsa.illinois.edu/bety/",
BETYDB_KEY="9999999999999999999999999999999999999999",

CLEANED_META_DIR="cleanmetadata_out/",
LAS_DIR="las_out/",
PLOTCLIP_DIR="plotclip_out/",
CANOPY_HEIGHT_DIR="canopy_height_out/",

METADATA=${LEVEL_0_PATH}${UUID}"_metadata.json",
METADATA_CLEANED=${CLEANED_META_DIR}${UUID}"_metadata_cleaned.json",
EAST_PLY=${LEVEL_1_PATH}${UUID}"__Top-heading-east_0.ply",
WEST_PLY=${LEVEL_1_PATH}${UUID}"__Top-heading-west_0.ply",
EAST_LAS=${LAS_DIR}${UUID}"__Top-heading-east_0.las",
WEST_LAS=${LAS_DIR}${UUID}"__Top-heading-west_0.las"

# cleanmetadata
WORKING_SPACE=${CLEANED_META_DIR}
SENSOR="scanner3DTop",
USERID=""

ls ${METADATA}
mkdir -p ${WORKING_SPACE}
BETYDB_URL=${BETYDB_URL} BETYDB_KEY=${BETYDB_KEY} singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/cleanmetadata:latest --result print --metadata ${METADATA} --working_space ${WORKING_SPACE} ${SENSOR} ${USERID}
ls ${METADATA_CLEANED}

# ply2las
# east
# working space is input directory, created for temp use
PLY_FILE=${EAST_PLY}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${LAS_DIR}${UUID}"_WEST/"
LAS_DIR=${LAS_DIR}

ls ${EAST_PLY}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
cp ${PLY_FILE} ${WORKING_SPACE}$(basename ${PLY_FILE})
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/ply2las --result print --metadata ${METADATA} --working_space ${WORKING_SPACE}
cp ${WORKING_SPACE}*.las ${LAS_DIR}
ls ${EAST_LAS}

# ply2las
# west
# working space is input directory, created for temp use
PLY_FILE=${WEST_PLY}
METADATA=${METADATA_CLEANED}
WORKING_SPACE=${LAS_DIR}${UUID}"_WEST/"
LAS_DIR=${LAS_DIR}

ls ${WEST_PLY}
ls ${METADATA_CLEANED}
mkdir -p ${WORKING_SPACE}
cp ${PLY_FILE} ${WORKING_SPACE}$(basename ${PLY_FILE})
singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/ply2las --result print --metadata ${METADATA} --working_space ${WORKING_SPACE}
cp ${WORKING_SPACE}*.las ${LAS_DIR}
ls ${WEST_LAS}



