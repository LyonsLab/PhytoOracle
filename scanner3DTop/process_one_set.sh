#!/bin/bash

BETYDB_URL="http://128.196.65.186:8000/bety/"
BETYDB_KEY="wTtaueVXsUIjtqaxIaaaxsEMM4xHyPYeDZrg8dCD"
HPC_PATH="/xdisk/ericlyons/big_data/egonzalez/PhytoOracle/scanner3DTop/"
SIMG_PATH="/xdisk/ericlyons/big_data/singularity_images/"
PLANT_LOC=${HPC_PATH}"season10_plant_detections.csv"

CLEANED_META_DIR="cleanmetadata_out/"
MERGE_DIR="icp_registration_out/"
GEO_REF_DIR="rotation_registration_out/"
GEO_COR_DIR="geocorrect_out/"
SCALE_ROT_DIR="scale_rotate_out/"

METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
EAST_PLY=${HPC_PATH}${RAW_DATA_PATH}${UUID}"__Top-heading-east_0.ply"
WEST_PLY=${HPC_PATH}${RAW_DATA_PATH}${UUID}"__Top-heading-west_0.ply"
MERGE_PLY=${MERGE_DIR}${UUID}"_icp_merge.ply"
MERGE_REF_PLY=${GEO_REF_DIR}${UUID}"_icp_merge_registered.ply"

HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
DATA_BASE_URL="128.196.142.42/"
set -e

# Stage the data from HTTP server
#mkdir -p ${RAW_DATA_PATH}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${METADATA} -O ${METADATA}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${EAST_PLY} -O ${EAST_PLY}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${WEST_PLY} -O ${WEST_PLY}


# Merge east and west pointcloud
EAST_PLY=${EAST_PLY}
WEST_PLY=${WEST_PLY}
#METADATA=${METADATA_CLEANED}
MERGE_DIR=${MERGE_DIR}
WORKING_SPACE=${HPC_PATH}${RAW_DATA_PATH}
SIMG_PATH=${SIMG_PATH}


mkdir -p ${MERGE_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/ply2las:2.1 --result print --working_space ${WORKING_SPACE} --metadata ${METADATA}
singularity run ${SIMG_PATH}3d_icp_registration.simg -w ${WEST_PLY} -e ${EAST_PLY}

# Scale and rotate point cloud 
MERGE_PLY=${MERGE_PLY}
METADATA=${METADATA}
SCALE_ROT_DIR=${SCALE_ROT_DIR}

mkdir -p ${SCALE_ROT_DIR}
singularity run ${SIMG_PATH}3d_scale_rotate.simg -m ${METADATA} ${MERGE_PLY}

# Geo-reference merged point cloud 
MERGE_PLY=${MERGE_PLY}
METADATA=${METADATA}
GEO_REF_DIR=${GEO_REF_DIR}

mkdir -p ${GEO_REF_DIR}
singularity run ${SIMG_PATH}3d_geo_ref.simg -m ${METADATA} ${MERGE_PLY}

# Correct georeferencing of merged point cloud 
METADATA=${METADATA}
MERGE_REF_PLY=${MERGE_REF_PLY}
GEO_COR_DIR=${GEO_COR_DIR}
PLANT_LOC=${PLANT_LOC}

mkdir -p ${GEO_COR_DIR}

singularity run ${SIMG_PATH}3d_geo_cor.simg -m ${METADATA} -l ${PLANT_LOC} ${MERGE_REF_PLY} 

# plotclip
# west
#BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/
#METADATA=${METADATA_CLEANED}
#PLOTCLIP_DIR="plotclip_out/"
#WORKING_SPACE=${PLOTCLIP_DIR}
#EPSG="32612"
#SENSOR="scanner3DTop"
#LAS_FILE=${WEST_LAS}

#mkdir -p ${WORKING_SPACE}
#BETYDB_LOCAL_CACHE_FOLDER=cached_betydb/ singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/plotclip:3.0 --working_space "/mnt/${WORKING_SPACE}" --metadata "/mnt/${METADATA}" --epsg ${EPSG} ${SENSOR} "/mnt/${LAS_FILE}"
#mv plotclip_out/result.json plotclip_out/${UUID}.json

# create tarball of plotclip result
#
#tar -cvf ${UUID}_plotclip.tar ${PLOTCLIP_DIR}

