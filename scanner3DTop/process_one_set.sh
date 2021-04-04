#!/bin/bash

HPC_PATH="/xdisk/cjfrost/egonzalez/PhytoOracle/scanner3DTop/"
SIMG_PATH="/xdisk/ericlyons/big_data/singularity_images/"
#PLANT_LOC=${HPC_PATH}"season10_plant_detections.csv"
PLANT_LOC=${HPC_PATH}"plant_detections/2020-01-20_detection.csv"
#PLANT_LOC=${HPC_PATH}"season10_plant_locations_for_3d.csv"
PLANT_LOC_CLIP=${HPC_PATH}"season10_plant_locations_for_3d.csv"

CLEANED_META_DIR="cleanmetadata_out/"
MERGE_DIR="icp_registration_out/"
GEO_REF_DIR="rotation_registration_out/"
#GEO_COR_DIR="geocorrect_out/"
GEO_COR_DIR="3d_geo_correction_out/"
SCALE_ROT_DIR="scale_rotate_out/"
PLANT_CLIP_DIR="plantclip_out/"

METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
EAST_PLY=${HPC_PATH}${RAW_DATA_PATH}${UUID}"__Top-heading-east_0.ply"
WEST_PLY=${HPC_PATH}${RAW_DATA_PATH}${UUID}"__Top-heading-west_0.ply"
MERGE_PLY=${MERGE_DIR}${UUID}"_icp_merge.ply"
MERGE_REF_PLY=${GEO_REF_DIR}${UUID}"_icp_merge_registered.ply"
#GEO_COR_PLY=${GEO_COR_DIR}${UUID}"_icp_merge_registered_geocorrected_full.ply"
GEO_COR_PLY=${GEO_COR_DIR}${UUID}"_corrected.ply"
MERGE_PNG=${MERGE_DIR}${UUID}"_merged_east_west.png"

HTTP_USER="YOUR_USERNAME"
HTTP_PASSWORD="PhytoOracle"
DATA_BASE_URL="128.196.142.42/"
set -e

# Stage the data from HTTP server
#mkdir -p ${RAW_DATA_PATH}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${METADATA} -O ${METADATA}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${EAST_PLY} -O ${EAST_PLY}
#wget --user ${HTTP_USER} --password ${HTTP_PASSWORD} ${DATA_BASE_URL}${WEST_PLY} -O ${WEST_PLY}

#################################
# Merge east and west pointcloud#
#################################
EAST_PLY=${EAST_PLY}
WEST_PLY=${WEST_PLY}
#METADATA=${METADATA_CLEANED}
MERGE_DIR=${MERGE_DIR}
WORKING_SPACE=${HPC_PATH}${RAW_DATA_PATH}
SIMG_PATH=${SIMG_PATH}


mkdir -p ${MERGE_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt docker://agpipeline/ply2las:2.1 --result print --working_space ${WORKING_SPACE} --metadata ${METADATA}
#singularity run ${SIMG_PATH}3d_icp_registration.simg -w ${WEST_PLY} -e ${EAST_PLY}
#NEW
singularity run ${SIMG_PATH}3d_icp_merge.simg -w ${WEST_PLY} -e ${EAST_PLY}

###############################
# Scale and rotate point cloud#
###############################
#MERGE_PLY=${MERGE_PLY}
#METADATA=${METADATA}
#SCALE_ROT_DIR=${SCALE_ROT_DIR}

#mkdir -p ${SCALE_ROT_DIR}
#singularity run ${SIMG_PATH}3d_scale_rotate.simg -m ${METADATA} ${MERGE_PLY}

###################################
# Geo-reference merged point cloud#
###################################
MERGE_PLY=${MERGE_PLY}
METADATA=${METADATA}
GEO_REF_DIR=${GEO_REF_DIR}

mkdir -p ${GEO_REF_DIR}
singularity run ${SIMG_PATH}3d_geo_ref.simg -m ${METADATA} ${MERGE_PLY}

###############################################
# Correct georeferencing of merged point cloud#
############################################### 
METADATA=${METADATA}
MERGE_REF_PLY=${MERGE_REF_PLY}
GEO_COR_DIR=${GEO_COR_DIR}
PLANT_LOC=${PLANT_LOC}
MERGE_PNG=${MERGE_PNG}

mkdir -p ${GEO_COR_DIR}
##singularity run ${SIMG_PATH}3d_geo_cor.simg -m ${METADATA} -l ${PLANT_LOC} -d Y ${MERGE_REF_PLY} 
#NEW
singularity run ${SIMG_PATH}3d_geo_correction.simg -m ${HPC_PATH}model_weights_2021-03-26_10e_season10_3dpng.pth -d ${HPC_PATH}season10_lettuce_rgb_complete.csv -s 2020-03-03 -p ${MERGE_REF_PLY} -j ${METADATA} -i ${MERGE_PNG} 

#############################
# Clip out individual plants#
#############################
PLANT_LOC_CLIP=${PLANT_LOC_CLIP}
GEO_COR_PLY=${GEO_COR_PLY}
PLANT_CLIP_DIR=${PLANT_CLIP_DIR}

mkdir -p ${PLANT_CLIP_DIR}
singularity run ${SIMG_PATH}3d_plant_clip.simg -c ${PLANT_LOC_CLIP} ${GEO_COR_PLY}

####################################
# create tarball of plotclip result#
####################################
tar -cvf ${UUID}_plantclip.tar ${PLANT_CLIP_DIR}
