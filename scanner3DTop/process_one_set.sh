#!/bin/bash

HPC_PATH="/xdisk/ericlyons/data/emmanuelgonzalez/season_10/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}
PLANT_LOC_CLIP=${HPC_PATH}"season10_plant_detections_cleaned.csv"
DATE="`echo ${RAW_DATA_PATH} | grep -Eo '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}' | tail -1`"

CLEANED_META_DIR="cleanmetadata_out/"
MERGE_DIR="icp_registration_out/"
GEO_REF_DIR="rotation_registration_out/"
GEO_COR_DIR="3d_geo_correction_out/"
SCALE_ROT_DIR="scale_rotate_out/"
PLANT_CLIP_DIR="plantclip_out/"
DOWNSAMPLE_DIR="downsample_out/"
HEATMAP_DIR="heatmap_out/"

METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
EAST_PLY=${HPC_PATH}${RAW_DATA_PATH}${UUID}"__Top-heading-east_0.ply"
WEST_PLY=${HPC_PATH}${RAW_DATA_PATH}${UUID}"__Top-heading-west_0.ply"
MERGE_PLY=${MERGE_DIR}${UUID}"_icp_merge.ply"
MERGE_REF_PLY=${GEO_REF_DIR}${UUID}"_icp_merge_registered.ply"
GEO_COR_PLY=${GEO_COR_DIR}${UUID}"_corrected.ply"
MERGE_PNG=${MERGE_DIR}${UUID}"_merged_east_west.png"
#DOWNSAMPLED_PLY=${DOWNSAMPLE_DIR}${UUID}"_corrected_downsampled.ply"
DOWNSAMPLED_PLY=${DOWNSAMPLE_DIR}${UUID}"_icp_merge_registered_downsampled.ply"
COORD_JSON=${HEATMAP_DIR}${UUID}"_corrected_downsampled.json"
HEATMAP=${HEATMAP_DIR}${UUID}"_corrected_downsampled_heatmap.png"

#################################
# Merge east and west pointcloud#
#################################
EAST_PLY=${EAST_PLY}
WEST_PLY=${WEST_PLY}
MERGE_DIR=${MERGE_DIR}
WORKING_SPACE=${HPC_PATH}${RAW_DATA_PATH}
SIMG_PATH=${SIMG_PATH}

mkdir -p ${MERGE_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_icp_merge.simg -w ${WEST_PLY} -e ${EAST_PLY}

###################################
# Geo-reference merged point cloud#
###################################
MERGE_PLY=${MERGE_PLY}
METADATA=${METADATA}
GEO_REF_DIR=${GEO_REF_DIR}

mkdir -p ${GEO_REF_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_geo_ref.simg -m ${METADATA} ${MERGE_PLY}

########################
# Downsample pointcloud#
########################
DOWNSAMPLE_DIR=${DOWNSAMPLE_DIR}
MERGE_REF_PLY=${MERGE_REF_PLY}

mkdir -p ${DOWNSAMPLE_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_down_sample.simg ${MERGE_REF_PLY}

####################
# Generate heatmap #
####################
DOWNSAMPLED_PLY=${DOWNSAMPLED_PLY}
HEATMAP_DIR=${HEATMAP_DIR}
ORTHO_PATH=${HPC_PATH}"*.tif"

mkdir -p ${HEATMAP_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_heat_map.simg -d ${DOWNSAMPLED_PLY} -t ${ORTHO_PATH}







###############################################
# Correct georeferencing of merged point cloud#
############################################### 
#METADATA=${METADATA}
#MERGE_REF_PLY=${MERGE_REF_PLY}
#GEO_COR_DIR=${GEO_COR_DIR}
#PLANT_LOC_CLIP=${PLANT_LOC_CLIP}
#MERGE_PNG=${MERGE_PNG}
#DATE=${DATE}

#mkdir -p ${GEO_COR_DIR}
#singularity run ${SIMG_PATH}3d_geo_correction.simg -m ${HPC_PATH}model_weights_2021-03-26_10e_season10_3dpng.pth -d ${PLANT_LOC_CLIP} -s ${DATE} -p ${MERGE_REF_PLY} -j ${METADATA} -i ${MERGE_PNG}

#############################
# Clip out individual plants#
#############################
#PLANT_LOC_CLIP=${PLANT_LOC_CLIP}
#GEO_COR_PLY=${GEO_COR_PLY}
#PLANT_CLIP_DIR=${PLANT_CLIP_DIR}
#DATE=${DATE}

#mkdir -p ${PLANT_CLIP_DIR}
#singularity run ${SIMG_PATH}3d_plant_clip.simg -c ${PLANT_LOC_CLIP} -d ${DATE} ${GEO_COR_PLY}

####################################
# create tarball of plotclip result#
####################################
#tar -cvf ${UUID}_plantclip.tar ${PLANT_CLIP_DIR}
