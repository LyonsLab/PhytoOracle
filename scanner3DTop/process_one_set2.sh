#!/bin/bash

HPC_PATH="/xdisk/dukepauli/emmanuelgonzalez/testing/season_10/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}
PLANT_LOC_CLIP=${HPC_PATH}"season10_plant_detections_cleaned.csv"
DATE="`echo ${RAW_DATA_PATH} | grep -Eo '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}' | tail -1`"

#MERGE_DIR="icp_registration_out/"
#GEO_REF_DIR="rotation_registration_out/"
#GEO_COR_DIR="3d_geo_correction_out/"
#SCALE_ROT_DIR="scale_rotate_out/"
#PLANT_CLIP_DIR="plantclip_out/"
#DOWNSAMPLE_DIR="downsample_out/"
#HEATMAP_DIR="heatmap_out/"
PLANT_CROP="individual_plants_out/"

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



#######################################################################################################
################
#Post-processing#
################
singularity run ${SIMG_PATH}3d_postprocessing.simg -i ${HPC_PATH}alignment/ -o postprocessing_out -f ${SUBDIR%/} -t ${HPC_PATH}transfromation.json -p ${HPC_PATH}stereoTop_full_season_clustering.csv -s 10 -d ${DATE} -l ${HPC_PATH}gcp_season_10.txt

############
#Plant crop#
############
PLANT_CROP=${PLANT_CROP}

singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_crop_individual_plants.simg -i postprocessing_out/ -p ${HPC_PATH}stereoTop_full_season_clustering.csv -f ${SUBDIR%/} -o ${PLANT_CROP} -s 10 -d ${DATE}

#tar -cvf ${SUBDIR%/}_individual_plants.tar ${SUBDIR%/}
#mkdir -p individual_plants_out/
#mv ${SUBDIR%/}_individual_plants.tar individual_plants_out/

tar -cvf ${SUBDIR%/}_individual_plants.tar ${PLANT_CROP}

#######################################################################################################
#DEPRECATED
#######################################################################################################

#################################
# Merge east and west pointcloud#
#################################
#EAST_PLY=${EAST_PLY}
#WEST_PLY=${WEST_PLY}
#MERGE_DIR=${MERGE_DIR}
#WORKING_SPACE=${HPC_PATH}${RAW_DATA_PATH}
#SIMG_PATH=${SIMG_PATH}

#mkdir -p ${MERGE_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_icp_merge.simg -w ${WEST_PLY} -e ${EAST_PLY}

###################################
# Geo-reference merged point cloud#
###################################
#MERGE_PLY=${MERGE_PLY}
#METADATA=${METADATA}
#GEO_REF_DIR=${GEO_REF_DIR}

#mkdir -p ${GEO_REF_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_geo_ref.simg -m ${METADATA} ${MERGE_PLY}

########################
# Downsample pointcloud#
########################
#DOWNSAMPLE_DIR=${DOWNSAMPLE_DIR}
#MERGE_REF_PLY=${MERGE_REF_PLY}

#mkdir -p ${DOWNSAMPLE_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_down_sample.simg ${MERGE_REF_PLY}

####################
# Generate heatmap #
####################
#DOWNSAMPLED_PLY=${DOWNSAMPLED_PLY}
#HEATMAP_DIR=${HEATMAP_DIR}
#ORTHO_PATH=${HPC_PATH}"*.tif"

#mkdir -p ${HEATMAP_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_heat_map.simg -d ${DOWNSAMPLED_PLY} -t ${ORTHO_PATH}
