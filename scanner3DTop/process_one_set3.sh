#!/bin/bash

HPC_PATH="/xdisk/ericlyons/data/emmanuelgonzalez/testing/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}
#PLANT_CROP="plantcrop/"
PLANT_CROP='individual_plants_out/'
MODEL='dgcnn_trained_on_normalized_arman_format_training_data_perfect_partial_full_30epcs_98acc_47loss.pth'

#############################################################################################
#Combine individual plant component point clouds into a single, individual plant pointcloud.#
#############################################################################################
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_individual_plant_registration.simg -i ${HPC_PATH}${PLANT_CROP} -p ${PLANT_NAME}

#################################################
#Run inference to seperate soil and plant points#
#################################################
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}dgcnn_single_plant_soil_segmentation.simg -i ${HPC_PATH}${PLANT_CROP}combined_pointclouds/${PLANT_NAME} --model_path ${HPC_PATH}${MODEL}

cd ${HPC_PATH}${PLANT_CROP}
tar -cvf ${PLANT_NAME}_combined_pointclouds.tar combined_pointclouds/${PLANT_NAME}
tar -cvf ${PLANT_NAME}_plant_reports.tar plant_reports/${PLANT_NAME}
