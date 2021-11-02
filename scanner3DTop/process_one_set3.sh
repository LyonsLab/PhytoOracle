#!/bin/bash

HPC_PATH="/xdisk/cjfrost/egonzalez/season_10/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}
PLANT_CROP="plantcrop/"

#############################################################################################
#Combine individual plant component point clouds into a single, individual plant pointcloud.#
#############################################################################################
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_individual_plant_registration.simg -i ${HPC_PATH}${PLANT_CROP} -p ${PLANT_NAME}
cd ${HPC_PATH}${PLANT_CROP}
tar -cvf ${PLANT_NAME}_combined_pointclouds.tar combined_pointclouds/${PLANT_NAME}
tar -cvf ${PLANT_NAME}_plant_reports.tar plant_reports/${PLANT_NAME}

