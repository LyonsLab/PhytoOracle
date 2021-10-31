#!/bin/bash

HPC_PATH="/xdisk/dukepauli/emmanuelgonzalez/testing/season_10/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}
PLANT_LOC_CLIP=${HPC_PATH}"season10_plant_detections_cleaned.csv"
DATE="`echo ${RAW_DATA_PATH} | grep -Eo '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}' | tail -1`"

PLANT_CROP="plantcrop/"

singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_individual_plant_registration.simg -i ${HPC_PATH}${PLANT_CROP} -p ${PLANT_NAME}

#tar -cvf ${SUBDIR%/}_individual_plants.tar ${SUBDIR%/}
#mkdir -p individual_plants_out/
#mv ${SUBDIR%/}_individual_plants.tar individual_plants_out/

cd ${HPC_PATH}${PLANT_CROP}
tar -cvf ${PLANT_NAME}_combined_pointclouds.tar combined_pointclouds/${PLANT_NAME}
tar -cvf ${PLANT_NAME}_plant_reports.tar plant_reports/${PLANT_NAME}

