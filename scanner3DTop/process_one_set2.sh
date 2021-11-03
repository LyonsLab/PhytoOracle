#!/bin/bash

HPC_PATH="/xdisk/dukepauli/emmanuelgonzalez/testing/season_10/PhytoOracle/scanner3DTop/"
SIMG_PATH=${HPC_PATH}
DATE="`echo ${RAW_DATA_PATH} | grep -Eo '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}' | tail -1`"
PLANT_CROP="individual_plants_out/"

################
#Post-processing#
################
singularity run ${SIMG_PATH}3d_postprocessing.simg -i ${HPC_PATH}alignment/ -o postprocessing_out -f ${SUBDIR%/} -t ${HPC_PATH}transfromation.json -p ${HPC_PATH}stereoTop_full_season_clustering.csv -s 10 -d ${DATE} -l ${HPC_PATH}gcp_season_10.txt

############
#Plant crop#
############
PLANT_CROP=${PLANT_CROP}

singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}3d_crop_individual_plants.simg -i postprocessing_out/ -p ${HPC_PATH}stereoTop_full_season_clustering.csv -f ${SUBDIR%/} -o ${PLANT_CROP} -s 10 -d ${DATE}

tar -cvf ${SUBDIR%/}_individual_plants.tar ${PLANT_CROP}
