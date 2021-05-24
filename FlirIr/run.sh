#!/bin/bash

DATE=${1%/} #"flirIrCamera-2020-09-09__03-03-09-696" # Change to appropriate date (or data folder)!
SENSOR="flirIrCamera"

#PIPE_PATH='/xdisk/ericlyons/big_data/cosi/PhytoOracle_flirIr/'
SIMG_PATH='/scratch/singularity_images/'

FLIR_DIR="flir2tif_out/"
BIN_DIR="bin2tif_out/"
STITCH_ORTHO_DIR="stitched_ortho_out/"
PLOTCLIP_DIR="plotclip_out/"
GPS_DIR="gpscorrect_out/"

./replace.py ${DATE}
./replace_process_one.py $PWD
./entrypoint.sh
# Phase 2: Orthomosaic building and temperature extraction
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}flirfieldplot.simg -d ${DATE} -o ${STITCH_ORTHO_DIR} ${FLIR_DIR}
#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}plotclip_geo.simg -sen ${SENSOR} -shp season11_multi_latlon_geno.geojson -o ${PLOTCLIP_DIR} ${STITCH_ORTHO_DIR}${DATE}"_ortho.tif"

#cd ${PLOTCLIP_DIR}
#for subdir in *; do mv ${subdir}/*_ortho.tif ${subdir}/${subdir}_ortho.tif; done;
#cd ../

#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}po_temp_cv2stats.simg -g season11_multi_latlon_geno.geojson -o plot_meantemp_out/ -d ${DATE} ${PLOTCLIP_DIR}

#singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}flir_plant_tempS11.simg -d ${DATE}  -g season11_multi_latlon_geno.geojson -m model_weights_sorghum_flir.pth -c 20 ${PLOTCLIP_DIR}

