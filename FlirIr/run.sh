#!/bin/bash

# Defining Variables and Paths
DATE="2020-02-27" # Change to appropriate date (or data folder)!
SENSOR="flirIrCamera"

PIPE_PATH='/xdisk/ericlyons/big_data/cosi/PhytoOracle_flirIr/'
SIMG_PATH='/xdisk/ericlyons/big_data/singularity_images/'

FLIR_DIR="flir2tif_out/"
BIN_DIR="bin2tif_out/"
STITCH_ORTHO_DIR="stitched_ortho_out/"
PLOTCLIP_DIR="plotclip_out/"
GPS_DIR="gpscorrect_out/"

ORTHO_PATH=${PIPE_PATH}'ortho_out'
CSV_PATH=${PIPE_PATH}'img_coords_out/'${DATE}'_coordinates.csv'
GPS_CS_PATH=${ORTHO_PATH}"${DATE}_out/${DATE}_coordinates_CORRECTED.csv"

# Phase 1: CCTools Parallelization to HPC nodes (Image conversion)
python3 gen_files_list.py ${DATE} > raw_data_files.json
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 5
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
/home/u12/cosi/cctools-7.1.6-x86_64-centos7/bin/jx2json main_wf_phase1.jx -a bundle_list.json > main_workflow_phase1.json

# -a advertise to catalog server
/home/u12/cosi/cctools-7.1.6-x86_64-centos7/bin/makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle_FLIR -p 9123 -M PhytoOracle_FLIR -dall -o dall.log --disable-cache $@

# Phase 2: Parallelization on single node (GPS correction)
#module load singularity/3.2.1
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}collect_gps_latest.simg  --scandate ${DATE} ${FLIR_DIR}

mkdir -p ${ORTHO_PATH}/${DATE}_out/
mkdir -p ${ORTHO_PATH}/${DATE}_out/SIFT/
mkdir -p ${ORTHO_PATH}/${DATE}_out/logs/
cp ${CSV_PATH} ${ORTHO_PATH}/${DATE}_out/
cp ${PIPE_PATH}season10_ind_lettuce_2020-05-27.csv ${ORTHO_PATH}/${DATE}_out/
cp ${PIPE_PATH}lids.txt ${ORTHO_PATH}/${DATE}_out/
mv ${FLIR_DIR} ${BIN_DIR}
mv ${BIN_DIR} ${ORTHO_PATH}/${DATE}_out/

singularity exec ${SIMG_PATH}geo_correction_image_2.simg python ../Lettuce_Image_Stitching/Dockerized_GPS_Correction_FLIR.py ${DATE} ../Lettuce_Image_Stitching/geo_correction_config_FLIR.txt ${PATH_PATH}ortho_out

mv ${ORTHO_PATH}/${DATE}_out/${BIN_DIR} ${PIPE_PATH} && mv ${BIN_DIR} ${FLIR_DIR}

# Phase 3: Orthomosaic building and temperature extraction
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}flirfieldplot.simg -d ${DATE} -o ${STITCH_ORTHO_DIR} flir2tif_out/
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}plotclip_geo.simg -sen ${SENSOR} -shp season10_multi_latlon_geno_up.geojson -o ${PLOTCLIP_DIR} ${STITCH_ORTHO_DIR}${DATE}"_ortho_NaN.tif"
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}stitch_plots.simg ${PLOTCLIP_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt ${SIMG_PATH}po_temp_cv2stats.simg -g season10_multi_latlon_geno_up.geojson -o plot_meantemp_out/ -d ${DATE} ${PLOTCLIP_DIR}

#mv *_plotclip.tar plotclip_out/
#cd plotclip_out/
#for f in *.tar; do tar -xvf "$f"; done
#mkdir -p plotclip_raw_tars/
#mv *.tar plotclip_raw_tars/
#mv plotclip_raw_tars/ ../

#cd ../
#/home/u12/cosi/singularity/scripts/run-singularity -B $(pwd):/mnt --pwd /mnt/ docker://cosimichele/po_meantemp:latest -g season10_multi_latlon_geno_up.geojson plotclip_out/plotclip_out/
