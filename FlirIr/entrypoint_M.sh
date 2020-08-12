#!/bin/bash

DATE="2020-02-08"
SENSOR="flirIrCamera"
STITCH_ORTHO_DIR="stitched_ortho_out/"
PLOTCLIP_DIR="plotclip_out/"

# Phase 1
python3 gen_files_list.py ${DATE} > raw_data_files.json
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 5
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
/home/u12/cosi/cctools-7.1.5-x86_64-centos7/bin/jx2json main_wf_phase1.jx -a bundle_list.json > main_workflow_phase1.json

# -a advertise to catalog server
/home/u12/cosi/cctools-7.1.5-x86_64-centos7/bin/makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle -p 9123 -M PhytoOracle_FLIR -dall -o dall.log --disable-cache $@

# Phase 2
#module load singularity/3.2.1

singularity run -B $(pwd):/mnt --pwd /mnt/ docker://cosimichele/flirfieldplot -d ${DATE} -o ${STITCH_ORTHO_DIR} flir2tif_out/
singularity run -B $(pwd):/mnt --pwd /mnt/ docker://emmanuelgonzalez/plotclip_geo -sen ${SENSOR} -shp season10_multi_latlon_geno_up.geojson -o ${PLOTCLIP_DIR} ${STITCH_ORTHO_DIR}${DATE}"_ortho_NaN.tif"
singularity run -B $(pwd):/mnt --pwd /mnt/ docker://emmanuelgonzalez/stitch_plots ${PLOTCLIP_DIR}
singularity run -B $(pwd):/mnt --pwd /mnt/ docker://cosimichele/po_temp_cv2stats -g season10_multi_latlon_geno_up.geojson -o plot_meantemp_out/ -d ${DATE} ${PLOTCLIP_DIR}

#mv *_plotclip.tar plotclip_out/
#cd plotclip_out/
#for f in *.tar; do tar -xvf "$f"; done
#mkdir -p plotclip_raw_tars/
#mv *.tar plotclip_raw_tars/
#mv plotclip_raw_tars/ ../

#cd ../
#/home/u12/cosi/singularity/scripts/run-singularity -B $(pwd):/mnt --pwd /mnt/ docker://cosimichele/po_meantemp:latest -g season10_multi_latlon_geno_up.geojson plotclip_out/plotclip_out/
