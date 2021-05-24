#!/bin/bash

SCAN_DATE=${1%/}

mkdir ${SCAN_DATE}_out/
mv individual_thermal_out/ flir2tif_out/ plotclip_out/ plot_meantemp_out/ stitched_ortho_out/ ./${SCAN_DATE}_out/ 
rm -r bundle* main_workflow_phase1.json.*
mv ${SCAN_DATE}_out/ flirS11_processed_RE/ 
mv ${SCAN_DATE} flirIrCamera_S11
