#!/bin/bash 
./entrypoint_p1.sh -c
./entrypoint_p2.sh -c
rm -r preprocessing_out/
rm -r sequential_alignment_out/
rm -r postprocessing_out/
rm -r individual_plants_out/
rm -r bundle/
rm -r makeflow.failed*
rm -r icp_registration_out/
rm -r rotation_registration_out/
rm -r geocorrect_out/
rm -r scale_rotate_out/
rm -r plantclip_out/
rm -r 3d_geo_correction_out/
rm -r downsample_out/
rm -r heatmap_out/ 
rm *.tif
rm -r wq-pool*
rm -f slurm-*
rm main_workflow_phase1.json
rm main_workflow_phase2.json
rm bundle_list.json
rm raw_data_files.jx 
rm raw_data_files.json
rm dall.log
rm main_workflow_phase1.json.*
rm main_workflow_phase2.json.*
rm main_workflow_phase2.jx.*
rm main_workflow_phase1.jx.*
