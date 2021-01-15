#!/bin/bash 
./entrypoint_p1.sh -c
rm -r cleanmetadata_out/
rm -r las_out/
rm -r plotclip_out/
rm -r bundle/
rm -r icp_registration_out/
rm -r rotation_registration_out/
rm -r geocorrect_out/
rm -r scale_rotate_out/
rm main_workflow_phase1.json
rm bundle_list.json
rm raw_data_files.jx 
rm raw_data_files.json
rm dall.log
rm main_workflow_phase1.json.*
rm main_workflow_phase2.json.*
rm main_workflow_phase2.jx.*
rm main_workflow_phase1.jx.*
