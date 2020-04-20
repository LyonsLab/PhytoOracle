#!/bin/bash

./entrypoint.sh -c
rm -R bundle/
rm -R makeflow.failed.*
rm -R cleanmetadata_out
rm -R soil_mask_out
rm -R canopy_cover_out
rm -rf .singularity/
rm -r meantemp_out
rm dall.log
rm bundle_list.json
rm main_wf_phase1.jx
rm main_wf_phase2.jx
rm main_workflow_phase1.json*
rm main_workflow_phase2.json*
rm raw_data_files.jx
rm raw_data_files.json
rm -R flir2tif_out
rm main_wf_phase1.json
