#!/bin/bash

./entrypoint.sh -c
./entrypoint-2.sh -c
rm -f makeflow.jx.args.*
rm -r bundle
rm bundle_list.json
rm raw_data_files.json
rm raw_data_files.jx
rm *.vrt
#rm -r cleanmetadata_out
rm -r bin2tif_out
rm -r gpscorrect_out
rm -r plotclip_out
rm -r plotclip_orthos/
rm dall.log
rm main_workflow_phase1.json.*
rm main_workflow_phase2.json.*
rm wq-pool-*
