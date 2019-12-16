#!/bin/bash

LEVEL_0_BASE_PATH=small_test_set/PNG/
LEVEL_1_BASE_PATH=small_test_set/PLY/

python3 gen_files_list.py $LEVEL_0_BASE_PATH LEVEL_0_PATH _metadata.json $LEVEL_1_BASE_PATH LEVEL_1_PATH __Top-heading-west_0.ply > raw_data_files.json

# 2 data set per bundle
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 2
mkdir -p bundle/
# split bundle file
python3 split_bundle_list.py  bundle_list.json bundle/
#php main_wf_phase1.php > main_wf_phase1.jx
#jx2json main_wf_phase1.jx > main_workflow_phase1.json

makeflow --jx main_workflow_phase1.jx $@
