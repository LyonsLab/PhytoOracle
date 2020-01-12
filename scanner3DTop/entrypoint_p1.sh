#!/bin/bash

LEVEL_0_BASE_PATH=scanner3DTop_data/PNG/
LEVEL_1_BASE_PATH=scanner3DTop_data/PLY/

python3 gen_files_list.py $LEVEL_0_BASE_PATH LEVEL_0_PATH _metadata.json $LEVEL_1_BASE_PATH LEVEL_1_PATH __Top-heading-west_0.ply > raw_data_files.json

# 1 data set per bundle
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 1
mkdir -p bundle/
# split bundle file
python3 split_bundle_list.py  bundle_list.json bundle/

jx2json main_workflow_phase1.jx -a bundle_list.json > main_workflow_phase1.json
makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log $@

