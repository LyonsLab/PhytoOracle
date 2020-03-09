#!/bin/bash

python3 gen_files_list.py 2020-01-14/ RAW_DATA_PATH _metadata.json > raw_data_files.json
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 1
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
jx2json main_workflow_phase1.jx -a bundle_list.json > main_workflow_phase1.json
makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log $@

