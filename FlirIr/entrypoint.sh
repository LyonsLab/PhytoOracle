#!/bin/bash 

python3 gen_files_list.py 2020-09-15 > raw_data_files.json
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 50
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
jx2json main_wf_phase1.jx -a bundle_list.json > main_workflow_phase1.json

# -a advertise to catalog server
makeflow -r 100 -T wq --json main_workflow_phase1.json -a -M phytooracle_flir -p 0 -dall -o dall.log --disable-cache $@
