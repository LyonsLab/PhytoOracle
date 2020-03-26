#!/bin/bash

#python3 gen_files_list.py 2019-12-19_5sets/ > raw_data_files.json
#python3 gen_bundles_list.py raw_data_files.json bundle_list.json 2
#mkdir -p bundle/
#python3 split_bundle_list.py  bundle_list.json bundle/"

php main_wf_phase2.php > main_wf_phase2.jx
jx2json main_wf_phase2.jx > main_workflow_phase2.json


# -a advertise to catalog server
makeflow -T local --json main_workflow_phase2.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log --disable-cache $@

