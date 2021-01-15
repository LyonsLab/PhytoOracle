#!/bin/bash

#python3 gen_files_list.py 2019-12-13 > raw_data_files.json
#python3 gen_bundles_list.py raw_data_files.json bundle_list.json 6
#mkdir -p bundle/
#python3 split_bundle_list.py bundle_list.json bundle/
${HOME}/cctools-7.1.6-x86_64-centos7/bin/jx2json main_wf_phase-2.jx -a bundle_list.json > main_workflow_phase2.json


# -a advertise to catalog server
${HOME}/cctools-7.1.6-x86_64-centos7/bin/makeflow -T wq --json main_workflow_phase2.json -a -M phytooracle -N phytooracle -p 60221 -R 1000 -dall -o dall.log --disable-cache $@

