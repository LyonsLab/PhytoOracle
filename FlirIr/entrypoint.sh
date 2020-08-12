#!/bin/bash

# Phase 1
module load python/3.8
python3 gen_files_list.py 2020-03-03 > raw_data_files.json
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 5
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
/home/u12/cosi/cctools-7.1.5-x86_64-centos7/bin/jx2json main_wf_phase1.jx -a bundle_list.json > main_workflow_phase1.json

# -a advertise to catalog server
/home/u12/cosi/cctools-7.1.5-x86_64-centos7/bin/makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle -p 9123 -dall -o dall.log --disable-cache $@

# Phase 2
module load singularity

mv *_plotclip.tar plotclip_out/
cd plotclip_out/
for f in *.tar; do tar -xvf "$f"; done
mkdir -p plotclip_raw_tars/
mv *.tar plotclip_raw_tars/
mv plotclip_raw_tars/ ../

cd ../
singularity run -B $(pwd):/mnt --pwd /mnt/ docker://cosimichele/po_meantemp:latest -g season10_multi_latlon_geno_up.geojson plotclip_out/plotclip_out/
