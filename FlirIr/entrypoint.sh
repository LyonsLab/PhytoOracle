#!/bin/bash

# Phase 1
python3 gen_files_list.py 2020-01-16 > raw_data_files.json
python3 gen_bundles_list.py raw_data_files.json bundle_list.json 2
mkdir -p bundle/
python3 split_bundle_list.py  bundle_list.json bundle/
php main_wf_phase1.php > main_wf_phase1.jx
jx2json main_wf_phase1.jx > main_wf_phase1.json


# -a advertise to catalog server
makeflow -T wq --json main_wf_phase1.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log --disable-cache $@

# Phase 2
# Remove headers
for i in `ls meantemp_out`
do sed -i '1d' meantemp_out/${i}/meantemp_geostreams.csv
done

# Concatenate all files
for i in `ls meantemp_out`
do cat meantemp_out/${i}/meantemp_geostreams.csv >> t0.csv
done

# Edit Phas 2 output
cut -d ',' -f3,4,5,6,7 t0.csv > t1.csv
awk -F, '{print $4,$5,$1,$2,$3}' OFS=, t1.csv > t2.csv
sed 's/^.\{9\}//' t2.csv > meantemp.csv
sed -i '1 i\Plot, Temperature, Latitute, Longitude, Time' meantemp.csv

# Cleanup ***REMOVE THIS TO SEE MEANTEMP INTERMEDIATE***
rm -r t* meantemp_out/*
mv meantemp.csv meantemp_out/
