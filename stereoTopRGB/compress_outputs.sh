#!/bin/bash
SCAN_DATE=${1%/}

tar -cvf ${SCAN_DATE}_bin2tif.tar bin2tif_out/
tar -cvf ${SCAN_DATE}_gpscorrect.tar gpscorrect_out/
tar -cvf ${SCAN_DATE}_plotclip_tifs.tar plotclip_out/
cp bundle_list.json bundle/
cp raw_data_files.json bundle/
tar -cvf ${SCAN_DATE}_bundle.tar bundle/
tar -cvf ${SCAN_DATE}_plotclip_orthos.tar ${SCAN_DATE}_plotclip_orthos/
rm -r ${SCAN_DATE}_plotclip_orthos/
mv ${SCAN_DATE} processed_scans/
mkdir -p ${SCAN_DATE}
mv ${SCAN_DATE}_* ${SCAN_DATE}/
