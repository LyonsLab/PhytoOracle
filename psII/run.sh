#!/bin/bash 
SCAN_DATE=${1%/}
module load python
set -e 

echo "Processing ${SCAN_DATE}"
./replace.py ${SCAN_DATE}

./entrypoint.sh

ls *_segmentation.tar | xargs -I {} tar -xvf {}

singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/psii_fluorescence_aggregation:latest -od ${SCAN_DATE}_fluorescence_aggregation -of ${SCAN_DATE}_fluorescence_aggregation -m multithresh.json psii_segmentation_out/*/*.csv

mv bundle_list.json bundle/
tar -czvf ps2Top-${SCAN_DATE}-bundle.tar.gz bundle/ 
tar -czvf ps2Top-${SCAN_DATE}-tif.tar.gz tifresize_out/
tar -czvf ps2Top-${SCAN_DATE}-cleanmetadata.tar.gz cleanmetadata_out/
tar -czvf ps2Top-${SCAN_DATE}-segmentation.tar.gz psii_segmentation_out/
mv ${SCAN_DATE}_fluorescence_aggregation/ fluorescence_outs/
mkdir -p ${SCAN_DATE}_segmentation/
mv *_segmentation.tar ${SCAN_DATE}_segmentation/

echo "Finished processing ${SCAN_DATE}"
