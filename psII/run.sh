#!/bin/bash 
SCAN_DATE=${1%/}
SIMG_PATH='/scratch/singularity_images/'
#module load python
set -e 

echo "Processing ${SCAN_DATE}"
./replace.py ${SCAN_DATE}
./replace_process_one.py $PWD 
./entrypoint.sh

ls *_segmentation.tar | xargs -I {} tar -xvf {}

singularity run ${SIMG_PATH}psii_fluorescence_aggregation.simg -od ${SCAN_DATE}_fluorescence_aggregation -of ${SCAN_DATE}_fluorescence_aggregation -m multithresh.json psii_segmentation_out

#cp bundle_list.json bundle/
#tar -cvf ps2Top-${SCAN_DATE}-bundle.tar bundle/ 
#tar -cvf ps2Top-${SCAN_DATE}-tif.tar bin2tif_out/
#tar -cvf ps2Top-${SCAN_DATE}-segmentation.tar psii_segmentation_out/
#mv ${SCAN_DATE}_fluorescence_aggregation/ fluorescence_outs/
#mkdir -p ${SCAN_DATE}_segmentation/
#mv *_segmentation.tar ${SCAN_DATE}_segmentation/
#mv ${SCAN_DATE} processed_scans/ 
#mkdir ${SCAN_DATE}
#mv ps2Top*${SCAN_DATE}* ${SCAN_DATE}
#mv ${SCAN_DATE}_segmentation/ segmentation_outs/  
echo "Finished processing ${SCAN_DATE}"
