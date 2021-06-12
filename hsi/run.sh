#!/bin/bash 
SCAN_DATE=${1%/}
PIPE_PATH=$PWD'/'
set -e 

echo "Processing ${SCAN_DATE}"
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./download.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'
sbatch worker_scripts/po_work_puma_slurm.sh
./replace.py ${SCAN_DATE}
./replace_process_one.py $PWD 
./entrypoint.sh
scancel --name=po_worker
rm -r ${SCAN_DATE}

cp bundle_list.json bundle/
tar -cvf ${SCAN_DATE}-bundle.tar bundle/
tar -cvf ${SCAN_DATE}-hsi_h5_out.tar hsi_h5_out/
tar -cvf ${SCAN_DATE}-hsi_rgb_out.tar hsi_rgb_out/
mkdir ${SCAN_DATE}
mv ${SCAN_DATE}-* ${SCAN_DATE}
 
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./upload.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'
echo "Finished processing ${SCAN_DATE}"
