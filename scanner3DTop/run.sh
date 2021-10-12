#!/bin/bash 
SCAN_DATE=${1%/}
PIPE_PATH=$PWD'/'

#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Processing ${SCAN_DATE}" -c "gantry_data_updates"
#sbatch worker_scripts/po_work_puma_slurm.sh
#echo "> Processing ${SCAN_DATE} 3D scan."
#ssh filexfer 'cd' "${PIPE_PATH}" '&& ./download.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

SCAN_DATE="`echo ${SCAN_DATE} | grep -Eo '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}' | tail -1`"

# Set paths
#./replace.py ${SCAN_DATE}
#./replace_process_one.py $PWD

# Run pipeline
#./entrypoint_p1.sh
#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Finished processing scanner3DTop-${SCAN_DATE}. Compressing outputs." -c "gantry_data_updates"

# Tar outputs
singularity run ${PIPE_PATH}3d_sequential_align.simg -i preprocessing_out/ -o sequential_alignment_out/
#tar -cvf ${SCAN_DATE}_geo_correction.tar 3d_geo_correction_out/
#tar -cvf ${SCAN_DATE}_icp_registration.tar icp_registration_out/
#tar -cvf ${SCAN_DATE}_rotation_registration.tar rotation_registration_out/
#tar -cvf ${SCAN_DATE}_down_sampled.tar downsample_out/
#tar -cvf ${SCAN_DATE}_heat_map.tar heatmap_out/ 
#mkdir -p processed_scans
#mv ${SCAN_DATE}/ processed_scans/

# Place outputs in a single directory for upload
#mkdir -p ${SCAN_DATE}
#mv *.tar ${SCAN_DATE}
#./clean.sh

#Upload outputs
#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Finished compressing scanner3DTop-${SCAN_DATE}. Now uploading to CyVerse." -c "gantry_data_updates"
#ssh filexfer 'cd' "${PIPE_PATH}" '&& ./upload.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

#Cancel workers and clean the file system.
#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Upload complete. See ouputs at /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_1/scanner3DTop/${SCAN_DATE}" -c "gantry_data_updates"
#scancel --name=po_worker
#rm -r ${SCAN_DATE}
#rm -r processed_scans/
#rm *.tif 
