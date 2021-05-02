#!/bin/bash 
SCAN_DATE=${1%/}
PIPE_PATH=$PWD'/'

sbatch worker_scripts/po_work_puma_slurm.sh
echo "> Processing ${SCAN_DATE} 3D scan."
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./download.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

# Set paths
./replace.py $SCAN_DATE
./replace_process_one.py $PWD

# Run pipeline
./entrypoint_p1.sh

# Tar outputs
tar -cvf ${SCAN_DATE}_geo_correction.tar 3d_geo_correction_out/
tar -cvf ${SCAN_DATE}_icp_registration.tar icp_registration_out/
tar -cvf ${SCAN_DATE}_rotation_registration.tar rotation_registration_out/
ls *_plantclip.tar | xargs -I {} tar -xvf {}
tar -cvf ${SCAN_DATE}_individual_plant.tar plantclip_out/
#rm *_plantclip.tar 
mkdir -p processed_scans
mv ${SCAN_DATE}/ processed_scans/

# Place outputs in a single directory for upload
mkdir -p ${SCAN_DATE}
rm *_plantclip.tar
mv *.tar ${SCAN_DATE}
./clean.sh

#Upload outputs
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./upload.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

#Cancel workers and clean the file system.
scancel --name=po_worker
rm -r ${SCAN_DATE}
rm -r processed_scans/ 
