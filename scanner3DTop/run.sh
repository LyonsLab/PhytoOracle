#!/bin/bash 
SCAN_DATE=${1%/}
PIPE_PATH=$PWD'/'

#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Processing ${SCAN_DATE}" -c "gantry_data_updates"
sbatch worker_scripts/po_work_puma_slurm.sh
echo "> Processing ${SCAN_DATE} 3D scan."
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./download.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

#SCAN_DATE="`echo ${SCAN_DATE} | grep -Eo '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}' | tail -1`"

# Set paths
./replace.py ${SCAN_DATE}
./replace_process_one.py $PWD

# Run pipeline
./entrypoint_p1.sh
#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Finished processing ${SCAN_DATE}. Compressing outputs." -c "gantry_data_updates"

singularity run ${PIPE_PATH}3d_sequential_align.simg -i preprocessing_out/ -o sequential_alignment_out/

# Run pipeline 2
#./entrypoint_p2.sh

# Compress outputs
mkdir -p processed_scans/ 
mv ${SCAN_DATE} processed_scans/
mkdir ${SCAN_DATE}

cd ${PIPE_PATH}preprocessing_out/
tar -cvf ${SCAN_DATE}_east_preprocessed.tar east/
tar -cvf ${SCAN_DATE}_east_downsampled_preprocessed.tar east_downsampled/
tar -cvf ${SCAN_DATE}_west_preprocessed.tar west/
tar -cvf ${SCAN_DATE}_west_downsampled_preprocessed.tar west_downsampled/
tar -cvf ${SCAN_DATE}_merged_preprocessed.tar merged/
tar -cvf ${SCAN_DATE}_merged_downsampled_preprocessed.tar merged_downsampled/
tar -cvf ${SCAN_DATE}_metadata.tar metadata/
mkdir -p preprocessing/
mv *.tar preprocessing/ 
mv preprocessing/ ${PIPE_PATH}${SCAN_DATE}
cd ${PIPE_PATH}

cd ${PIPE_PATH}sequential_alignment_out/
tar -cvf ${SCAN_DATE}_east_aligned.tar east/
tar -cvf ${SCAN_DATE}_east_downsampled_aligned.tar east_downsampled/
tar -cvf ${SCAN_DATE}_west_aligned.tar west/
tar -cvf ${SCAN_DATE}_west_downsampled_aligned.tar west_downsampled/
tar -cvf ${SCAN_DATE}_merged_aligned.tar merged/
tar -cvf ${SCAN_DATE}_merged_downsampled_aligned.tar merged_downsampled/
tar -cvf ${SCAN_DATE}_metadata.tar metadata/
mkdir -p alignment/
mv *.tar alignment/
mv alignment/ ${PIPE_PATH}${SCAN_DATE}
cd ${PIPE_PATH}

#cd ${PIPE_PATH}postprocessing_out/
#tar -cvf ${SCAN_DATE}_east_postprocessed.tar east/
#tar -cvf ${SCAN_DATE}_east_downsampled_postprocessed.tar east_downsampled/
#tar -cvf ${SCAN_DATE}_west_postprocessed.tar west/
#tar -cvf ${SCAN_DATE}_west_downsampled_postprocessed.tar west_downsampled/
#tar -cvf ${SCAN_DATE}_merged_postprocessed.tar merged/
#tar -cvf ${SCAN_DATE}_merged_downsampled_postprocessed.tar merged_downsampled/
#mkdir -p postprocessing/
#mv *.tar postprocessing/
#mv postprocessing/ ${PIPE_PATH}${SCAN_DATE}
#cd ${PIPE_PATH}

#ls *_individual_plants.tar | xargs -I {} tar -xvf {}
#rm *_individual_plants.tar
#cd ${PIPE_PATH}individual_plants_out/
#tar -cvf ${SCAN_DATE}_east_plants.tar east/ 
#tar -cvf ${SCAN_DATE}_east_downsampled_plants.tar east_downsampled/
#tar -cvf ${SCAN_DATE}_west_plants.tar west/
#tar -cvf ${SCAN_DATE}_west_downsampled_plants.tar west_downsampled/
#tar -cvf ${SCAN_DATE}_merged_plants.tar merged/
#tar -cvf ${SCAN_DATE}_merged_downsampled_plants.tar merged_downsampled/
#mkdir -p plantcrop/
#mv *.tar plantcrop/
#mv plantcrop/ ${PIPE_PATH}${SCAN_DATE}
#cd ${PIPE_PATH}
#rm -r individual_plants_out/
#tar -cvf ${SCAN_DATE}_plant_crop.tar individual_plants_out/
#mv ${SCAN_DATE}_plant_crop.tar ${PIPE_PATH}${SCAN_DATE}
#rm -r individual_plants_out/


mv main_workflow_phase1.json* ${PIPE_PATH}${SCAN_DATE}
#mv main_workflow_phase2.json* ${PIPE_PATH}${SCAN_DATE}
mv bundle_list.json ${PIPE_PATH}${SCAN_DATE}
# Place outputs in a single directory for upload
#./clean.sh

#Upload outputs
#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Finished compressing scanner3DTop-${SCAN_DATE}. Now uploading to CyVerse." -c "gantry_data_updates"
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./upload.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

#Cancel workers and clean the file system.
#singularity run ${PIPE_PATH}gantry_notifications.simg -m "Upload complete. See ouputs at /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_1/scanner3DTop/${SCAN_DATE}" -c "gantry_data_updates"
scancel --name=po_worker
rm -r ${SCAN_DATE}
rm -r processed_scans/${SCAN_DATE}
#rm *.tif 
