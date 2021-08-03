#!/bin/bash 
SCAN_DATE=${1%/}
TIF_DIR='bin2tif_out/'
PLOTCLIP_DIR='plotclip_out/'
PIPE_PATH=$PWD'/'
SIMG_PATH=${PIPE_PATH}
OUT_PATH=${PIPE_PATH}'ortho_out/' 

#---------------------------------------------Download raw data
singularity run ${SIMG_PATH}slack_notification.simg -m "Downloading and preparing to process stereoTop-${SCAN_DATE}." 
echo "> Processing ${SCAN_DATE} RGB scan."
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./download.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

# ---------------------------------------------Workflow 1
echo "> Distributed workflow 1 of 2"
./replace.py "stereoTop-${SCAN_DATE}"
./replace_process_one.py $PWD
sbatch worker_scripts/po_work_puma_slurm.sh
./entrypoint.sh
scancel --name=po_worker_emm

# ---------------------------------------------Geocorrection
echo "> Geocorrection 1 of 1"
mkdir -p ortho_out/
singularity exec ${SIMG_PATH}full_geocorrection.simg python3 $HOME/Lettuce_Image_Stitching/Dockerized_GPS_Correction_HPC.py -d ${OUT_PATH} -b ${PIPEPATH}bin2tif_out -s ${SCAN_DATE} -c $HOME/Lettuce_Image_Stitching/geo_correction_config.txt -l ${PIPE_PATH}season12_all_bucket_gcps.txt -r $HOME/Lettuce_Image_Stitching

# ---------------------------------------------Workflow 2
echo "> Distributed workflow 2 of 2"
sbatch worker_scripts/po_work_puma_slurm.sh
./entrypoint-2.sh 
scancel --name=po_worker_emm

# ---------------------------------------------Deploy detection
echo "> Generating orthomosaics and detecting plants."
#ls *_plotclip.tar | xargs -I {} tar -xvf {}
#rm *_plotclip.tar
#singularity run ${SIMG_PATH}rgb_flir_stitch_plots.simg -o ${SCAN_DATE}_plotclip_orthos ${PLOTCLIP_DIR}
singularity exec ${SIMG_PATH}gdal_313.simg gdalbuildvrt mosaic.vrt gpscorrect_out/*.tif
singularity exec ${SIMG_PATH}gdal_313.simg gdal_translate -co COMPRESS=LZW -co BIGTIFF=YES -outsize 10% 10% -r cubic -co NUM_THREADS=ALL_CPUS mosaic.vrt ${SCAN_DATE}_ortho_10pct_cubic.tif
#singularity run ${SIMG_PATH}rgb_flir_plant_detection.simg -d ${SCAN_DATE} -m ${PIPE_PATH}model_weights_sorghum_rgb.pth -g ${PIPE_PATH}season11_multi_latlon_geno.geojson -c plant -t RGB -o season11_plant_detection ${SCAN_DATE}_plotclip_orthos 

# ---------------------------------------------Compress output
singularity run ${SIMG_PATH}slack_notification.simg -m "Finished processing stereoTop-${SCAN_DATE}. Now compressing outputs."
echo "> Compressing outputs."
tar -cvf ${SCAN_DATE}_bin2tif.tar bin2tif_out/
tar -cvf ${SCAN_DATE}_gpscorrect.tar gpscorrect_out/
#tar -cvf ${SCAN_DATE}_plotclip_tifs.tar plotclip_out/
cp bundle_list.json bundle/
cp raw_data_files.json bundle/
tar -cvf ${SCAN_DATE}_bundle.tar bundle/
#tar -cvf ${SCAN_DATE}_plotclip_orthos.tar ${SCAN_DATE}_plotclip_orthos/ 
#rm -r ${SCAN_DATE}_plotclip_orthos/
rm -r stereoTop-${SCAN_DATE}
mkdir -p ${SCAN_DATE}
mv ${SCAN_DATE}_* ${SCAN_DATE}/
singularity run ${SIMG_PATH}slack_notification.simg -m "Finished compressing stereoTop-${SCAN_DATE}. Now uploading to the CyVerse DataStore."

#-----------------------------------------------Upload outputs
ssh filexfer 'cd' "${PIPE_PATH}" '&& ./upload.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'
singularity run ${SIMG_PATH}slack_notification.simg -m "Upload complete. See outputs at /iplant/home/shared/phytooracle/season_11_sorghum_yr_2020/level_1/stereoTop/${SCAN_DATE}/."
