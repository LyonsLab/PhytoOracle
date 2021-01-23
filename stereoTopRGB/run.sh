#!/bin/bash 

SCAN_DATE=${1%/}
TIF_DIR='bin2tif_out/'
PLOTCLIP_DIR='plotclip_out/'
PIPE_PATH=$PWD'/'
SIMG_PATH='/xdisk/ericlyons/big_data/singularity_images/'
OUT_PATH=${PIPE_PATH}'ortho_out/'
#CSV_PATH=${PIPE_PATH}'img_coords_out/'${SCAN_DATE}'_coordinates.csv'
set -e 

echo "> Processing ${SCAN_DATE} RGB scan."

ssh filexfer 'cd' "${PIPE_PATH}" '&& ./download.sh' ${SCAN_DATE} ${PIPE_PATH} '&& exit'

# --------------------------------------------------
echo "> Distributed workflow 1 of 2"
./replace.py ${SCAN_DATE}
./replace_process_one.py $PWD
./entrypoint.sh


# --------------------------------------------------
echo "> Geocorrection 1 of 1"
mkdir -p ortho_out/
singularity exec ${SIMG_PATH}full_geocorrection.simg python3 $HOME/Lettuce_Image_Stitching/Dockerized_GPS_Correction_HPC.py -d ${OUT_PATH} -b ${PIPEPATH}bin2tif_out -s ${SCAN_DATE} -c $HOME/Lettuce_Image_Stitching/geo_correction_config.txt -l ${PIPE_PATH}gcp_season_10.txt -r $HOME/Lettuce_Image_Stitching

# --------------------------------------------------
echo "> Distributed workflow 2 of 2"
./entrypoint-2.sh 

# --------------------------------------------------
echo "> Generating orthomosaics and detecting plants."
ls *_plotclip.tar | xargs -I {} tar -xvf {}
rm *_plotclip.tar
singularity run ${SIMG_PATH}rgb_flir_stitch_plots.simg -o ${SCAN_DATE}_plotclip_orthos ${PLOTCLIP_DIR}
singularity exec ${SIMG_PATH}gdal_313.simg bash ortho.sh gpscorrect_out ${SCAN_DATE}
singularity run ${SIMG_PATH}rgb_flir_plant_detection.simg -d ${SCAN_DATE} -m ${PIPE_PATH}model_weights.pth -g ${PIPE_PATH}season10_multi_latlon_geno.geojson -t RGB -o season10_plant_detection ${SCAN_DATE}_plotclip_orthos 

# --------------------------------------------------
echo "> Compressing outputs."
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
 
# --------------------------------------------------
echo "> Done processing ${SCAN_DATE} RGB scan."
