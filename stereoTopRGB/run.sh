#!/bin/bash 
#module load python/3.8/3.8.0

SCAN_DATE=${1%/}
TIF_DIR='bin2tif_out/'
PLOTCLIP_DIR='plotclip_out/'
PIPE_PATH='/xdisk/ericlyons/big_data/egonzalez/PhytoOracle/stereoTopRGB/'
SIMG_PATH='/xdisk/ericlyons/big_data/singularity_images/'
OUT_PATH=${PIPE_PATH}'ortho_out/'
CSV_PATH=${PIPE_PATH}'img_coords_out/'${SCAN_DATE}'_coordinates.csv'
set -e 

echo "Processing ${SCAN_DATE} RGB scan."
# Part 1 > distributed
./replace.py ${SCAN_DATE}

./entrypoint.sh

# Part 2 > undistributed
#module load singularity
singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/collect_gps:latest --scandate ${SCAN_DATE} ${TIF_DIR}

singularity exec ${SIMG_PATH}geo_correction_image_2.simg python ../Lettuce_Image_Stitching/Dockerized_GPS_Correction_HPC.py -d ${OUT_PATH} -b ${PIPE_PATH}bin2tif_out -g ${CSV_PATH} -s ${SCAN_DATE} -c ../Lettuce_Image_Stitching/geo_correction_config.txt -l ${PIPE_PATH}lids.txt -u ${PIPE_PATH}season10_ind_lettuce_2020-05-27.csv

# Part 3 > distributed
ls ${CSV_PATH}

./replace_csv.py -s ${SCAN_DATE} process_one_set2.sh

./entrypoint-2.sh 

ls *_plotclip.tar | xargs -I {} tar -xvf {} &&

mkdir ${SCAN_DATE}_plotclip_tars/ &&

mv *_plotclip.tar ${SCAN_DATE}_plotclip_tars/ &&

singularity run -B $(pwd):/mnt --pwd /mnt docker://emmanuelgonzalez/stitch_plots:latest ${PLOTCLIP_DIR}

tar -czvf ${SCAN_DATE}_bin2tif.tar.gz bin2tif_out/
tar -czvf ${SCAN_DATE}_cleanmetadata.tar.gz cleanmetadata_out/
tar -czvf ${SCAN_DATE}_gpscorrect.tar.gz gpscorrect_out/
mv bundle_list.json bundle/
tar -czvf ${SCAN_DATE}_bundle.tar.gz bundle/
mkdir ${SCAN_DATE}_plotclip_orthos/ 
mv plotclip_out/*/*_ortho.tif ${SCAN_DATE}_plotclip_orthos/
tar -czvf ${SCAN_DATE}_plotclip_orthos.tar.gz ${SCAN_DATE}_plotclip_orthos/ 

echo "Done processing ${SCAN_DATE} RGB scan."
