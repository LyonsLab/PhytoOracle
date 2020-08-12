#!/bin/bash

DATE="2020-02-08"
DIR=${DATE}"_processed/"
NAME=${DATE}"_processed"
TAR=${NAME}".tar"
CLEANMETA=${DATE}"_cleanmetadata_out/"
FLIR2TIF=${DATE}"_flir2tif_out/"
PLOTCLIP=${DATE}"_plotclip_out/"
PLOTMEAN=${DATE}"_plot_meantemp_out/"
STITCHED=${DATE}"_stitched_ortho_out/"

mkdir $DIR
mv cleanmetadata_out/ flir2tif_out/ plotclip_out/ plot_meantemp_out/ stitched_ortho_out/ ${DIR}
mv ${DIR} ../FlirIr_processed_final/folders/
cd ../FlirIr_processed_final/folders/${DIR}
cp -r cleanmetadata_out/ ${CLEANMETA} && cp -r flir2tif_out/ ${FLIR2TIF} && cp -r plotclip_out/ ${PLOTCLIP} && cp -r plot_meantemp_out/ ${PLOTMEAN} && cp -r stitched_ortho_out/ ${STITCHED}
mv ${CLEANMETA} ../../levels/0.cleametadata_processed/
mv ${FLIR2TIF}  ../../levels/1.flir2tif_processed/
mv ${STITCHED} ../../levels/2.stitched_ortho_processed/
mv ${PLOTCLIP} ../../levels/3.plotclip_processed/
mv ${PLOTMEAN} ../../levels/4.plot_meantemp_out/
cd ../
tar -cvf ${TAR} ${DIR}
mv ${TAR} ../tars/
