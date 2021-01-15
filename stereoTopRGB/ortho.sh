#!/bin/bash 
TIF_DIR=${1%/}'/'
DATE=${2%/}
set -e 

gdalbuildvrt mosaic.vrt ${TIF_DIR}*.tif
gdal_translate -co COMPRESS=LZW -co BIGTIFF=YES -outsize 10% 10% -r cubic -co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 512 mosaic.vrt ${DATE}_ortho_10pct_cubic.tif
