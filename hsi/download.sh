#!/bin/bash 
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
#iget -fKPVT /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTop/stereoTop-${TAR_FILE}.tar
iget -fKPVT /iplant/home/shared/phytooracle/season_12_sorghum_soybean_sunflower_tepary_yr_2021/level_0/VNIR/${TAR_FILE}.tar.gz
tar -xzvf ${TAR_FILE}.tar.gz

