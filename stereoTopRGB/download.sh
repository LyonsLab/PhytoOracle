#!/bin/bash 
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
./clean.sh
iget -fKPVT /iplant/home/shared/phytooracle/season_11_sorghum_yr_2020/level_0/stereoTop/stereoTop-${TAR_FILE}.tar.gz
tar -xzvf stereoTop-${TAR_FILE}.tar.gz
rm stereoTop-${TAR_FILE}.tar.gz
