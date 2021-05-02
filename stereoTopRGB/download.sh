#!/bin/bash 
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
iget -fKPVT /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/stereoTop/stereoTop-${TAR_FILE}.tar
tar -xvf stereoTop-${TAR_FILE}.tar
rm stereoTop-${TAR_FILE}.tar
mv stereoTop/* .
