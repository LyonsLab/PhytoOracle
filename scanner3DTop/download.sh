#!/bin/bash 
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
iget -fKPVT /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/scanner3DTop/scanner3DTop-${TAR_FILE}.tar
tar -xvf scanner3DTop-${TAR_FILE}.tar
rm scanner3DTop-${TAR_FILE}.tar
mv scanner3DTop/* .
