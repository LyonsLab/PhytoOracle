#!/bin/bash 
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
iget -fKPVT /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_0/scanner3DTop/${TAR_FILE}.tar.gz
tar -xzvf ${TAR_FILE}.tar.gz
rm ${TAR_FILE}.tar.gz
#mv scanner3DTop/* .
