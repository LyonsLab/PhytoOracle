#!/bin/bash
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
#cd ${TAR_FILE}

icd /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_1/scanner3DTop
#icd /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_1/scanner3DTop/${TAR_FILE}

iput -rfKPVT ${TAR_FILE}
./clean.sh
#iput -rKPVT alignment 
#iput -rKPVT preprocessing
