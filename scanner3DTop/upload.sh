#!/bin/bash
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}

icd /iplant/home/shared/phytooracle/season_12_sorghum_soybean_sunflower_tepary_yr_2021/level_1/scanner3DTop

iput -rfKPVT ${TAR_FILE}
./clean.sh
