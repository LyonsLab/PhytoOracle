#!/bin/bash
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
icd /iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/level_1/scanner3DTop

iput -rKPVT ${TAR_FILE}
