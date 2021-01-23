#!/bin/bash 
TAR_FILE=${1}
WORK_DIR=${2}

cd ${WORK_DIR}
#iget -fKPVT /iplant/home/shared/terraref/ua-mac/raw_tars/season_10_yr_2020/stereoTop/stereoTop-${TAR_FILE}.tar
iget -fKPVT /iplant/home/shared/terraref/ua-mac/raw_tars/old/stereoTop-${TAR_FILE}.tar
tar -xvf stereoTop-$(basename $TAR_FILE).tar

#prefix="stereoTop-"
#foo=$(basename ${TAR_FILE%.*})
#foo=${foo#"$prefix"}
#echo ${foo}
mv stereoTop/* .
