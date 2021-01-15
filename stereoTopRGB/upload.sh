#!/bin/bash 
SCAN_DATE=${1%/}

# Upload to iRODS
icd /iplant/home/shared/terraref/ua-mac/level_1/season_10_yr_2020/stereoTop
iput -rKPVT ${SCAN_DATE}

