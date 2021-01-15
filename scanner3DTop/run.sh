#!/bin/bash 
#./clean_sing_cache.sh &&
#./clean_sing_cache.sh
SCAN_DATE=${1%/}

./replace.py $SCAN_DATE

./entrypoint_p1.sh
