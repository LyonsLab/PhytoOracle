#!/bin/bash 
#./clean_sing_cache.sh &&
#./clean_sing_cache.sh
SCAN_DATE=${1%/}

./replace.py $SCAN_DATE
./replace_process_one.py $PWD

./entrypoint_p1.sh
