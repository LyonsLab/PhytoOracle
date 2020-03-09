#!/bin/bash

python3 gen_files_list.py 2020-02-12-subset/ RAW_DATA_PATH _metadata.json > raw_data_files.json

python3 gen_bundles_list.py raw_data_files.json bundle_list.json 1
