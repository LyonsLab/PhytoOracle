#!/bin/bash

./entrypoint.sh -c
rm -f makeflow.jx.args.*
rm -r bundle
rm bundle_list.json
rm raw_data_files.json
rm raw_data_files.jx
rm -r cleanmetadata_out
rm -r bin2tif_out
rm -r gpscorrect_out
rm dall.log
