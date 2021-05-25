#!/bin/bash 

./entrypoint.sh -c 
#rm -r cleanmetadata_out/
rm -r bin2tif_out/
rm -r bundle/
rm -r psii_segmentation_out/
rm -r segmentation_outs/
rm bundle_list.json
rm raw_data_files.json
rm -r wq-pool-*
rm main_workflow_phase1.json
rm dall.log
