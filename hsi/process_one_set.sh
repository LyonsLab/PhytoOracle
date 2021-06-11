#!/bin/bash

DATA_PATH="/tmp/"
HPC_PATH="/xdisk/cjfrost/egonzalez/po_season11/PhytoOracle/hsi/"
#SIMG_PATH="/xdisk/ericlyons/big_data/singularity_images/"
GEOJ=${HPC_PATH}"season11_multi_latlon_geno.geojson"
H5_DIR="hsi_h5_out/"
RGB_DIR="hsi_rgb_out/"

METADATA=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_metadata.json"
HDR_FILE=${HPC_PATH}${RAW_DATA_PATH}${UUID}"_raw.hdr"
H5_FILE=${HPC_PATH}${H5_DIR}${UUID}".h5"
set -e

# Generate H5/pseudo-RGB
RGB_DIR=${RGB_DIR}
H5_DIR=${H5_DIR}
H5_FILE=${H5_FILE}

singularity run -B $(pwd):/mnt --pwd /mnt ${HPC_PATH}hsi_envi_to_h5.simg ${HDR_FILE} 

# Add soil/NDVI mask to H5 file
#H5_FILE=${H5_FILE}

#singularity run -B $(pwd):/mnt --pwd /mnt ${HPC_PATH}hsi_soil_ndvi_mask.simg ${H5_FILE}
