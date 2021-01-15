#!/bin/bash 
TAR_FILE=${1}
ssh filexfer ./download.sh $TAR_FILE && exit

#/xdisk/ericlyons/big_data/egonzalez/PhytoOracle/stereoTopRGB/interactive_sh/gpu_init_puma.sh 
#echo "${TAR_FILE%.*}"
#echo $(basename ${TAR_FILE%.*})
#mv /xdisk/ericlyons/big_data/egonzalez/PhytoOracle/stereoTopRGB/stereoTop/$(basename ${stereoTop-*TAR_FILE%.*}) ../
#prefix="stereoTop-"
#foo=$(basename ${TAR_FILE%.*})
#echo $foo
#foo=${foo#"$prefix"}
#echo $foo
#/xdisk/ericlyons/big_data/egonzalez/PhytoOracle/stereoTopRGB/run.sh ${TAR_FILE%.*}
