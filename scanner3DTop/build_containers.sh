#!/bin/bash 

singularity build 3d_icp_merge.simg docker://phytooracle/3d_icp_merge:latest
singularity build 3d_geo_ref.simg docker://phytooracle/3d_geo_registration:latest
#singularity build 3d_geo_correction.simg docker://phytooracle/3d_geo_correction:latest
#singularity build 3d_plant_clip.simg docker://phytooracle/3d_plant_clip:latest
singularity build 3d_down_sample.simg docker://phytooracle/3d_down_sample:latest 
singularity build 3d_heat_map.simg docker://phytooracle/3d_heat_map:latest 
