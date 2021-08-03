#!/bin/bash 
wget https://raw.githubusercontent.com/phytooracle/pipeline_automation/main/pipeline.py
singularity build slack_notification.simg docker://phytooracle/slack_notifications:latest
singularity build rgb_flir_plant_detection.simg docker://phytooracle/rgb_flir_plant_detection:latest 
singularity build rgb_flir_stitch_plots.simg docker://phytooracle/rgb_flir_stitch_plots:latest
singularity build gdal_313.simg docker://osgeo/gdal:ubuntu-full-3.1.3
singularity build rgb_bin2tif.simg docker://phytooracle/rgb_bin_to_tif:latest 
singularity build edit_gps.simg docker://phytooracle/rgb_flir_edit_gps:latest
singularity build rgb_flir_plot_clip_geojson.simg docker://phytooracle/rgb_flir_plot_clip_geojson:latest

