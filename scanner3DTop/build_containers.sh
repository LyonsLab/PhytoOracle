#!/bin/bash 

singularity build gantry_notifications.simg docker://phytooracle/slack_notifications:latest
singularity build 3d_preprocessing.simg docker://phytooracle/3d_preprocessing:latest
singularity build 3d_sequential_align.simg docker://phytooracle/3d_sequential_align:latest

