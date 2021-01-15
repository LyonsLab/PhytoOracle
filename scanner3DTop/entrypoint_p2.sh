#!/bin/bash

#ls *.tar | xargs -I {} tar -xvf {}

#python3 gen_plot_list.py plotclip_out/ cleanmetadata_out/ LAS_FILES .las > plot_list.json

/home/u31/emmanuelgonzalez/cctools-7.1.2-x86_64-centos7/bin/makeflow --jx main_workflow_phase2.jx --jx-args plot_list.json $@

