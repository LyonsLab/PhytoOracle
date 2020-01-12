#!/bin/bash

python3 gen_plot_list.py plotclip_out/ cleanmetadata_out/ LAS_FILES .las > plot_list.json

makeflow --jx main_workflow_phase2.jx --jx-args plot_list.json $@

