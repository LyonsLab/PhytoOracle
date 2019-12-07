#!/bin/bash


makeflow -T wq --jx main_workflow.jx --jx-args raw_data_files.jx $@ -p 6000

