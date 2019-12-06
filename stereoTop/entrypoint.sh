#!/bin/bash


makeflow --jx main_workflow.jx --jx-args raw_data_files.jx $@

