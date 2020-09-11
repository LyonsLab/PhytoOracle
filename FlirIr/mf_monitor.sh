#!/bin/bash 

module load python/2.7/2.7.14
export CCTOOLS_HOME=/home/u12/cosi/cctools-7.1.6-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH
/home/u12/cosi/cctools-7.1.6-x86_64-centos7/bin/makeflow_monitor main_workflow_phase1.json.makeflowlog
