#!/bin/bash 

module load python/2.7/2.7.14
export CCTOOLS_HOME=${HOME}/cctools-7.1.2-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH
${HOME}/cctools-7.1.12-x86_64-centos7/bin/makeflow_monitor main_workflow_phase1.json.makeflowlog
