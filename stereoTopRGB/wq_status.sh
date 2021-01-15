#!/bin/bash 

#module load python/3
export CCTOOLS_HOME=/home/u31/emmanuelgonzalez/cctools-7.1.6-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH
watch -n 1 /home/u31/emmanuelgonzalez/cctools-7.1.6-x86_64-centos7/bin/work_queue_status 
