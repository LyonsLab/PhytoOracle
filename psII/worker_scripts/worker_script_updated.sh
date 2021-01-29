#!/bin/bash 

export CCTOOLS_HOME=${HOME}/cctools-7.1.12-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

${HOME}/cctools-7.1.12-x86_64-centos7/bin/slurm_submit_workers -N phytooracle -t 3600 -p "--account=<account> --partition=standard --nodes=1 --ntasks=1 --ntasks-per-node=1" 90
