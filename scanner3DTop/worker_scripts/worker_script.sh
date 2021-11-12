#!/bin/bash 

export CCTOOLS_HOME=${HOME}/cctools-7.1.6-x86_64-centos7
export PATH=${CCTOOLS_HOME}/bin:$PATH

#${HOME}/cctools-7.1.6-x86_64-centos7/bin/work_queue_worker -M phyto_oracle-atmo --cores 94 -t 259200
#${HOME}/cctools-7.1.6-x86_64-centos7/bin/work_queue_factory -T local -M phyto_oracle-atmo -w 40 -W 90 --workers-per-cycle 10 --cores=1 -t 900
${HOME}/cctools-7.1.6-x86_64-centos7/bin/slurm_submit_workers -p "--account=<account> --partition=standard --nodes=1 --ntasks=94 --cpus-per-task=1" -N phyto_oracle-atmo 94
