#!/bin/bash

${HOME}/cctools-7.1.12-x86_64-centos7/bin/jx2json main_workflow_phase-2.jx -a bundle_list.json > main_workflow_phase2.json


# -a advertise to catalog server
${HOME}/cctools-7.1.12-x86_64-centos7/bin/makeflow -T wq --json main_workflow_phase2.json -a -r 2 -M phytooracle_3d -N phytooracle_3d -p 60221 -dall -o dall.log --disable-cache $@
