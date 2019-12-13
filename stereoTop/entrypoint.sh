#!/bin/bash


# -a advertise to catalog server
# -J max 200 remote job
makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle-atmo -p 9123 -J 200 -dall -o dall.log $@

