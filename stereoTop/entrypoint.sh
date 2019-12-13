#!/bin/bash


# -a advertise to catalog server
makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log $@

