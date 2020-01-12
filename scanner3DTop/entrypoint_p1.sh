#!/bin/bash

jx2json main_workflow_phase1.jx -a bundle_list.json > main_workflow_phase1.json
makeflow -T wq --json main_workflow_phase1.json -a -N phyto_oracle-atmo -p 9123 -dall -o dall.log $@

