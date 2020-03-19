#!/bin/bash

./entrypoint.sh -c
rm -f makeflow.jx.args.*
rm -r bundle
rm -r cleanmetadata_out
rm -r soilmask_out
rm dall.log
