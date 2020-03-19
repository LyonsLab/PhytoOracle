#!/bin/bash


sudo ps -A | grep work_queue | awk '{$1=$1}1' | cut -d' ' -f1 | xargs -I {} sudo kill {}
rm -rf /scratch/worker-*



