#!/bin/bash 

qsub -I -N phytooracle -W group_list=lyons-lab -q standard -l select=1:ncpus=16:mem=62gb:pcmem=4gb -l walltime=24:0:0
