#!/bin/bash 
#62 or 250Gb 
qsub -I -N <name> -W group_list=<group> -q windfall -l select=1:ncpus=1:mem=250gb:pcmem=4gb -l walltime=2:0:0
