#!/bin/bash 

qsub -I -N phytooracle -W group_list=lyons-lab -q standard -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True -l walltime=6:0:0
