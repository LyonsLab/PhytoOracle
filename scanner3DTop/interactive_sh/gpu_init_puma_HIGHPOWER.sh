#!/bin/bash -l 

#qsub -I -N phytooracle -W group_list=lyons-lab -q standard -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True -l walltime=20:0:0

#qsub -I -N phytooracle -W group_list=lyons-lab -q standard -l select=1:ncpus=96:mem=512gb -l walltime=20:0:0
# mem: either 470 or 3000

srun --nodes=1 --mem=512GB --ntasks=1 --cpus-per-task=94 --time=24:00:00 --job-name=po_mcomp --account=lyons-lab --partition=standard --mpi=pmi2 --pty bash -i
