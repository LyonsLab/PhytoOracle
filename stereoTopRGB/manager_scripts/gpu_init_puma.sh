#!/bin/bash -l 

srun --nodes=1 --mem=470GB --ntasks=1 --cpus-per-task=94 --time=24:00:00 --job-name=po_mcomp --account=dukepauli --partition=standard --mpi=pmi2 --pty bash -i
