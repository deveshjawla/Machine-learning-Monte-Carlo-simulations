#!/bin/bash
#SBATCH -p batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000
#SBATCH -t 30:00:00

/home/jawla/julia-1.4.2/bin/julia /home/jawla/Simulations/1/data_maker.jl

