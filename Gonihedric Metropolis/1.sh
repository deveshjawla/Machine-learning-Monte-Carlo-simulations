#!/bin/bash
#SBATCH -p batch
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=2000
#SBATCH -t 48:00:00

/home/jawla/julia-1.4.2/bin/julia /home/jawla/Simulations/1/1.jl

