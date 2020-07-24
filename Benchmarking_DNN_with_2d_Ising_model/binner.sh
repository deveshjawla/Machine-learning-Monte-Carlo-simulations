#!/bin/bash
#SBATCH -p bigmem
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=5:00
/home/jawla/julia-1.4.2/bin/julia /home/jawla/Simulations/deeplearning_gonihedric/to_bin.jl
