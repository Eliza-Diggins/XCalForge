#!/bin/bash
#SBATCH --job-name=arf_libgen
#SBATCH --output=logs/arf_libgen_%j.out
#SBATCH --error=logs/arf_libgen_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=5
#SBATCH --ntasks=25
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eliza.diggins@utah.edu
#SBATCH --partition=notchpeak
#SBATCH --account=wik

# ----------------------------- #
# Load environment/modules      #
# ----------------------------- #

# Load modules or activate conda
module purge
module load heasoft/6.31.1
module load openmpi/4.1.3

# If using a conda environment:
source /uufs/astro.utah.edu/common/home/u1281896/xrmodifier/.venv/bin/activate

# ----------------------------- #
# Run the MPI-based script      #
# ----------------------------- #

mpirun -np ${SLURM_NTASKS} python /uufs/astro.utah.edu/common/home/u1281896/xrmodifier/test.py
