#!/bin/sh

#SBATCH -A ecode

## Email settings
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=foreba10@msu.edu

## Firts arg is name of experiment

## Job name settings
#SBATCH --job-name=WerryBorld

## Time requirement in format "days-hours:minutes"
#SBATCH --time=6-0:00

## Memory requirement in megabytes
#SBATCH --mem-per-cpu=5120                                                 

module load GNU/8.2.0-2.31.1

cd /mnt/scratch/$USER/WerryBorld

## Make a directory for this condition
mkdir $1
cd $1

# Get files
cp /mnt/home/$USER/WerryBorld/*.py .

# Make a new results directory for this run
mkdir results

# Run script
python3 run.py > outfile.txt
