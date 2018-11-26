

#!/bin/csh

# simple template script for submission of a serial job on hamilton with SLURM # directives

# please make a copy of this file and edit the items included with the

# brackets < >

# in particular the name of the executable on the last line

#############################################################################

## All lines that start with #SBATCH will be processed by SLURM.

## Lines in example that start with white space between # and SBATCH will be

## ignored and to

## activate those directives edit this file and remove the white space

## name of the job

# SBATCH -J <job_name>

## By default SLURM uses as working directory the directory from where the

## job script is submitted. To change this the standard Linux cd command has

## to be used.

## By default SLURM combines the standard output and error streams in a single

## file with extension .out

## These streams can be directed to separate files with these two directives

# SBATCH -o <file name>

# SBATCH -e <file name>

## send email to the user when certain type of events occurr to the job

SBATCH --mail-type=<ALL>

## where <type> can be BEGIN, END, FAIL, REQUEUE and ALL

## to email address

SBATCH --mail-user wthw55@durham.ac.uk

## default email name is that of the submitting user as known to the system.

## specify project name (currently commented out)

## only edit this if you don't want your default project charged

##$ -A test

## The wall clock alotted to this job can be set with the -t option in the format

## -t hh:mm:ss for h(ours), m(inutes) and s(econds).

# SBATCH -t hh:mm:ss

#############################################################################

## The number of requested processes is specified with the -n option and

## is 1 for a serial job

SBATCH -n 1

## Request submission to a queue (partition) for serial jobs

SBATCH -p seq6.q

module purge

module load slurm/current

## Load any modules required here

module load python/2.7.13

## Execute command

srun ./spectroscopy_tests.py