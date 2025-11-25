#!/bin/sh
#SBATCH --job-name=${'%s' % fname}		#Job name
#SBATCH --ntasks=4					# Number of processes
#SBATCH --partition=${'%s' % partition}
#SBATCH --nodes=1					# Maximum number of nodes to be allocated
#SBATCH --output=log.log
#SBATCH --time ${'%s' % runtime}


module load swan/${'%s' % swan_module}
swan_omp_exe=${'%s' % swan_exe}

export OMP_NUM_THREADS=4

echo ----------------------------------------------------------------------
echo Run of
echo $swan_omp_exe
echo with OpenMP on linux-cluster.
echo SGE_O_WORKDIR : $SGE_O_WORKDIR
echo HOSTNAME : $HOSTNAME
echo OMP_NUM_THREADS : $OMP_NUM_THREADS
echo ----------------------------------------------------------------------
mkdir output
cp "${'%s' % fname}".swn INPUT

$swan_omp_exe
rm swaninit
rm norm_end




