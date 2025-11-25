#!/bin/sh

export OMP_NUM_THREADS=4

echo "main dir in docker:"
<%text>
echo ${PWD}
</%text>
cd 01_sims/${'%s' % fname}


echo \$\{PWD\}
mkdir output
cp ${'%s' % fname}.swn INPUT

swan_omp
rm swaninit
rm norm_end

cd ../../

python3 02_scripts/analyse_simulation.py --sim_name ${'%s' % fname}


echo "Applying chmod 777 to simulation directory..."
chmod -R 777 01_sims/${'%s' % fname}
chmod -R 777 00_results/${'%s' % fname}


echo "Finished"





