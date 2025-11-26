#!/bin/sh
#SBATCH --job-name=${'%s' % fname}		#Job name
#SBATCH --ntasks=4					# Number of processes
#SBATCH --partition=${'%s' % partition}
#SBATCH --nodes=1					# Maximum number of nodes to be allocated
#SBATCH --output=log.log
#SBATCH --time ${'%s' % runtime}

## go to folder of docker
cd ../../03_docker/
## load docker
docker load < delft3dfm_py.tar

## show all loaded dockers
docker image ls

## go to main directory (this directory is mount)
cd ..

## set docker and run command
image=delft3dfm_py:2026.01-release
command=01_sims/${'%s' % fname}/run_swan.sh

<%text>
model_dir=${PWD}
</%text>
work_dir=.

echo "Model and work dir:"
echo $model_dir
echo $work_dir
<%text>
docker run \
    --rm \
    --mount "type=bind,source=${model_dir},target=/data" \
    --workdir "/data/${work_dir}" \
    $image \
    $command
</%text>