#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpu_a100      # <- or cpu_amd, gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bgtj-tgirails
#SBATCH --time=12:00:00      # hh:mm:ss for the job
##SBATCH --mail-user=leca5365@colorado.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options
#SBATCH --job-name=s2-2m-embeddings

set -x
export OMP_NUM_THREADS=8

cd /u/leca5365/Documents/satclip/scripts
source /u/leca5365/Documents/miniconda3/bin/activate satclip313


echo "job is starting on `hostname`"

# python precompute_embeddings.py /projects/bgtj/t_satclip/Data/landsat45-1M/
# python generate_index_local.py /projects/bgtj/t_satclip/Data/sentinel2-2M sentinel2
python precompute_embeddings.py sentinel2 /projects/bgtj/t_satclip/Data/sentinel2-2M/
