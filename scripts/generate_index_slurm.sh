#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p RM-shared
#SBATCH -t 24:00:00

set -x

cd /ocean/projects/cis250170p/lcai5/Documents/satclip/scripts

source /ocean/projects/cis250170p/lcai5/miniconda3/bin/activate satclip313

python generate_index_local.py /ocean/projects/cis260118p/shared/satclip-s2-1M-2.0/

