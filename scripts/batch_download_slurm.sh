#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 72:00:00
#SBATCH -n 128

set -x

cd /ocean/projects/cis250170p/lcai5/Documents/satclip/scripts

source /ocean/projects/cis250170p/lcai5/miniconda3/bin/activate satclip313

python download_s2_batch_local.py --low 0 --high 1300000 --output_fn s2_timestamp_1M_metadata.csv --s2_parquet_fn s2l2a_2_5_2024_clouds_lt_20_date_gt_2021_01_01.parquet --batch_size 128 --num_workers 128 --img_output_dir /ocean/projects/cis260118p/shared/satclip-s2-1M-2.0
