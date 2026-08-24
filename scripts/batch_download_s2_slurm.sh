#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --job-name=s2-data-download
#SBATCH --time=168:00:00      # hh:mm:ss for the job
##SBATCH --mail-user=leca5365@colorado.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options


set -x
export OMP_NUM_THREADS=96

cd /u/leca5365/Documents/satclip/scripts
source /u/leca5365/Documents/miniconda3/bin/activate satclip313


echo "job is starting on `hostname`"

export OUTPUT_DIR=/u/leca5365/Data/s2-2M

srun -u python download_s2_batch_local.py --low 0 --high 2000000 --img_output_dir $OUTPUT_DIR/images --output_fn $OUTPUT_DIR/s2_timestamp_2M_metadata.csv --s2_parquet_fn /u/leca5365/Data/s2-parquets/s2l2a_clouds_lt_2020_12_31_gt_2015_07_04.parquet  --batch_size 96 --num_workers 96

