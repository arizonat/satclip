#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpu_a100      # <- or cpu_amd
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bgtj-tgirails
#SBATCH --time=168:00:00      # hh:mm:ss for the job
##SBATCH --mail-user=leca5365@colorado.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options
#SBATCH --job-name=tsatclip-sweep
#SBATCH --array=0-8
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -x
export OMP_NUM_THREADS=16

cd /u/leca5365/Documents/satclip/satclip
source /u/leca5365/Documents/miniconda3/bin/activate satclip313

echo "job is starting on `hostname`"

# Setup hyperparameter file
num_samples=("100k" "150k" "200k")
tpes=("toy" "toy_norm_year" "norm_year")

touch experiments_params.txt
for s in "${num_samples[@]}"; do
    for t in "${tpes[@]}"; do
	echo "$s $t" >> experiments_params.txt
    done
done

# Run the SLURM jobs
# SLURM_ARRAY_TASK_ID is 0-indexed, sed is 1-indexed
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" experiments_params.txt)
read -r NUM_SAMPLE TPE <<< "$line"

echo "Task $SLURM_ARRAY_TASK_ID: samples=$NUM_SAMPLE tpe=$TPE"

srun -u python main.py --config "./configs/default.yaml" \
     --data.index_fn="index-balanced-$NUM_SAMPLE.csv" \
     --data.temporal_positional_encoding="$TPE" \
     --model.tpe_type="$TPE" \
     --trainer.logger="[                                                                                                                                                                                                                              {                                                                                                                                                                                                                                                   \"class_path\": \"lightning.pytorch.loggers.TensorBoardLogger\",                                                                                                                                                                                  \"init_args\": {                                                                                                                                                                                                                                    \"save_dir\": \"temporal_satclip_logs\",                                                                                                                                                                                                          \"name\": \"temporal_satclip-s2\",                                                                                                                                                                                                                \"version\": \"tsatclip-s2-$NUM_SAMPLE-$TPE\"                                                                                                                                                                                                   }                                                                                                                                                                                                                                               },                                                                                                                                                                                                                                                {                                                                                                                                                                                                                                                   \"class_path\": \"lightning.pytorch.loggers.WandbLogger\",                                                                                                                                                                                        \"init_args\": {                                                                                                                                                                                                                                    \"save_dir\": \"temporal_satclip_logs\",                                                                                                                                                                                                          \"project\": \"temporal_satclip-s2\",                                                                                                                                                                                                             \"name\": \"tsatclip-s2-$NUM_SAMPLE-$TPE\",                                                                                                                                                                                               
        \"version\": \"tsatclip-s2-$NUM_SAMPLE-$TPE\"                                                                                                                                                                                            
      }                                                                                                                                                                                                                                           
    }                                                                                                                                                                                                                                             
  ]"


     # --trainer.logger[0].class_path=lightning.pytorch.loggers.TensorBoardLogger \
     # --trainer.logger[0].init_args.save_dir="temporal_satclip_logs" \
     # --trainer.logger[0].init_args.name="temporal_satclip-s2" \
     # --trainer.logger[0].init_args.version="tsatclip-s2-$NUM_SAMPLE_$TPE" \
     # --trainer.logger[1].class_path=lightning.pytorch.loggers.WandbLogger \
     # --trainer.logger[1].init_args.save_dir="temporal_satclip_logs" \
     # --trainer.logger[1].init_args.project="temporal_satclip-s2" \
     # --trainer.logger[1].init_args.name="tsatclip-s2-$NUM_SAMPLE_$TPE" \
     # --trainer.logger[1].init_args.version="tsatclip-s2-$NUM_SAMPLE_$TPE_$SLURM_ARRAY_TASK_ID" \
