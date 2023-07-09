#!/bin/bash
#SBATCH -J survsurf_one_surf_each_sigmoid_one_gpu_tune
#SBATCH --output=/home/yc366/rds/hpc-work/hpc_out/survsurf_one_surf_each_sigmoid_one_gpu_tune_%a.out
#SBATCH --error=/home/yc366/rds/hpc-work/hpc_out/survsurf_one_surf_each_sigmoid_one_gpu_tune_%a.err
#SBATCH -A SCHONLIEB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL 
#SBATCH --array=2

module load rhel8/default-amp  
source activate env_notebook_train_test_survsurf

INPUT_DIR='/home/yc366/rds/hpc-work/data_markov_one_surf_each/'
RUNTIME_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_runtime/"
OUTPUT_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_results/"
BEST_HYPER_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_results/"
N_GPU=1
TUNE=y
TUNE_LR=n
N_TRIALS_TUNE=64
N_EPOCHS_TUNE=100
N_EPOCHS_TRAIN=200

# Launch script using our defined variables
python ./pop_train_script.py  --inputdir $INPUT_DIR --runtimedir $RUNTIME_DIR --outputdir $OUTPUT_DIR --best_hyper_dir $BEST_HYPER_DIR --num_gpu $N_GPU --tune $TUNE --tunelr $TUNE_LR --n_trials_tune $N_TRIALS_TUNE --n_epochs_tune $N_EPOCHS_TUNE --n_epochs_train $N_EPOCHS_TRAIN

echo "Finished array: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"