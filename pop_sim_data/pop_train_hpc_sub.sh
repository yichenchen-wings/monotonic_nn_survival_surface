#!/bin/bash
#SBATCH -J survsurf_one_surf_each_sigmoid_wide_hpc_tune
#SBATCH --output=/home/yc366/rds/hpc-work/hpc_out/survsurf_one_surf_each_sigmoid_wide_hpc_tune/%a.out
#SBATCH --error=/home/yc366/rds/hpc-work/hpc_out/survsurf_one_surf_each_sigmoid_wide_hpc_tune/%a.err
#SBATCH -A SCHONLIEB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mail-type=ALL 
#SBATCH --array=1

## Comment out the block below for local run
# module load rhel8/default-amp  
# source activate env_notebook_train_test_survsurf


## Uncomment and edit the block below for hpc run
# INPUT_DIR='/home/yc366/rds/hpc-work/data_markov_one_surf_each/'
# RUNTIME_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}_runtime/"
# OUTPUT_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}_results/"

## Uncomment and edit the block below for local run
INPUT_DIR="/home/yc366/repos/monotonic_nn_survival_surface/pop_sim_data/data_raw/"
RUNTIME_DIR="/home/yc366/repos/monotonic_nn_survival_surface/pop_sim_data/data_runtime/"
OUTPUT_DIR="/home/yc366/repos/monotonic_nn_survival_surface/pop_sim_data/results_runtime/"

USE_GPU=n
TUNE=y
TUNE_LR=n
N_TRIALS_TUNE=64
N_EPOCHS_TUNE=100
N_EPOCHS_TRAIN=1000
PATIENCE=100
WEIGHTED=y

# Launch script using our defined variables
python ./pop_train_script.py  \
--inputdir $INPUT_DIR \
--runtimedir $RUNTIME_DIR \
--outputdir $OUTPUT_DIR \
--tune $TUNE \
--tunelr $TUNE_LR \
--n_trials_tune $N_TRIALS_TUNE \
--n_epochs_tune $N_EPOCHS_TUNE \
--n_epochs_train $N_EPOCHS_TRAIN \
--patience $PATIENCE \
--weighted $WEIGHTED \
--use_gpu $USE_GPU

echo "Finished array: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"