#!/bin/bash
#SBATCH -J survsurf_one_surf_each_sigmoid_wide_hpc_tune_wbias
#SBATCH --output=/home/yc366/rds/hpc-work/hpc_out/survsurf_one_surf_each_sigmoid_wide_hpc_tune_wbias/%a.out
#SBATCH --error=/home/yc366/rds/hpc-work/hpc_out/survsurf_one_surf_each_sigmoid_wide_hpc_tune_wbias/%a.err
#SBATCH -A SCHONLIEB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL 
#SBATCH --array=1

module load rhel8/default-amp  
source activate env_notebook_train_test_survsurf

INPUT_DIR='/home/yc366/rds/hpc-work/data_markov_one_surf_each_wbias/'
RUNTIME_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}_runtime/"
OUTPUT_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}_results/"
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
--weighted $WEIGHTED

echo "Finished array: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"