#!/bin/bash
#SBATCH -J paired_g_inside_dy_loss_wbias
#SBATCH --output=/home/yc366/rds/hpc-work/hpc_out/paired_g_inside_dy_loss_wbias/%a.out
#SBATCH --error=/home/yc366/rds/hpc-work/hpc_out/paired_g_inside_dy_loss_wbias/%a.err
#SBATCH -A SCHONLIEB-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes 1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL 
#SBATCH --array=1

## Comment out the block below for local run
#module load rhel8/default-amp  
module load rhel8/default-icl
source activate env_notebook_train_test_survsurf

## Uncomment and edit the block below for hpc run
INPUT_DIR='/home/yc366/rds/hpc-work/data_markov_one_surf_each_wbias/'
RUNTIME_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}_data_runtime/"
OUTPUT_DIR="/home/yc366/rds/hpc-work/hpc_out/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}_train_results/"

## Uncomment and edit the block below for local run
# INPUT_DIR="/home/yc366/repos/monotonic_nn_survival_surface/pop_sim_data/data_raw/"
# RUNTIME_DIR="/home/yc366/repos/monotonic_nn_survival_surface/pop_sim_data/data_runtime/"
# OUTPUT_DIR="/home/yc366/repos/monotonic_nn_survival_surface/pop_sim_data/results_runtime/"

USE_GPU=n
TRANS_ONLY=y
TUNE=n
N_TRIALS_TUNE=32
BATCH_SIZE=100
N_EPOCHS_TUNE=200
N_EPOCHS_TRAIN=1600
PATIENCE=200
WEIGHTED=n

if [ "$TUNE" = "y" ]; then
    # Launch script using our defined variables
    python ./pop_train_script.py  \
    --inputdir $INPUT_DIR \
    --runtimedir $RUNTIME_DIR \
    --outputdir $OUTPUT_DIR \
    --trans_only $TRANS_ONLY \
    --tune $TUNE \
    --n_trials_tune $N_TRIALS_TUNE \
    --batch_size $BATCH_SIZE \
    --n_epochs_tune $N_EPOCHS_TUNE \
    --n_epochs_train $N_EPOCHS_TRAIN \
    --patience $PATIENCE \
    --weighted $WEIGHTED \
    --use_gpu $USE_GPU
else
    for i in {20..100..20} # run training script with 5 diff seeds
    do 
        python ./pop_train_script.py  \
        --inputdir $INPUT_DIR \
        --runtimedir $RUNTIME_DIR \
        --outputdir $OUTPUT_DIR \
        --trans_only $TRANS_ONLY \
        --tune $TUNE \
        --n_trials_tune $N_TRIALS_TUNE \
        --batch_size $BATCH_SIZE \
        --n_epochs_tune $N_EPOCHS_TUNE \
        --n_epochs_train $N_EPOCHS_TRAIN \
        --patience $PATIENCE \
        --weighted $WEIGHTED \
        --use_gpu $USE_GPU \
        --model_init_seed_train $i
    done

fi

echo "Finished array: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"