import argparse

import time
start_time = time.time()

# ============ capture user input =============

parser = argparse.ArgumentParser(description='Train a SurvSurf model')
parser.add_argument(
    '-i', '--inputdir', 
    help='absolute path to the directory containing raw inputs.'
)

parser.add_argument(
    '-r', '--runtimedir', 
    help='absolute path to the directory for storing runtime data.'
)

parser.add_argument(
    '-o', '--outputdir', 
    help='absolute path to the directory for storing the output files.'
)

parser.add_argument(
    '--tune', 
    type=str,
    default='n',
    help='whether to tune the hyperparameters, default to False'
)

parser.add_argument(
    '--tunelr', 
    type=str,
    default='n',
    help='whether to tune the learning rate, default to False, only applicable when --tune False'
)

parser.add_argument(
    '--n_trials_tune', 
    type=int,
    default=128,
    help='number of trials to run'
)

parser.add_argument(
    '--n_epochs_tune', 
    type=int,
    default=100,
    help='number of trials to run'
)

parser.add_argument(
    '--n_epochs_train', 
    type=int,
    default=200,
    help='number of trials to run'
)

parser.add_argument(
    '--patience', 
    type=int,
    default=50,
    help='number of epochs with no improvement before early stopping'
)

parser.add_argument(
    '--weighted', 
    type=str,
    default='y',
    help='whether to weight the on-transition coordinates more'
)

parser.add_argument(
    '--use_gpu', 
    type=str,
    default='n',
    help='whether or not to use GPU for training'
)
args = parser.parse_args()



# %%
import pandas as pd

import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

# %%
import torch
torch.manual_seed(100)

import random
random.seed(15)

import numpy as np
np.random.seed(30)

# %%
with torch.no_grad():
    torch.cuda.empty_cache()

# %%
DIR_INPUTS = args.inputdir
DIR_RUNTIME_DATA = args.runtimedir
DIR_RUNTIME_RESULTS = args.outputdir
TUNE = True if args.tune == 'y' else False
TRANS_ONLY=False
TUNE_LR = True if args.tunelr == 'y' else False# only effective if TUNE is False
N_TRIALS_TUNE = args.n_trials_tune #100
MAX_EPOCH_TUNE = args.n_epochs_tune
MAX_EPOCH_TRAIN = args.n_epochs_train
PATIENCE = args.patience
WEIGHTED = True if args.weighted == 'y' else False
USE_GPU = True if args.use_gpu == 'y' else False
print(args._get_kwargs())

torch.set_float32_matmul_precision('medium')


# %%
# Clean up the runtime data folder
import shutil
if not os.path.isdir(DIR_RUNTIME_DATA):
    os.mkdir(DIR_RUNTIME_DATA)
    
if not os.path.isdir(DIR_RUNTIME_RESULTS):
    os.mkdir(DIR_RUNTIME_RESULTS)

# %%
# Clean up the runtime data folder
import shutil
if TUNE:
    for root, dirs, files in os.walk(DIR_RUNTIME_DATA):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    for root, dirs, files in os.walk(DIR_RUNTIME_RESULTS):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

# %% [markdown]
# ### Create train-validation and test sets (into csvs)

# %%
df_features = pd.read_csv(os.path.join(DIR_INPUTS,'df_features.csv'), index_col=0)
df_features.index = df_features.index.rename('subject')
df_features = df_features.reset_index()
df_state_history_sampled_max = pd.read_csv(os.path.join(DIR_INPUTS,'df_state_history_sampled_max.csv'),  index_col=0)


# %%
MAX_STATES = df_state_history_sampled_max['state'].max()
N_TIME = df_state_history_sampled_max['time'].max()

# %%
N_TIME

# %%
MAX_STATES

# %%
df_features.head()

# %%
df_state_history_sampled_max.head()

# %%
df_state_history_sampled_max.max()

# %%
subj_train = df_features['subject'][::df_features['subject'].size//1000]
subj_train_tune = subj_train.sample(n=500)
subj_val = df_features['subject'].loc[~df_features['subject'].isin(subj_train)].sample(n=500)
subj_test = df_features['subject'].loc[
    ~(
        df_features['subject'].isin(subj_train) |
        df_features['subject'].isin(subj_val)
    )
]


for suffix, subj in [
    ('train_tune', subj_train_tune),
    ('train', subj_train),
    ('val', subj_val),
    ('test', subj_test)
]:
    print(f'n_subj in {suffix}:{len(subj)}')
    df_features_sub = df_features.loc[
        df_features['subject'].isin(subj),
        :
    ]
    df_features_sub.to_csv(os.path.join(DIR_RUNTIME_DATA, f'df_features_{suffix}.csv'))

    df_state_history_sampled_max_sub = df_state_history_sampled_max.loc[
        df_state_history_sampled_max['subject'].isin(subj),
        :
    ]
    df_state_history_sampled_max_sub.to_csv(os.path.join(DIR_RUNTIME_DATA, f'df_state_history_sampled_max_{suffix}.csv'))

del df_state_history_sampled_max_sub
del df_features_sub
del df_state_history_sampled_max
del df_features

# %% [markdown]
# ## Load train val and test sets

# %%
from torch.utils.data import DataLoader
from monotonic_nn_surv_surf.utils.datasets_def import DatasetFeatANDtgy

# %%
ds_train = DatasetFeatANDtgy(
    path_feat_by_subj=os.path.join(DIR_RUNTIME_DATA,'df_features_train.csv'),
    path_state_history_max_grade=os.path.join(DIR_RUNTIME_DATA,'df_state_history_sampled_max_train.csv'),
    max_grade=MAX_STATES, 
    max_time=N_TIME,
    trans_only=TRANS_ONLY,
    weighted=WEIGHTED
)
loader_train = DataLoader(ds_train, batch_size=1000,shuffle=True)

ds_train_tune = DatasetFeatANDtgy(
    path_feat_by_subj=os.path.join(DIR_RUNTIME_DATA,'df_features_train_tune.csv'),
    path_state_history_max_grade=os.path.join(DIR_RUNTIME_DATA,'df_state_history_sampled_max_train_tune.csv'),
    max_grade=MAX_STATES, 
    max_time=N_TIME,
    trans_only=TRANS_ONLY,
    weighted=WEIGHTED
)
loader_train_tune = DataLoader(ds_train, batch_size=500,shuffle=True)

ds_val = DatasetFeatANDtgy(
    path_feat_by_subj=os.path.join(DIR_RUNTIME_DATA,'df_features_val.csv'),
    path_state_history_max_grade=os.path.join(DIR_RUNTIME_DATA,'df_state_history_sampled_max_val.csv'),
    max_grade=MAX_STATES, 
    max_time=N_TIME,
    trans_only=TRANS_ONLY,
    weighted=WEIGHTED
)
loader_val = DataLoader(ds_val, batch_size=1000)

ds_test = DatasetFeatANDtgy(
    path_feat_by_subj=os.path.join(DIR_RUNTIME_DATA,'df_features_test.csv'),
    path_state_history_max_grade=os.path.join(DIR_RUNTIME_DATA,'df_state_history_sampled_max_test.csv'),
    max_grade=MAX_STATES, 
    max_time=N_TIME,
    trans_only=TRANS_ONLY,
    weighted=WEIGHTED
)
loader_test = DataLoader(ds_test, batch_size=1000)
# %%
loader_train = DataLoader(ds_train, batch_size=1000,shuffle=True)
loader_train_tune = DataLoader(ds_train, batch_size=500,shuffle=True)

# %%
len(ds_train_tune)

# %%
len(ds_train)

# %%
len(ds_val)

# %% [markdown]
# ## Train model

# %%
if USE_GPU:
    device = 'gpu'
else:
    device = 'cpu'

# %%
import pytorch_lightning as pl

pl.seed_everything(seed=20)

# %%
from monotonic_nn_surv_surf.utils.pl_model_wrapper import LitSurvSurf
from monotonic_nn_surv_surf.utils.surv_surf_latent import SurvSurfLatent, LatentFeatFC

# %% [markdown]
# #### Hyperparam

# %%

def objective(trial):    
    n_monotone_layers = trial.suggest_int('n_monotone_layers', 4, 16)
    n_monoton_neurons = trial.suggest_int('n_monoton_neurons', 8, 64)
    n_feat_layers = trial.suggest_int('n_feat_layers', 4, 16)
    n_feat_neurons = trial.suggest_int('n_feat_neurons', 8, 64)
    p_dropout = trial.suggest_uniform('p_dropout', 0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    model = SurvSurfLatent(
        mono_net_sizes=[n_feat_neurons] + [n_monoton_neurons]*n_monotone_layers + [1],
        latent_feat_transformer=LatentFeatFC(
            input_size=3, 
            output_size=n_feat_neurons, 
            neurons_per_layer=(n_feat_layers-1)*[n_feat_neurons],
            dropout_p=p_dropout
        ),
    )
    
    model_lit = LitSurvSurf(model=model, lr=learning_rate)
     
    trainer = pl.Trainer(
        default_root_dir=DIR_RUNTIME_RESULTS,
        logger=False, 
        accelerator=device, 
        max_epochs=MAX_EPOCH_TUNE, 
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
    )      
    
    trainer.fit(
        model=model_lit, 
        train_dataloaders=loader_train_tune ,
        val_dataloaders=loader_val
    )
    
    score_val = trainer.test(model_lit, loader_val)[0]['test_loss']
    
    return score_val


# %%
import optuna
if TUNE:
    sampler = optuna.samplers.TPESampler(seed=17)

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS_TUNE)

# %% [markdown]
# ### read best hyperparams and tune lr if necessary

# %%
import json

if TUNE:
    print('best hyperparam and tuning performance:')
    print(study.best_trial.value)

    best_hyper_params = study.best_trial.params
    print(best_hyper_params)
    
    # Serializing json
    json_object = json.dumps(best_hyper_params, indent=4)
    
    # Writing to sample.json
    with open(os.path.join(DIR_RUNTIME_RESULTS,"best_hyper_params.json"), "w") as outfile:
        outfile.write(json_object)
else:
    # Opening JSON file
    with open(os.path.join(DIR_RUNTIME_RESULTS,"best_hyper_params.json"), 'r') as openfile:
    
        # Reading from json file
        best_hyper_params = json.load(openfile)

n_feat_neurons = best_hyper_params['n_feat_neurons']
n_monoton_neurons = best_hyper_params['n_monoton_neurons']
n_monotone_layers = best_hyper_params['n_monotone_layers']
n_feat_layers = best_hyper_params['n_feat_layers']
p_dropout = best_hyper_params['p_dropout']
learning_rate = best_hyper_params['learning_rate']


# %%
best_hyper_params

# %%
model = SurvSurfLatent(
    mono_net_sizes=[n_feat_neurons] + [n_monoton_neurons]*n_monotone_layers + [1],
    latent_feat_transformer=LatentFeatFC(
        input_size=3, 
        output_size=n_feat_neurons, 
        neurons_per_layer=(n_feat_layers-1)*[n_feat_neurons],
        dropout_p=p_dropout
    ),
)

model_lit = LitSurvSurf(model=model, lr=learning_rate, print_epoch=True)

# %%
logger = pl.loggers.CSVLogger(save_dir=DIR_RUNTIME_RESULTS)

if not TUNE:
    if TUNE_LR:
        trainer = pl.Trainer(
                accelerator=device, 
                default_root_dir=DIR_RUNTIME_RESULTS,
                enable_progress_bar=False,
                logger=logger
            )
        tuner = pl.tuner.Tuner(trainer)

        # 3. Tune learning rate
        lr_finder = tuner.lr_find(
                model_lit, 
                train_dataloaders=loader_train,
                val_dataloaders=loader_val,
        )

        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model_lit.hparams.lr = new_lr
        print(f'found suggested lr at {new_lr}')

# %% [markdown]
# ### actual train

# %%
early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
chkpt_min_val_loss = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)

trainer = pl.Trainer(
    accelerator=device, 
    max_epochs=MAX_EPOCH_TRAIN, 
    logger=logger,
    enable_progress_bar=False,
    default_root_dir=DIR_RUNTIME_RESULTS,
    check_val_every_n_epoch=1,
    callbacks=[early_stop, chkpt_min_val_loss]
)
trainer.fit(
    model=model_lit, 
    train_dataloaders=loader_train ,
    val_dataloaders=loader_val
)

# %%
trainer.test(model=model_lit, dataloaders=loader_train)

# %%
trainer.test(model=model_lit, dataloaders=loader_val)

# %%
trainer.test(model=model_lit, dataloaders=loader_test)

# %% [markdown]
# ## Inspect learning curve

# %%
dir_logs = os.path.join(DIR_RUNTIME_RESULTS, 'lightning_logs')
dir_logs

# %%
latest_ver = sorted(os.listdir(dir_logs))[-1]
latest_ver

# %%
epoch_metrics = pd.read_csv(os.path.join(dir_logs, f'{latest_ver}/metrics.csv'))
epoch_metrics.head(20)

# %%
epoch_metrics = epoch_metrics.groupby('epoch').apply(
    lambda df: 
    pd.Series(
        [
            df['val_loss'].iloc[0],
            df['train_loss'].iloc[-1]
        ],
        index=['val_loss','train_loss']
    ), 
).reset_index()

# %%
epoch_metrics['val_loss'].min()

# %%
fig, ax = plt.subplots(1,1)
ax.plot(
    epoch_metrics['epoch'],
    epoch_metrics['train_loss'],
    label='train_loss'
)
ax.plot(
    epoch_metrics['epoch'],
    epoch_metrics['val_loss'],
    label='val_loss'
)
ax.set(xlabel='epochs', ylabel='loss')
ax.legend()
fig.savefig(os.path.join(DIR_RUNTIME_RESULTS,'./learning_curve.pdf'))


# %%
with torch.no_grad():
    torch.cuda.empty_cache()

time_taken_s = int(time.time() - start_time)
hr = time_taken_s//(3600)
hr_in_s = hr*3600
minutes = (time_taken_s - hr_in_s)//60
min_in_s = minutes*60
seconds = time_taken_s - hr_in_s - min_in_s

print('='*10 + 'END' + '='*10)
print(f'Time taken to run full script: {str(hr).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}')


