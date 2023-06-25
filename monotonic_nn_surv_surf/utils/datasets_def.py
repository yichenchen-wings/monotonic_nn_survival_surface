
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class DatasetFeatANDtgy(Dataset):
    def __init__(self, path_feat_by_subj, path_state_history_max_grade, max_grade, max_time, trans_only):
        self.max_grade = max_grade
        self.max_time = max_time
        self.trans_only = trans_only


        df_features = pd.read_csv(path_feat_by_subj, index_col=0)
        assert 'subject' in df_features.columns
        cols_feats = df_features.columns[df_features.columns != 'subject']

        df_state_history_max_grade = pd.read_csv(path_state_history_max_grade,  index_col=0)
       
        df_X_y_ready = self._get_X_y_ready(df_features=df_features,df_state_history_max_grade=df_state_history_max_grade)

        self.observed = df_X_y_ready

        if self.trans_only:
            balance_by = ['g','t']
        else:
            balance_by = 'trans_ref'
        weights = self.observed.groupby(balance_by).apply(lambda x: x.shape[0]).rename('weight')
        self.observed = self.observed.merge(
            right=weights.reset_index(),
            how='left',
            on=balance_by
        )
        self.observed['weight'] = 1-self.observed['weight']/self.observed['weight'].sum()
        self.observed['weight'] = self.observed['weight']/self.observed['weight'].sum() * self.observed['weight'].size

        self.X = torch.tensor(self.observed[cols_feats].values, dtype=torch.float32)
        self.t = torch.tensor(self.observed[['t']].values/max_time, dtype=torch.float32)
        self.g = torch.tensor(self.observed[['g']].values/max_grade, dtype=torch.float32)
        self.y = torch.tensor(self.observed[['y']].values, dtype=torch.float32)
        self.weight = torch.tensor(self.observed[['weight']].values, dtype=torch.float32)
        
    def _obs_to_labels_all_tg(self, df_subj_max_grade_traj):
        df_sorted = df_subj_max_grade_traj.sort_values('time')
        ts_raw = df_sorted['time'].values
        gs_raw = df_sorted['state_max_by_time'].values

        if ts_raw[0] != 0:
            ts_raw = np.r_[[0], ts_raw] 
            gs_raw = np.r_[[0], gs_raw]

        traj = pd.Series(gs_raw, index=ts_raw)

        rows = []
        for t in traj.index:
            g_obs = traj[t]
            for g in range(1, self.max_grade+1):    
                if g in [g_obs, g_obs+1]:
                    trans_ref = 'at_trans'
                else:
                    trans_ref = 'off_trans'
                if g > g_obs:
                    rows.append(
                        {
                            't':t,
                            'g':g,
                            'y':0,
                            'trans_ref':trans_ref
                        }
                    )
                else:
                    rows.append(
                        {
                            't':t,
                            'g':g,
                            'y':1,
                            'trans_ref':trans_ref
                        }
                    )
        return pd.DataFrame(rows)
    
    def _obs_to_labels_trans_tg(self, df_subj_max_grade_traj):
        df_sorted = df_subj_max_grade_traj.sort_values('time')
        ts_raw = df_sorted['time'].values
        gs_raw = df_sorted['state_max_by_time'].values

        if ts_raw[0] != 0:
            ts_raw = np.r_[[0], ts_raw] 
            gs_raw = np.r_[[0], gs_raw]

        traj = pd.Series(gs_raw, index=ts_raw)

        rows = []
        for t in traj.index:
            g_obs = traj[t]
            for g in [g_obs, g_obs+1]:
                if (g == 0) or (g > self.max_grade):
                    continue
                if g > g_obs:
                    rows.append(
                        {
                            't':t,
                            'g':g,
                            'y':0,
                            'trans_ref':'not_possible'
                        }
                    )
                else:
                    rows.append(
                        {
                            't':t,
                            'g':g,
                            'y':1,
                            'trans_ref':'happened'
                        }
                    )
        return pd.DataFrame(rows)
    
    def _get_y(self, df_X_y):
        if self.trans_only:
            df_y = df_X_y.groupby('subject').apply(self._obs_to_labels_trans_tg)
        else:
            df_y = df_X_y.groupby('subject').apply(self._obs_to_labels_all_tg)
        return df_y
    
    def _get_X_y_ready(self, df_features, df_state_history_max_grade):
        assert 'subject' in df_features.columns
        assert 'subject' in df_state_history_max_grade.columns
        
        assert 'time' in df_state_history_max_grade.columns
        assert 'state_max_by_time' in df_state_history_max_grade.columns

        df_X_y = df_features.merge(df_state_history_max_grade, how='left', on='subject')
        df_y = self._get_y(df_X_y).reset_index()

        df_X_y_ready = df_features.merge(
            df_y,
            how='left',
            on=['subject']
        )
        return df_X_y_ready
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.t[index], self.g[index], self.y[index], self.weight[index]
