#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:45:38 2022

@author: brede
"""
import pickle
import os
import numpy as np
import pandas as pd


def import_n_summarise(folder):
    path = os.path.join('res/', folder)
    fs = os.listdir(path)
    best_runs = {}
    all_runs = {'file': [],
                'time': [],
                'loss': []}
    for f in fs:
        with open(os.path.join(path, f), 'rb') as file:
            res = pickle.load(file)
        test = res.get_all_runs()
        losses = [r['loss'] if r['loss']is not None else 5 for r in test]
        deltas = [r['time_stamps']['finished'] - r['time_stamps']['started'] for r in test]
        best_runs[f] = test[np.argmin(losses)]['info'][1]
        best_runs[f]['file'] = f
        best_runs[f]['runs'] = len(losses)
        best_runs[f]['loss'] = np.min(losses)
        best_runs[f]['m_time'] = np.mean(deltas)
        best_runs[f]['sd_time'] = np.std(deltas)
        all_runs['file'] += list(np.repeat(f, len(losses)))
        all_runs['time'] += deltas
        all_runs['loss'] += losses
    best_runs = pd.DataFrame(best_runs.values())
    best_runs.to_csv(f'../work/thesis/data/{folder}.csv', index=False)
    all_runs = pd.DataFrame(all_runs)
    all_runs.to_csv(f'../work/thesis/data/{folder}_all_runs.csv', index=False)

if __name__ == '__main__':
    folders = ['dbscan_100','dbscan_500', 'slic_100', 'slic_500','ncuts_100']
    for folder in folders:
        import_n_summarise(folder)
