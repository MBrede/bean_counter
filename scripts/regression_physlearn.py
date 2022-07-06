#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:29:05 2022

@author: brede
"""
import pickle
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from physlearn import Regressor

def build_prep_stack(config):
    prep_stack = ['', '', '', '']
    if config['hist.pos']:
        prep_stack[config['hist.pos']-1] = 'prep.hist.()'

    if config['nlmeans.pos']:
        if prep_stack[config['nlmeans.pos']-1]:
            config['nlmeans.pos'] = [i for i,e in enumerate(prep_stack) if not e][0]+1
        prep_stack[config['nlmeans.pos']-1] = f'prep.nlMeans.(h={config["nlmeans.h"]}, temp_winSize={config["nlmeans.temp"]}, search_winSize={config["nlmeans.search"]})'

    if config['gauss.pos']:
        if prep_stack[config['gauss.pos']-1]:
            config['gauss.pos'] = [i for i,e in enumerate(prep_stack) if not e][0]+1
        prep_stack[config['gauss.pos']-1] = f'prep.gauss.(k_size=({config["gauss.k"]}, {config["gauss.k"]}))'

    if config['sav.pos']:
        if prep_stack[config['sav.pos']-1]:
            config['sav.pos'] = [i for i,e in enumerate(prep_stack) if not e][0]+1
        prep_stack[config['sav.pos']-1] = f'prep.sav.(w_size={config["sav.window"]}, grad={config["sav.poly"]})'

    prep_stack = [p for p in prep_stack if p]
    return prep_stack


def build_clust_settings(config):
    poss_cluster = f'analyse.SLIC.(compactness={config["SLIC.compactness"]}, n_segments={config["SLIC.n_segments"]})'
    return poss_cluster


path = 'res/slic_500/'
fs = os.listdir(path)
best_runs = {}
for f in fs:
    with open(os.path.join(path, f), 'rb') as file:
        res = pickle.load(file)
    test = res.get_all_runs()
    losses = [r['loss'] if r['loss']is not None else 5 for r in test]
    best_runs[f] = test[np.argmin(losses)]['info'][1]
    best_runs[f]['file'] = f
    best_runs[f]['loss'] = np.min(losses)


y = []
X = []
index = []
for k in best_runs:
    dummy = best_runs[k]
    image = cv2.imread(os.path.join('../data/', dummy['file'][:-9] + '.tif'), 0)
    for i in range(5):
        image = cv2.pyrDown(image)
    dummy['image'] = image
    for i in range(4):
        if i % 2 == 0:
            dummy['image'] = np.flip(dummy['image'], axis = 0)
        if i < 2:
            dummy['image'] = np.flip(dummy['image'], axis = 1)
        X.append(dummy['image'].flatten())
        y.append(pd.DataFrame({tk: [dummy[tk]] for tk in dummy if tk not in ['file', 'image', 'target']}))
        index.append(dummy['file'][:-9] + '_' + str(i))

X = pd.DataFrame(np.array(X))
X.index = index
y = pd.concat(y)
y.index = index
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
regressor_choice = 'HistGradientBoostingRegressor'
pipeline_transform = 'quantilenormal'
reg = Regressor(regressor_choice=regressor_choice, pipeline_transform=pipeline_transform)
y_pred = reg.fit(X_train, y_train).predict(X_test)

score = reg.score(y_test, y_pred)

print(score)
