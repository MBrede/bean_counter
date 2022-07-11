#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:55:26 2022

@author: brede
"""
import keras
import pickle
import os
import numpy as np
import cv2
import regex as re
from random import shuffle
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
import ConfigSpace as CS
from hpbandster.core.worker import Worker

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import BOHB as BOHB


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


def parsenumkey(key, path_to_image):
    """Extract meta-info from `.tif`-file.

    Args:
        key (str): Label of meta-information to extract.
        fname (str): Path to file.

    Returns:
        float: Extracted meta-information.

    """
    with open(path_to_image, "rb") as f:
        for line in f:
            match = re.search(b"".join([b"^", key, b"=([-.0-9e]*)"]), line)
            if match is not None:
                break
    return float(match.group(1))

def import_image_wo_databar(path, compression = 4):
    image = cv2.imread(path, 0)
    dbheight = int(parsenumkey(b"DatabarHeight", path))
    image = image[: len(image) - dbheight]
    for i in range(compression):
        image = cv2.pyrDown(image)
    return(image)

class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


        Y = []
        X = []
        for k in best_runs:
            dummy = best_runs[k]
            image = import_image_wo_databar(os.path.join('../data/',dummy['file'][:-9] + '.tif'))
            dummy['image'] = image
            for i in range(4):
                if i % 2 == 0:
                    dummy['image'] = np.flip(dummy['image'], axis = 0)
                if i < 2:
                    dummy['image'] = np.flip(dummy['image'], axis = 1)
                X.append(dummy['image'].flatten()/255)
                Y.append([dummy[tk] for tk in dummy if tk not in ['file', 'image', 'target']])
        index = list(range(len(Y)))
        shuffle(index)
        Y = np.array(Y).T
        Y = [(np.mean(y) - y )/ np.sqrt(np.var(y)) for y in Y]
        self.Y = [list(np.array(Y).T[i]) for i in index]
        self.X = [X[i] for i in index]

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        hidden = CS.UniformIntegerHyperparameter('hiddenLayer', lower=0, upper=4)
        batchSize = CS.UniformIntegerHyperparameter('batchSize', lower=1, upper=5)
        cs.add_hyperparameters([hidden, batchSize])
        condtions = []
        depth1 = CS.UniformIntegerHyperparameter('layer1Depth', lower=1, upper=10)
        acti1 = CS.CategoricalHyperparameter('layer1Acti', choices = ['relu', 'sigmoid', 'softmax','tanh'])
        condtions.append(CS.GreaterThanCondition(depth1, hidden, 0))
        condtions.append(CS.GreaterThanCondition(acti1, hidden, 0))
        depth2 = CS.UniformIntegerHyperparameter('layer2Depth', lower=1, upper=10)
        acti2 = CS.CategoricalHyperparameter('layer2Acti', choices = ['relu', 'sigmoid', 'softmax','tanh'])
        condtions.append(CS.GreaterThanCondition(depth2, hidden, 1))
        condtions.append(CS.GreaterThanCondition(acti2, hidden, 1))
        depth3 = CS.UniformIntegerHyperparameter('layer3Depth', lower=1, upper=10)
        acti3 = CS.CategoricalHyperparameter('layer3Acti', choices = ['relu', 'sigmoid', 'softmax','tanh'])
        condtions.append(CS.GreaterThanCondition(depth3, hidden, 2))
        condtions.append(CS.GreaterThanCondition(acti3, hidden, 2))
        depth4 = CS.UniformIntegerHyperparameter('layer4Depth', lower=1, upper=10)
        acti4 = CS.CategoricalHyperparameter('layer4Acti', choices = ['relu', 'sigmoid', 'softmax','tanh'])
        condtions.append(CS.GreaterThanCondition(depth4, hidden, 3))
        condtions.append(CS.GreaterThanCondition(acti4, hidden, 3))
        cs.add_hyperparameters([depth1, depth2, depth3, depth4, acti1, acti2, acti3, acti4])
        cs.add_conditions(condtions)
        return(cs)


    def compute(self, config, budget, **kwargs):
        keras.backend.clear_session()
        layer=[{'n': config[f'layer{i}Depth'], 'activation': config[f'layer{i}Acti']} for i in range(1, config['hiddenLayer']+1)]
        model = baseline_model(self.X, self.Y, layer)
        estimators = []
        estimators.append(('mlp', KerasRegressor(model=model, epochs=int(round(budget)), batch_size=config['batchSize'], verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=5)
        results = cross_val_score(pipeline, self.X, self.Y, cv=kfold, scoring='r2')
        loss = 1 - float(np.mean(results))
        info = [config, list(results)]
        return({
                    'loss': loss,  # this is the a mandatory field to run hyperband
                    'info': info  # can be used for any user-defined information - also mandatory
                })




def baseline_model(X, Y, deep_layer = []):
    model = keras.Sequential()
    model.add(keras.layers.Dense(X[0].flatten().shape[0], 
                          input_shape = X[0].flatten().shape,
                          kernel_initializer='normal', activation='relu'))
    for layer in deep_layer:
        model.add(keras.layers.Dense(2**layer['n'],
                              kernel_initializer='normal', 
                              activation=layer['activation']))
    model.add(keras.layers.Dense(len(Y[0]), kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model




if __name__ == '__main__':
    NS = hpns.NameServer(run_id='reg_net1', host='127.0.0.1', port=None)
    NS.start()

    w = MyWorker(nameserver='127.0.0.1', run_id='reg_net1')
    w.run(background=True)

    bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = 'reg_net1', nameserver='127.0.0.1',
                  min_budget=10, max_budget=300
               )
    res = bohb.run(n_iterations=50)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    with open('res/reg_net.pkl', 'wb') as f:
        pickle.dump(res, f)
