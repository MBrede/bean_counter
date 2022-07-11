#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: brede
"""
import _preprocess
import _analysis
from model import GrainImage
import os
import inspect
import itertools
import random
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime


def build_prep_possibilities(precs):
    poss_settings = {}
    for fun in precs:
        label = str(fun).split('_')[2]
        poss_settings[label] = []
        args = dict(inspect.signature(fun)._parameters)
        args = {a: str(args[a]).split('=')[1] for a in args if a != 'self'}
        for k in args:
            if len(args[k]) < 3:
                args[k] = range(1, int(args[k]) * 2 + 1)
            else:
                args[k] = [[str((i, i))] for i in
                           range(1, int(args[k][1:-1].split(', ')[0]) * 2 + 1)]
        combs = [p[0] if len(p) == 1 else p
                 for p in itertools.product(*list(args.values()))]
        for comb in combs:
            dummy = ', '.join(['%s=%s' % (k, comb[i])
                               for i, k in enumerate(args.keys())])
            poss_settings[label].append('prep.%s.(%s)' % (label, dummy))
    return poss_settings

def build_clust_settings():
    poss_cluster = {"eps": np.round(np.linspace(0.1, 3, 30),1),
                    "min_samples": np.round(np.linspace(2,30,15)),
                    "color_weight": np.round(np.linspace(0.1, 3, 30),1)}
    poss_cluster = ['analyse.dbscan.(eps=%s, min_samples=%d, color_weight=%s)' % p
                    for p in list(itertools.product(*list(poss_cluster.values())))]
    return poss_cluster



def main(n_tests = 5):
    results = []
    path = '../data'
    pictures = os.listdir(path)
    pictures = [p for p in pictures if p[-4:] == '.tif']
    precs = [f[1] for f in inspect.getmembers(_preprocess) if f[0][0:6] == "_prep_"]
    poss_settings = build_prep_possibilities(precs)
    poss_cluster = build_clust_settings()
    stacks = []
    order = list(poss_settings.keys())
    for i in range(n_tests):
        stack = []
        random.shuffle(order)
        for k in order:
            if random.random() < .75:
                stack.append(random.sample(poss_settings[k],1)[0])
        stacks.append(stack)
    cluster = random.sample(poss_cluster, n_tests)
    stack = stacks[0]
    picture = pictures[0]
    clust = cluster[0]
    for i, stack in tqdm(enumerate(stacks)):
        for picture in pictures:
            print(picture)
            mod = GrainImage(os.path.join(path, picture))
            for op in stack:
                mod.add_batch_op(op)
            for clust in cluster:
                start = datetime.now()
                mod.run_batch()
                try:
                    mod.run_analysis(clust)
                except ValueError:
                    print('Bad Settings!')
                    results.append({'image': picture,
                                    'clust': clust,
                                    'stack': stack,
                                    'ks': 99,
                                    'comment': 'no discriminance'})
                    break
                if (datetime.now() - start).total_seconds() > 1800:
                    print('Timeout!')
                    results.append({'image': picture,
                                    'clust': clust,
                                    'stack': stack,
                                    'ks': 99,
                                    'comment': 'timeout'})
                    break
                ks = mod.calc_KS_gen()
                results.append({'image': picture,
                                'clust': clust,
                                'stack': stack,
                                'ks': ks})
            del mod
        with open('res/randomSearchRes_%d.json' % i, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    main(20)
