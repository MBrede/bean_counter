from model import GrainImage
import os
import pickle
from time import sleep

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


class MyWorker(Worker):
    def __init__(self, image, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.path = image
        self.mod = GrainImage(image)

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('hist.pos', lower=0, upper=4))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nlmeans.pos', lower=0, upper=4))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nlmeans.h', lower=1, upper=30))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nlmeans.temp', lower=7, upper=19))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nlmeans.search', lower=21, upper=51))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('gauss.pos', lower=0, upper=4))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('gauss.k', lower=1, upper=15))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('sav.pos', lower=0, upper=4))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('sav.window', lower=1, upper=51))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('sav.poly', lower=1, upper=11))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('max_grain_ratio', lower=1, upper=7, q=0.1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('SLIC.compactness', lower=0.01, upper=0.3, q=0.01))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('SLIC.n_segments', lower=100, upper=10000, q = 100))
        return(config_space)


    def compute(self, config, budget, **kwargs):
        sleep(.1)
        stack = build_prep_stack(config)
        for op in stack:
            self.mod.add_batch_op(op)
            self.mod.sample_window(budget)
            self.mod.set_max_ratio(config['max_grain_ratio'])
            self.mod.run_batch()
            self.mod.run_analysis(build_clust_settings(config))
            ks = self.mod.calc_KS_gen()

        return({
                    'loss': float(ks),  # this is the a mandatory field to run hyperband
                    'info': [self.path, config, ks]  # can be used for any user-defined information - also mandatory
                })


if __name__ == '__main__':
    path = '../data'
    pictures = os.listdir(path)
    pictures = [p for p in pictures if p[-4:] == '.tif']
    for p in pictures:
        NS = hpns.NameServer(run_id='slic1', host='127.0.0.1', port=None)
        NS.start()

        workers=[]
        for i in range(64):
            w = MyWorker(image = os.path.join(path, p), nameserver='127.0.0.1',run_id='slic1', id=i)
            w.run(background=True)
            workers.append(w)

        bohb = BOHB(  configspace = w.get_configspace(),
                      run_id = 'slic1', nameserver='127.0.0.1',
                      min_budget=0.45, max_budget=1.0
                   )
        res = bohb.run(n_iterations=500, min_n_workers=len(workers))

        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

        with open(f'res/{p[:-4]}_slic.pkl', 'wb') as f:
            pickle.dump(res, f)
