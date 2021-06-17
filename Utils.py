"""
Copyright 2020 Jan Sosulski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2021 GitHub, Inc.

This file was modified from the original by Jan Sosulski
"""
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from ResampledLDA import ResampledLinearDiscriminantAnalysis as ResLDA
from ResampledLDA import ShrinkageLinearDiscriminantAnalysis as SLDA
from Vectorizer import Vectorizer

from moabb.paradigms import P300
from moabb.datasets import EPFLP300, bi2013a
from moabb.datasets import BNCI2014009 as bnci_1
from moabb.datasets import BNCI2014008 as bnci_als
from moabb.datasets import BNCI2015003 as bnci_2
from Spot_pilot import SpotPilotData

import pyriemann
import numpy as np

def create_lda_pipelines(cfg, feature_preprocessing_key, N_channels):
    pipelines = dict()
    fs = cfg['default']['data_preprocessing']['sampling_rate']
    cfg_vect = cfg['default'][feature_preprocessing_key]['feature_preprocessing']
    c_jm = cfg_vect['jumping_means_ival']
    c_sel = cfg_vect['select_ival']
    vectorizers = dict()
    key = 'numerous'
    #for key in c_jm:
    vectorizers[f'jm_{key}'] = dict(vec=Vectorizer(jumping_mean_ivals=c_jm[key]['ival']), D=c_jm[key]['D'], fs=fs)
    #for key in c_sel:
    #    vectorizers[f'sel_{key}'] = dict(vec=Vectorizer(select_ival=c_sel[key]['ival']), D=c_sel[key]['D'], fs=fs)

    classifiers = dict(
        r_lda = ResLDA(),
    )
    print(classifiers)
    for v_key in vectorizers.keys():
        D = vectorizers[v_key]['D']
        vec = vectorizers[v_key]['vec']
        for c_key in classifiers.keys():
            clf = clone(classifiers[c_key])
            clf.N_times = D
            new_key = f'{v_key}_{c_key}'
            clf.preproc = vec  # why does this not persist :(
            pipelines[new_key] = make_pipeline(vec, clf)

    return pipelines


def get_benchmark_config(dataset_name, cfg_prepro, data_path, subjects=None, sessions=None):
    benchmark_cfg = dict()
    paradigm = P300(resample=cfg_prepro['sampling_rate'], fmin=cfg_prepro['fmin'], fmax=cfg_prepro['fmax'],
                    baseline=cfg_prepro['baseline'])
    load_ival = [0, 1]
    if dataset_name == 'spot_single':
        d = SpotPilotData(reject_non_iid=False, load_single_trials=True)
        d.path = data_path
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = d.N_channels
    elif dataset_name == 'epfl':
        d = EPFLP300()
        d.interval = load_ival
        d.unit_factor = 1
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 32
    elif dataset_name == 'bnci_1':
        d = bnci_1()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 16
    elif dataset_name == 'bnci_als':
        d = bnci_als()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 8
    elif dataset_name == 'bnci_2':
        d = bnci_2()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 8
    elif dataset_name == 'braininvaders':
        d = bi2013a()
        d.interval = load_ival
        if subjects is not None:
            d.subject_list = [d.subject_list[i] for i in subjects]
        n_channels = 16
    else:
        raise ValueError(f'Dataset {dataset_name} not recognized.')

    benchmark_cfg['dataset'] = d
    benchmark_cfg['N_channels'] = n_channels
    benchmark_cfg['paradigm'] = paradigm
    return benchmark_cfg
