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
from moabb.evaluations import WithinSessionEvaluation
import moabb
import numpy as np
import warnings
from datetime import datetime as dt
import yaml
import argparse
import uuid
import os
from pathlib import Path
import ResampledLDA
from shutil import copyfile
from utils import create_lda_pipelines

from utils import get_benchmark_config
import time

LOCAL_CONFIG_FILE = r'local_config.yaml'
ANALYSIS_CONFIG_FILE = r'analysis_config.yaml'

t0 = time.time()

##############################################################################
# Argument and configuration parsing
##############################################################################

# Open local configuration
with open(LOCAL_CONFIG_FILE, 'r') as conf_f:
    local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

RESULTS_ROOT = Path(local_cfg['results_root'])
DATA_PATH = local_cfg['data_root']

with open(ANALYSIS_CONFIG_FILE, 'r') as conf_f:
    ana_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

VALID_DATASETS = ['spot_single', 'epfl', 'bnci_1', 'bnci_als', 'bnci_2', 'braininvaders']

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('dataset', help=f'Name of the dataset. Valid names: {VALID_DATASETS}')
parser.add_argument('subjects_sessions', help='[Optional] Indices of subjects to benchmark.', type=str, nargs='*')
args = parser.parse_args()

print(args)

dataset_name = args.dataset
subject_session_args = args.subjects_sessions
if ' ' in dataset_name and len(subject_session_args) == 0:
    subject_session_args = dataset_name.split(' ')[1:]
    dataset_name = dataset_name.split(' ')[0]
if dataset_name not in VALID_DATASETS:
    raise ValueError(f'Invalid dataset name: {dataset_name}. Try one from {VALID_DATASETS}.')
if len(subject_session_args) == 0:
    subjects = None
    sessions = None
else:  # check whether args have format [subject, subject, ...] or [subject:session, subject:session, ...]
    if np.all([':' in s for s in subject_session_args]):
        subjects = [int(s.split(':')[0]) for s in subject_session_args]
        sessions = [int(s.split(':')[1]) for s in subject_session_args]
    elif not np.any([':' in s for s in subject_session_args]):
        subjects = [int(s.split(':')[0]) for s in subject_session_args]
        sessions = None
    else:
        raise ValueError('Currently, mixed subject:session and only subject syntax is not supported.')
print(f'Subjects: {subjects}')
print(f'Sessions: {sessions}')

start_timestamp_as_str = dt.now().replace(microsecond=0).isoformat()

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

moabb.set_log_level('warn')

np.random.seed(42)

##############################################################################
# Create pipelines
##############################################################################

labels_dict = {'Target': 1, 'NonTarget': 0}

prepro_cfg = ana_cfg['default']['data_preprocessing']

bench_cfg = get_benchmark_config(dataset_name, prepro_cfg, data_path=DATA_PATH, subjects=subjects,
                                 sessions=sessions)

if hasattr(bench_cfg['dataset'], 'stimulus_modality'):
    feature_preprocessing_key = bench_cfg['dataset'].stimulus_modality
else:
    feature_preprocessing_key = ana_cfg['default']['fallback_modality']

labels_dict = {'Target': 1, 'NonTarget': 0}

pipelines = dict()
pipelines.update(create_lda_pipelines(ana_cfg, feature_preprocessing_key, 31))

# IF YOU WANT TO DOWNLOAD ALL DATA FIRST, UNCOMMENT THE NEXT LINE
#pipelines = dict(test=pipelines['jm_few_lda_p_cov'])
##############################################################################
# Evaluation
##############################################################################

identifier = f'{dataset_name}_subj_{subjects if subjects is not None else "all"}' \
             f'_sess_{sessions if sessions is not None else "all"}_{start_timestamp_as_str}'.replace(' ', '')
unique_suffix = f'{identifier}_{uuid.uuid4()}'

evaluation = WithinSessionEvaluation(paradigm=bench_cfg['paradigm'], datasets=bench_cfg['dataset'],
                                     overwrite=True, random_state=8)
results = evaluation.process(pipelines)
print(results)
result_path = RESULTS_ROOT / f'{identifier}_results.csv'.replace(':', '_')

results.to_csv(result_path, encoding='utf-8', index=False)
t1 = time.time()
print(f'Benchmark run completed. Elapsed time: {(t1-t0)/3600} hours.')

