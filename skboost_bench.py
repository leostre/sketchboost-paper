import argparse

import joblib
import os

from itertools import product
from joblib import Parallel, delayed
from multiprocessing import Queue

from utils import run_cv_loop

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--ngpus', type=int)
parser.add_argument('-r', '--runner', type=str)

NAME = 'SKETCHBOOST_ALLK'

RUNNER_PATH = 'runner.py'
DEBUG = True

TIMEOUT = 3600 * 24
HARD_TIMEOUT = 3600 * 36
NTHREADS = 8
SEED = 42

# benchmark_params
data_path = 'data/processed'
benchmark_path = 'runs'


def get_pb_results(name, benchmark_path, data_path, dataset, method, dim, runner, rewrite=False):
    gpu = q.get()

    params = joblib.load(os.path.join(benchmark_path, 'baselines_and_params.pkl'))[dataset, 'cb']['params']
    params['method'] = method
    params['dim'] = dim
    params['use_hess'] = True

    params = {**params, **{'lr': 0.015, 'ntrees': 20000, 'es': 500}}

    trial = method + '_' + str(dim)
    print(params)

    run_cv_loop(name, gpu, benchmark_path, data_path, dataset, runner, trial, params, rewrite=rewrite)

    q.put(gpu)


if __name__ == '__main__':

    args = parser.parse_args()
    runner = args.runner
    q = Queue(maxsize=args.ngpus)
    for i in range(args.ngpus):
        q.put([i])

    data_info = joblib.load(os.path.join(data_path, 'data_info.pkl'))
    datasets = [
        # 'otto', 
                    #  'dionis', 
                    #  'helena',
                    #    'sf-crime', 
                       'moa',
                        #    'delicious', 
                        #    'mediamill', 
                        #    'scm20d', 
                        #    'rf1'
                           ]

    strats = ['topk',
            #    'random', 'proj', 
               ]
    ks = [
        # 1,
        #   2, 
        # 5, 
        10,
          20
        ]

    combinations = [(ds, method, k) for (ds, method, k) in product(datasets, strats, ks) if data_info[ds]['nout'] > k
                    ] #+ [(ds, 'raw', 10000) for ds in datasets]

    # get runs
    Parallel(n_jobs=args.ngpus, backend="threading")(
        delayed(get_pb_results)(
            NAME, benchmark_path, data_path, ds, method, k, runner, rewrite=True
        ) for (ds, method, k) in combinations
    )
