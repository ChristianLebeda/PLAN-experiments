from collections import Counter
import sys
sys.path.append(".")
sys.path.append("..")
from plan.mean_estimation import plan_binary, binary_mean_only
from util.parameters import Results
from quantile_binary_search.method import random_rotation_mean
import numpy as np
import time
import os
import pandas as pd


def preprocess(data):
    universe = set()
    for vec in data:
        for y in vec:
            universe.add(y)

    universe_mapping = {}
    i = 0
    for u in sorted(universe):
        universe_mapping[u] = i
        i += 1

    for vec in data:
        for i, y in enumerate(vec):
            vec[i] = universe_mapping[y]

    return data, len(data), len(universe)

def kosarak(fn):
    import os
    from urllib.request import urlretrieve
    import h5py
    if not os.path.exists(fn):
        urlretrieve("http://ann-benchmarks.com/kosarak-jaccard.hdf5", fn)
    f = h5py.File(fn, "r")
    data = sparse_to_lists(f['train'], f['size_train']) + sparse_to_lists(f['test'], f['size_test'])
    return data

def POS(fn):
    import os
    from urllib.request import urlretrieve
    if not os.path.exists(fn):
        urlretrieve("https://github.com/cpearce/HARM/raw/master/datasets/BMS-POS.csv", fn)
    data = []
    with open(fn) as f:
        for line in f:
            data.append(list(map(int, line.split(",")[:-1])))
    return data

def sparse_to_lists(data, lengths):
    X = []
    index = 0
    for l in lengths:
        X.append(data[index:index+l])
        index += l
    return X


def synthetic_data(N, D, p=0.5, q=0.01, ratio=.2):
    vecs = []
    groundtruth = [p if i < int(ratio * D) else q for i in range(D)]
    for _ in range(N):
        vec = []
        arr = np.random.rand(D)
        for i in range(D):
            comp = p if i < int(ratio * D) else q
            if arr[i] < comp:
                vec.append(i)
        vecs.append(vec)
    return vecs, groundtruth


def get_groundtruth(data, N, D):
    c = Counter()
    for x in data:
        c.update(x)

    return np.array([c[i] / N for i in range(D)])


if __name__ == "__main__":
    # Avoid name collision by using timestamp in folder name
    folder = './output-binary-case-{t}/'.format(t=str(time.time()).split('.')[0])

    if not os.path.exists(folder):
        os.makedirs(folder)
        print('Creating folder', folder)

    runs = 50

    real_runs = {
            "kosarak": (kosarak, "kosarak.hdf5", 32768, []),
            "POS": (POS, "POS.csv", 2048, []),
    }

    methods = []
    if "PLAN" in sys.argv:
        methods.append("paper")
    if "PLAN-noscale" in sys.argv:
        methods.append("noscale")
    if "PLAN-oracle" in sys.argv:
        methods.append("oracle")

    # Experiments on real datasets
    for name, (f, fn, padded_D, results) in real_runs.items():
        # Skip experiments not in arguments
        if name not in sys.argv:
            continue

        print(f"running experiments on dataset: {name}")
        data = f(fn)

        exp_data, N, D = preprocess(data)

        gt_p = get_groundtruth(exp_data, N, D)

        print(D)

        for rho in [0.0625, 0.125, 0.25, 0.5, 1]:
            print("Running rho=", rho)
            for i in range(runs):
                for method in methods:
                    print("Running ", method)

                    start = time.time()
                    res = plan_binary(exp_data, N, D, rho, method=method)
                    end = time.time()
                    error = np.linalg.norm(gt_p - res.mean, ord=1)/2
                    error_unclipped = np.linalg.norm(gt_p - res.unclipped_mean, ord=1)/2
                    results.append({'label':f'PLAN({method})-clipped', 'run': i, 'error': error, 'rho': rho, 'time': end - start})
                    results.append({'label':f'PLAN({method})-unclipped', 'run': i, 'error': error_unclipped, 'rho': rho, 'time': end - start})
                    print(error)

            # Code for running IOME
            for i in range(2):
                if "IOME" not in sys.argv:
                    continue

                b = np.zeros((N, padded_D))

                for x, vec in enumerate(exp_data):
                    for y in vec:
                        b[x][y] = 1

                start = time.time()
                rr_res = random_rotation_mean(b, padded_D, (0, padded_D**.5), p=rho)
                end = time.time()
                rr_res = Results(rr_res.mean[:D], rr_res.unclipped_mean[:D])

                # Calculate errors
                rr_clipped_error = np.linalg.norm(gt_p - rr_res.mean, ord=1)/2
                rr_unclipped_error = np.linalg.norm(gt_p - rr_res.unclipped_mean, ord=1)/2

                results.append({'label':'instance optimal', 'run': i, 'error': rr_clipped_error, 'rho': rho, 'time': end - start})
                results.append({'label':'instance optimal unclipped', 'run': i, 'error': rr_unclipped_error, 'rho': rho, 'time': end - start})

                print("clipped", rr_clipped_error)
                print("unclipped", rr_unclipped_error)

            # Store to disk
            df = pd.DataFrame(results)
            df.to_csv(f'{folder}/{name}_errors.csv')


    if "synthetic" in sys.argv:
        # Experiments on synthetic data
        N = 4096
        synthetic_results = []
        for D in [256, 512, 1024, 2048]:
            for rho in [.0625, .125, .25, .5, 1]:
                for i in range(runs):
                    for ratio in [0.0, .25, .5, .75, 1.0]:
                        a, groundtruth = synthetic_data(N, D, ratio=ratio)

                        #print(ratio)
                        gt_p = get_groundtruth(a, N, D)
                        synthetic_results.append({'label': 'Empirical mean (non-private)', 'run': i, 'error': np.linalg.norm(gt_p -  groundtruth, ord=1)/2, 'n': N, 'd': D, "rho": rho, "ratio": ratio, "time": .0})

                        if len(methods):
                            print("Running with PLAN")
                        for method in methods:
                            print("Running ", method)
                            start = time.time()
                            paper_scaled_res = plan_binary(a, N, D, rho=rho, method=method)
                            end = time.time()
                            # Calculate errors
                            paper_scaled_clipped_error = np.linalg.norm(groundtruth -  paper_scaled_res.mean, ord=1)/2
                            paper_scaled_unclipped_error = np.linalg.norm(groundtruth -  paper_scaled_res.unclipped_mean, ord=1)/2
                            # Append to result list
                            # synthetic_results.append({'label':'PLAN, unclipped', 'run': i, 'error':  paper_scaled_unclipped_error, 'rho': rho, 'alpha': alpha, 'n': N, 'd': D})
                            # synthetic_results.append({'label':'PLAN, clipped', 'run': i, 'error':  paper_scaled_clipped_error, 'rho': rho, 'alpha': alpha, 'n': N, 'd': D})
                            synthetic_results.append({'label':f'PLAN({method}), unclipped', 'run': i, 'error':  paper_scaled_unclipped_error, 'rho': rho, "ratio": ratio, 'n': N, 'd': D, 'time': end - start})
                            synthetic_results.append({'label':f'PLAN({method}), clipped', 'run': i, 'error':  paper_scaled_clipped_error, 'rho': rho,  "ratio": ratio, 'n': N, 'd': D, 'time': start-end})
                            print("clipped",  paper_scaled_clipped_error)
                            print("unclipped",  paper_scaled_unclipped_error)


                        if "IOME" in sys.argv:
                            print("Running IOME")

                            b = np.zeros((N, D))

                            for x, vec in enumerate(a):
                                for y in vec:
                                    b[x][y] = 1

                            start = time.time()
                            rr_res = random_rotation_mean(b, D, (0, D**.5), p=rho)
                            end = time.time()
                            # Calculate errors
                            rr_clipped_error = np.linalg.norm(groundtruth - rr_res.mean, ord=1)/2
                            rr_unclipped_error = np.linalg.norm(groundtruth - rr_res.unclipped_mean, ord=1)/2

                            synthetic_results.append({'label':'instance optimal, unclipped', 'run': i, 'error': rr_unclipped_error, 'rho': rho,  "ratio": ratio, 'n': N, 'd': D, 'time': end - start})
                            synthetic_results.append({'label':'instance optimal, clipped', 'run': i, 'error': rr_clipped_error, 'rho': rho,  "ratio": ratio, 'n': N, 'd': D, 'time': end - start})
                            print("clipped", rr_clipped_error)
                            print("unclipped", rr_unclipped_error)

    #                    stds = np.sqrt(gt_p * (1-gt_p))
    #
    #                    print("Theory:")
    #                    print("Expected length of vector:", np.sum(gt_p)**.5)
    #                    print("Expected length of scaled vector:", np.sqrt(np.sum((gt_p**2 / (1-gt_p))**(1/3))))
    #                    print("error theirs: ", (4/3.1415)**.5 * D *  np.sum(gt_p)**.5 / (rho**.5 * N) / 2)
    #                    print("ours:", (4/3.1415)**.5 * np.sum(stds**(1/3)) * np.sqrt(np.sum((gt_p**2 / (1-gt_p))**(1/3))) / (rho**.5 * N) / 2)

        # Store to disk
        synthetic_frame = pd.DataFrame(synthetic_results)
        synthetic_frame.to_csv('{folder}/synthetic_errors.csv'.format(folder=folder))
