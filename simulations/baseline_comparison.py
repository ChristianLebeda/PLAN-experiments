import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import sys
sys.path.append(".")
sys.path.append("..")
import time
import os
from plan.mean_estimation import clip_to_vector, plan, plan_noscale
from quantile_binary_search.method import random_rotation_mean
from util.parameters import AlgorithmParameters, ErrorParameters, Results, ExperimentOutput, ExperimentSettings
from util.constants import STRATEGY_FROM_RANDOM_INTEGER_RANGE, STRATEGY_ONE_SKEWED, STRATEGY_UNIFORM_VALUE, STRATEGY_RANDOM_SKEWED, STRATEGY_R_SKEWED, STRATEGY_ZIPF_SKEWED, UNCLIPPED_MEAN_ESTIMATION_LABEL, MEAN_ESTIMATION_LABEL
from util.helpers import *

def plan_wrapper(input:AlgorithmParameters, fill:bool, regularization: bool, kaplan:bool):
    input.fill = fill
    input.regularization = regularization
    input.kaplan = kaplan
    return plan(input)


if __name__ == "__main__":
    # Create a folder to store results in; use timestamps in name to avoid collisions
    output_folder = './output-{t}/'.format(t=str(time.time()).split('.')[0])
    csv_filename = '{pre}dataframe.csv'.format(pre=output_folder)
    plot_filename = '{pre}plot_mean.pdf'.format(pre=output_folder)
    settings_filename = '{pre}settings.json'.format(pre=output_folder)
    

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("algorithms", nargs='*')
    parser.add_argument("--csv")

    args = parser.parse_args()

    experiments = ['GaussianA', 'GaussianAlow', 'GaussianB', 'GaussianBlow', 'GaussianC', 'GaussianClow']
    if args.experiment not in experiments:
        print('ERROR: experiment name must be one of', experiments, 'got:', args.experiment)
        exit(1)

    # Mapping between the CLI keywords and the name we used in the code
    internal_alg_mapping = {'PLAN':'PLAN', 
                            'PLAN-noscale':'PLAN-NOSCALE', 
                            'PLAN-binsearch':'PLAN-BIN', 
                            'IOME':'Instance optimal'}
    validalg = internal_alg_mapping.keys()

    if not len(args.algorithms):
        print("No algorithms listed! Only running empirical non-private mean.")

    if not all(alg in validalg for alg in args.algorithms):
        print('ERROR: algorithms must be from', validalg)
        exit(1)

    zipf_skew = lambda d, alpha: [(d/x)**alpha for x in range(1,d+1)] #Assumption: std >= 1

    # Strategies to create covariance matrix
    strategy_options = {
        STRATEGY_ONE_SKEWED: lambda skewed, rest, d: np.diag([skewed] + [rest for _ in range(d - 1)]),
        STRATEGY_FROM_RANDOM_INTEGER_RANGE: lambda start, stop, d: np.diag(np.random.choice(range(start, stop+1), size=d)),
        STRATEGY_UNIFORM_VALUE: lambda value, d: np.diag([value for _ in range(0,d)]),
        STRATEGY_RANDOM_SKEWED: lambda d: np.diag(random_skewed_elements(d)),
        STRATEGY_R_SKEWED: lambda var, d, r: np.diag(r_skewed_elements(var, d, r)), # Sums to 1+r
        STRATEGY_ZIPF_SKEWED: lambda d, alpha: np.diag(zipf_skew(d, alpha)), #alpha \in [0,2]
    }

    # ### Old settings for debug
    # n=100*dimension
    # rho=20 * dimension/n
    # distance_to_mean = np.sqrt(max_var)*std_coverage
    # universe=(10 * -np.sqrt(d) * (mean+distance_to_mean), 10 * np.sqrt(d) * (mean+distance_to_mean))
    # covariance_matrix = strategy_options[strategy](1., (1/(d-1))**2, d) # STRATEGY_ONE_SKEWED

    runs = 50
    if os.getenv('RUNS'):
        runs = int(os.getenv('RUNS'))
    
    print(f'Running {runs} trials per parameter choice')

    # General experiment settings
    step = 20
    joint_preprocess_budget_prop = 0.25 #Match instance optimal parameters
    centering_budget_prop = 0.25
    top_k_budget_prop = 0.25 #Match instance optimal parameters
    pre_variance = 0.1
    pre_alpha = 0.1
    std_coverage = 4
    max_std_distance = 2
    k_groups = 16
    use_kaplan  = False
    fill = False
    compare_empirical_mean = False
    chisquared = True
    beta = 0.1 #Probability of failure
    correlations = [0.]#[.0, 0.2, 0.5, 0.9]

    ####################################################
    ### EXPERIMENT SETTINGS: GAUSSIAN A TINY RHO
    ####################################################
    if args.experiment == 'GaussianAlow':
        k_groups = 1
        dimensions = [16, 32, 64, 128, 256, 512, 1024]#, 2048]
        ticks = len(dimensions)
        n = 100000
        ns = np.repeat(n, len(dimensions))
        rhos = [0.01]
        mean = 0
        statistical_means = [np.repeat(mean, x) for x in dimensions]
        covariance_matrices = []
        universes = []
        strategy = STRATEGY_UNIFORM_VALUE
        calc_strategies = lambda d, r:  np.diag(np.repeat(1, d))
        strategies = []
        rs = np.repeat(0, ticks)
        universe_scaling_factor = 50
        calc_universe = lambda d, _: (-np.sqrt(d)*(universe_scaling_factor/2), np.sqrt(d)*(universe_scaling_factor/2))
        x_axis = dimensions #Inputs to loop over in main experiment loop
        x_label = 'd' #arbitrary str
        x_scale = 'linear' #linear|log
        y_scale = 'linear' #linear|log
        frame_column = 'd' #needs to match pandas column names: 'Algorithm', 'u', 'n', 'd', 'rho', 'mean', 'cov, 'r', 'run, 'error'
        title = 'GAUSSIAN A tiny rho, rho={para}, n={n}, runs={runs}, k_groups={k_groups}, step={step}'.format(para=rhos, n=n, runs=runs, k_groups=k_groups, step=step)
    ####################################################
    ### END EXPERIMENT SETTINGS: GAUSSIAN A TINY RHO
    ###################################################

    ####################################################
    ### EXPERIMENT SETTINGS: GAUSSIAN C TINY RHO
    ####################################################
    if args.experiment == 'GaussianClow':
        k_groups = 1
        dimensions = [16, 32, 64, 128, 256, 512, 1024]# 2048]
        ticks = len(dimensions)
        rhos = [0.01]
        n = 100000
        ns = np.repeat(n, ticks)
        mean = 10
        statistical_means = [np.repeat(mean, x) for x in dimensions]
        compare_empirical_mean = False
        covariance_matrices = []
        universes = []
        strategy = STRATEGY_ZIPF_SKEWED
        calc_strategies = lambda d, alpha: np.diag(zipf_skew(d, alpha))
        strategies = []
        alpha = 2
        rs = np.repeat(alpha, ticks)
        universe_scaling_factor = 100
        calc_universe = lambda d, i: (-d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2, d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2)
        x_axis = dimensions #Inputs to loop over in main experiment loop
        x_label = 'd' #arbitrary str
        x_scale = 'linear' #linear|log
        y_scale = 'linear' #linear|log
        frame_column = 'd' #needs to match pandas column names: 'Algorithm', 'u', 'n', 'd', 'rho', 'mean', 'cov, 'r', 'run, 'error'
        title = 'GAUSSIAN C tiny rho, n={n}, runs={runs}, k_groups={k_groups}, alpha={alpha}, universe_scaling={scaling}'.format(n=n, runs=runs, k_groups=k_groups, alpha=alpha, scaling=universe_scaling_factor)
    ####################################################
    ### END EXPERIMENT SETTINGS: GAUSSIAN C TINY RHO
    ####################################################

    ####################################################
    ### EXPERIMENT SETTINGS: GAUSSIAN B CASE TINY RHO
    ####################################################
    if args.experiment == 'GaussianBlow':
        k_groups = 1
        d = 64
        rs = [0, 1, 2] #"Alphas"
        ticks = len(rs)
        dimensions = np.repeat(d, ticks)
        n = 100000
        ns = np.repeat(n, ticks)
        rhos = [0.01]#[1/16, 1/8, 1/4, 1/2, 1.]
        mean = 10
        statistical_means = np.array([np.repeat(mean, d)] * ticks)
        covariance_matrices = []
        universes = []
        strategy = STRATEGY_ZIPF_SKEWED
        # calc_strategies = lambda d, r: strategy_options[strategy](d, r)
        calc_strategies = lambda d, alpha: np.diag(zipf_skew(d, alpha))
        strategies = []
        step=20
        universe_scaling_factor = 100
        calc_universe = lambda d, i: (-d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2, d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2)
        # universe_scaling_factor = 50
        # calc_universe = lambda d, _: (-np.sqrt(d)*(universe_scaling_factor/2), np.sqrt(d)*(universe_scaling_factor/2))
        x_axis = rs #Inputs to loop over in main experiment loop
        x_label = 'alpha' #arbitrary str
        x_scale = 'linear' #linear|log
        y_scale = 'log' #linear|log
        frame_column = 'r' #needs to match pandas column names: 'Algorithm', 'u', 'n', 'd', 'rho', 'mean', 'cov, 'r', 'run, 'error'
        title = 'GAUSSIAN B tiny rho, d={dim}, rho={rho}, n={n}, mu={mu}, runs={runs}, k_groups={k_groups}, steps={step}, universe scaling={c}'.format(dim=d, rho=rhos, n=n, runs=runs, k_groups=k_groups, mu=mean, step=step, c=universe_scaling_factor)
    ####################################################
    ### END EXPERIMENT SETTINGS: GAUSSIAN B CASE TINY RHO
    ####################################################



    ####################################################
    ### EXPERIMENT SETTINGS: GAUSSIAN A
    ####################################################
    if args.experiment == 'GaussianA':
        k_groups = 1
        dimensions = [16, 32, 64, 128, 256, 512, 1024, 2048]
        ticks = len(dimensions)
        n = 4000
        ns = np.repeat(n, len(dimensions))
        rhos = [1/16, 1/8, 1/4, 1/2, 1.]
        mean = 0
        statistical_means = [np.repeat(mean, x) for x in dimensions]
        covariance_matrices = []
        universes = []
        strategy = STRATEGY_UNIFORM_VALUE
        calc_strategies = lambda d, r:  np.diag(np.repeat(1, d))
        strategies = []
        rs = np.repeat(0, ticks)
        universe_scaling_factor = 50
        calc_universe = lambda d, _: (-np.sqrt(d)*(universe_scaling_factor/2), np.sqrt(d)*(universe_scaling_factor/2))
        x_axis = dimensions #Inputs to loop over in main experiment loop
        x_label = 'd' #arbitrary str
        x_scale = 'linear' #linear|log
        y_scale = 'linear' #linear|log
        frame_column = 'd' #needs to match pandas column names: 'Algorithm', 'u', 'n', 'd', 'rho', 'mean', 'cov, 'r', 'run, 'error'
        title = 'GAUSSIAN A, rho={para}, n={n}, runs={runs}, k_groups={k_groups}, step={step}'.format(para=rhos, n=n, runs=runs, k_groups=k_groups, step=step)
    ####################################################
    ### END EXPERIMENT SETTINGS: GAUSSIAN A
    ###################################################

    # ####################################################
    # ### EXPERIMENT SETTINGS: GAUSSIAN C
    # ####################################################
    if args.experiment == 'GaussianC':
        k_groups = 1
        dimensions = [16, 32, 64, 128, 256, 512, 1024, 2048]
        ticks = len(dimensions)
        rhos = [1/16, 1/8, 1/4, 1/2, 1.]
        n = 10000
        ns = np.repeat(n, ticks)
        mean = 10
        statistical_means = [np.repeat(mean, x) for x in dimensions]
        compare_empirical_mean = False
        covariance_matrices = []
        universes = []
        strategy = STRATEGY_ZIPF_SKEWED
        calc_strategies = lambda d, alpha: np.diag(zipf_skew(d, alpha))
        strategies = []
        alpha = 2
        rs = np.repeat(alpha, ticks)
        universe_scaling_factor = 100
        calc_universe = lambda d, i: (-d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2, d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2)
        x_axis = dimensions #Inputs to loop over in main experiment loop
        x_label = 'd' #arbitrary str
        x_scale = 'linear' #linear|log
        y_scale = 'linear' #linear|log
        frame_column = 'd' #needs to match pandas column names: 'Algorithm', 'u', 'n', 'd', 'rho', 'mean', 'cov, 'r', 'run, 'error'
        title = 'GAUSSIAN C, n={n}, runs={runs}, k_groups={k_groups}, alpha={alpha}, universe_scaling={scaling}'.format(n=n, runs=runs, k_groups=k_groups, alpha=alpha, scaling=universe_scaling_factor)
    # ####################################################
    # ### END EXPERIMENT SETTINGS: GAUSSIAN C
    # ####################################################

    ####################################################
    ### EXPERIMENT SETTINGS: GAUSSIAN B CASE
    ####################################################
    if args.experiment == 'GaussianB':
        k_groups = 1
        d = 2048
        rs = [0, 0.5, 1, 1.5, 2] #"Alphas"
        ticks = len(rs)
        dimensions = np.repeat(d, ticks)
        n = 10000
        ns = np.repeat(n, ticks)
        rhos = [1/16, 1/8, 1/4, 1/2, 1.]
        correlation = [.0, 0.2, 0.5, 0.9]
        mean = 10
        statistical_means = np.array([np.repeat(mean, d)] * ticks)
        covariance_matrices = []
        universes = []
        strategy = STRATEGY_ZIPF_SKEWED
        # calc_strategies = lambda d, r: strategy_options[strategy](d, r)
        calc_strategies = lambda d, alpha: np.diag(zipf_skew(d, alpha))
        strategies = []
        step=20
        universe_scaling_factor = 100
        # calc_universe = lambda d: (-np.sqrt(d)*(universe_scaling_factor/2), np.sqrt(d)*(universe_scaling_factor/2))
        # calc_universe = lambda d, i: (-d*np.sqrt(covariance_matrices[i].max())*universe_scaling_factor/2, d*np.sqrt(covariance_matrices[i].max())*universe_scaling_factor/2)
        calc_universe = lambda d, i: (-d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2, d*(covariance_matrices[i].max()**0.25)*universe_scaling_factor/2)
        # calc_universe = lambda d, i: (-d*np.sqrt(d)*(universe_scaling_factor/2), d*np.sqrt(d)*(universe_scaling_factor/2))
        # calc_universe = lambda d: (sys.float_info.min, np.sqrt(sys.float_info.max))
        x_axis = rs #Inputs to loop over in main experiment loop
        x_label = 'alpha' #arbitrary str
        x_scale = 'linear' #linear|log
        y_scale = 'log' #linear|log
        frame_column = 'r' #needs to match pandas column names: 'Algorithm', 'u', 'n', 'd', 'rho', 'mean', 'cov, 'r', 'run, 'error'
        title = 'GAUSSIAN B, d={dim}, rho={rho}, n={n}, mu={mu}, runs={runs}, k_groups={k_groups}, steps={step}, universe scaling={c}'.format(dim=d, rho=rhos, n=n, runs=runs, k_groups=k_groups, mu=mean, step=step, c=universe_scaling_factor)
    ####################################################
    ### END EXPERIMENT SETTINGS: GAUSSIAN B CASE
    ####################################################

    
    for corr in correlations:
        output_folder = './output-{t}-{corr}/'.format(t=str(time.time()).split('.')[0], corr=corr)
        csv_filename = '{pre}dataframe.csv'.format(pre=output_folder)
        plot_filename = '{pre}plot_mean.pdf'.format(pre=output_folder)
        settings_filename = '{pre}settings.json'.format(pre=output_folder)

        strategies = []
        covariance_matrices = []
        universes = []

        # Pre-calculate the experiment settings so we can just iterate over an array of inputs later
        for d, x_axis_index in zip(dimensions, range(0, len(dimensions))):
            strategies.append(strategy)
            while True:
                covariance_matrix = calc_strategies(d, rs[x_axis_index])  
                # NEW
                v = np.sqrt(np.diag(covariance_matrix))
                v = [[x * y for x in v] for y in v]
                for i in range(len(v)):
                    for j in range(len(v)):
                        if i != j:
                            v[i][j] = corr * v[i][j] 
                covariance_matrix = np.array(v)
                try:
                    data = np.random.multivariate_normal(statistical_means[x_axis_index], covariance_matrix, size=ns[x_axis_index],check_valid="raise")
                    covariance_matrices.append(covariance_matrix)

                    universes.append(calc_universe(d, x_axis_index))
                    break
                except:
                    print("matrix is not SDP. trying again...")
                    pass


        # Datastructure to save all the settings used e-asily to file
        experimentSettings = ExperimentSettings(step=step,
                                                recenter_and_std_budget_proportion=joint_preprocess_budget_prop,
                                                recenter_budget_proportion=centering_budget_prop,
                                                top_k_budget_proportion=top_k_budget_prop,
                                                k=[], #TODO: remove parameter
                                                preprocess_variance=pre_variance,
                                                preprocess_alpha=pre_alpha,
                                                runs=runs,
                                                dimensions=dimensions,
                                                ns=ns,
                                                rhos=rhos,
                                                beta=beta,
                                                statistical_means=statistical_means[0][0],#We always use same mean
                                                covariance_matrices=covariance_matrices,
                                                universes=universes,
                                                strategies=strategies,
                                                rs=rs,
                                                chisquared=chisquared,
                                                correlation=corr
                                                )

        algorithms = {
            'PLAN-BIN': lambda x: plan_wrapper(x, fill=fill, regularization=True, kaplan=False), #Binary search
            'PLAN': lambda x: plan_wrapper(x, fill=fill, regularization=True, kaplan=True),
            'PLAN-NOSCALE': lambda x: plan_noscale(x),
            'Instance optimal': lambda x: random_rotation_mean(x.data, x.dimensions, x.universe, x.rho, x.step, x.recenter_and_std_budget_proportion),
            'Empirical mean (non-private)': lambda x: Results(np.mean(x.data, axis=0), np.empty(0)),
            }

        # Remove unused algorithms from algorithms dict. Empirical mean is always used
        for k, v in internal_alg_mapping.items():
            if k not in args.algorithms:
                del algorithms[v]

        include_unclipped = False #Unclipped means on/off
        
        error_terms = {
            # 'sqrt(d/n)': lambda x: np.sqrt(x.dimensions/x.n),
            # 'd/n*sqrt(rho)': lambda x: x.dimensions/(x.n*np.sqrt(x.rho)),
            # '1/d': lambda x: 1/x.dimensions,
            # 'Them': lambda x:(np.sqrt(x.dimensions)*np.linalg.norm(x.sigmas))/(x.n*np.sqrt(x.rho)),
            # 'Us': lambda x: np.linalg.norm(x.sigmas, ord=1)/(x.n*np.sqrt(x.rho)),
            # 'Sampling': lambda x: np.linalg.norm(x.sigmas)/(np.sqrt(x.n)), 
        }

        to_plot = {key: [] for key in list(error_terms.keys())}

        results = pd.DataFrame()
        if args.csv:
            results = pd.read_csv(args.csv)
            print("loaded csv")


        for x, x_axis_index in zip(x_axis, range(0, len(x_axis))):
            print(x_label, x)
            print("Universe: ", universes[x_axis_index])
            
            for rho_index, rho in enumerate(rhos):
                errors = {key: {MEAN_ESTIMATION_LABEL: []} for key in algorithms.keys()}
                to_add = []


                for i in range(runs):
                    # Fresh data each run, drawn from a set distribution
                    data = np.random.multivariate_normal(statistical_means[x_axis_index], covariance_matrices[x_axis_index], size=ns[x_axis_index],check_valid="raise")
                    # Algorithm parameters
                    d = dimensions[x_axis_index]
                    params = AlgorithmParameters(data, 
                                                    d, 
                                                    universes[x_axis_index], 
                                                    rho, 
                                                    alpha=pre_alpha, 
                                                    variance=pre_variance, 
                                                    step=step, 
                                                    k=0, #TODO: remove parameter 
                                                    cov=covariance_matrices[x_axis_index],
                                                    recenter_and_std_budget_proportion=joint_preprocess_budget_prop, 
                                                    recenter_budget_proportion=centering_budget_prop,
                                                    top_k_budget_proportion=top_k_budget_prop,
                                                    fill=False,
                                                    regularization=False,
                                                    k_groups=k_groups,
                                                    max_std_distance=max_std_distance,
                                                    kaplan=use_kaplan,
                                                    chisquared=chisquared,
                                                    beta=beta,
                    )

                    for key in algorithms.keys():

                        # Check if we've already ran this setting before, then reuse that result
                        if args.csv:
                            matches = find_match(results, key, universes[x_axis_index], ns[x_axis_index], dimensions[x_axis_index], rhos[x_axis_index], statistical_means[x_axis_index], strategies[x_axis_index], i, rs[x_axis_index])
                            if len(matches)>0:
                                print("already ran", key, dimensions[x_axis_index], ns[x_axis_index], rhos[x_axis_index])
                                continue

                        # Calculate and store
                        output = algorithms[key](params)
                        # Compare either with empirical mean, or statistical mean
                        ground_truth = np.mean(data, axis=0) if compare_empirical_mean else statistical_means[x_axis_index]
                        error = np.linalg.norm(output.mean-ground_truth)
                        errors[key][MEAN_ESTIMATION_LABEL].append(error)
                        to_add.append(output_to_dict(ExperimentOutput(key, params, statistical_means[x_axis_index][0], strategies[x_axis_index], rs[x_axis_index], i, error)))

                        # Calculate and store
                        if(include_unclipped):
                            if(len(output.unclipped_mean)>0): # Defensive: some algorithms don't clip, check if key's there first
                                unclipped = np.linalg.norm(output.unclipped_mean-statistical_means[x_axis_index])
                                if UNCLIPPED_MEAN_ESTIMATION_LABEL in errors[key].keys():
                                    errors[key][UNCLIPPED_MEAN_ESTIMATION_LABEL].append(unclipped)
                                else:
                                    errors[key][UNCLIPPED_MEAN_ESTIMATION_LABEL] = [unclipped]

                                to_add.append(output_to_dict(ExperimentOutput( "{k}-UNCLIPPED".format(k=key), params, statistical_means[x_axis_index][0], strategies[x_axis_index], rs[x_axis_index], i, unclipped)))
                        
                # Save error terms separatley
                for key in error_terms.keys():
                    error_params = ErrorParameters(n=ns[x_axis_index], dimensions=dimensions[x_axis_index], rho=rhos[x_axis_index], sigmas=np.sqrt(covariance_matrices[0].diagonal()))
                    to_plot[key].append((x, error_terms[key](error_params)))
                
                # Dump data to frame
                results = pd.concat([results, pd.DataFrame(to_add)])

        plot_and_save(results, to_plot, experimentSettings.__dict__, output_folder, plot_filename, csv_filename, settings_filename, x_label, x_scale, frame_column, y_scale, title)
    print("Universes used", universes)