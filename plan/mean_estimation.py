import numpy as np
import sys
sys.path.append(".")
sys.path.append("..")
from quantile_binary_search.method import quantile_binary_search_mp, quantile_binary_search
from kaplan_et_al.single_quantile_algo import kaplan_quantile
from util.parameters import Predictions, Results, AlgorithmParameters
from util.constants import MIN_VALUE

DEBUG = False # True

# Add Gaussian noise to spread out the values (to aid the quantile search)
def _plan_preprocess_data(data, alpha, universe, variance=0):
    if variance>0:
        noise = np.random.normal(loc=0, scale=np.sqrt(variance), size=data.shape) #Scale is std
        data += noise
    noised_and_spread_data = np.concatenate(((data+alpha),(data),(data-alpha)))
    to_return = np.clip(noised_and_spread_data, universe[0], universe[1]) #Consistency: clip to universe
    return to_return

def plan(input:AlgorithmParameters):
    outputs = _variance_aware_mean_pair_sample_quantile(input, input.rho*input.recenter_and_std_budget_proportion)
    return _topk_method(input, input.rho*(1-input.recenter_and_std_budget_proportion), outputs.center_predictions, outputs.std_predictions)

def plan_noscale(input:AlgorithmParameters):
    outputs = _variance_aware_mean_pair_sample_quantile(input, rho=input.rho*input.recenter_and_std_budget_proportion, estimate_std=False)
    return _topk_method(input, input.rho * (1 - input.recenter_and_std_budget_proportion), outputs.center_predictions, np.zeros(shape=input.dimensions))

def _variance_aware_mean_pair_sample_quantile(input:AlgorithmParameters, rho, estimate_std=True):
    data = _plan_preprocess_data(input.data, input.alpha, input.universe) if input.alpha>0 else input.data
    std_upper_bound = 0.5*(input.universe[1]-input.universe[0])**2

    search_budget_allocation = rho/(input.dimensions)
   
    center_prop=input.recenter_budget_proportion if estimate_std else 1

    if input.alpha>0:
        search_budget_allocation/=3 #We've increased sensitivity from 1 to 3 by adding two elements
    center_prediction_budget = search_budget_allocation * center_prop
    std_prediction_budget = search_budget_allocation*(1-center_prop)

    if input.kaplan:
        center_predictions = np.apply_along_axis(lambda x: kaplan_quantile(x, input.universe, 0.5, center_prediction_budget, None), 0, data)
    else:
        center_predictions = quantile_binary_search_mp(data, input.dimensions, 0.5*len(data), input.universe[1], center_prediction_budget, T=input.step, l=input.universe[0])

    if not estimate_std:
        return Predictions(center_predictions, np.ones(shape=input.dimensions))

    # Shuffle elements before we pair them to prevent attacks on accuracy
    rng = np.random.default_rng()
    rng.shuffle(data)
    odd, even = data[::2], data[1::2]
    pairwise = np.array([0.5* (x-y)**2 for (x, y) in zip(odd, even)])
    # Group into list of tuples to sum
    groups = list(zip(*[iter(pairwise)]*input.k_groups))
    # Sum and divide by k
    robust_std_estimates = np.array(list(map(lambda s: s/input.k_groups, map(sum, groups))))
    if input.fill and input.k_groups > 1:
        for _ in range(2 * input.k_groups - 1):
            rng.shuffle(data)
            odd, even = data[::2], data[1::2]
            pairwise = np.array([0.5* (x-y)**2 for (x, y) in zip(odd, even)])
            # Group into list of tuples to sum
            groups = list(zip(*[iter(pairwise)]*input.k_groups))
            # Sum and divide by k
            robust_std_estimates = np.concatenate((robust_std_estimates,
                np.array(list(map(lambda s: s/input.k_groups, map(sum, groups))))))
        std_prediction_budget /= 2 * input.k_groups


    if input.kaplan:
        var_predictions = np.apply_along_axis(lambda x: kaplan_quantile(x, (0, std_upper_bound), 0.5, std_prediction_budget, None), 0, robust_std_estimates)
    else:
        var_predictions = np.array(quantile_binary_search_mp(robust_std_estimates,input.dimensions, .5 * len(robust_std_estimates), std_upper_bound, std_prediction_budget, T=input.step, l=0))

    if input.chisquared:
        var_predictions = var_predictions/(((1-(2/(9*input.k_groups)))**3))

    std_predictions = np.sqrt(var_predictions)

    vars_to_print = 32
    offset = 28*14
    real_vars = np.var(input.data, axis=0)
    robust_mean =  np.mean(robust_std_estimates, axis=0)
    robust_median = np.median(robust_std_estimates, axis=0)
    assert real_vars.shape == var_predictions.shape
    assert real_vars.shape == robust_median.shape
    if DEBUG:
        print("Real var \n", real_vars[:vars_to_print])
        print("K_GROUP initial means are\n",robust_mean[:vars_to_print])
        print("K_GROUP initial medians are\n", robust_median[:vars_to_print])
        print("Predicted var \n", var_predictions[:vars_to_print])
        print("Error ||median - mean||", np.linalg.norm(robust_mean[:vars_to_print] - robust_median[:vars_to_print]))
        print("Error ||DP median - median||", np.linalg.norm(robust_median[:vars_to_print] - var_predictions[:vars_to_print]))

        print("Real center var \n", real_vars[offset:offset+vars_to_print])
        print("K_GROUP center means are\n",robust_mean[offset:offset+vars_to_print])
        print("K_GROUP center medians are\n", robust_median[offset:offset+vars_to_print])
        print("Predicted center var \n", var_predictions[offset:offset+vars_to_print])
        print("Error ||median - mean||", np.linalg.norm(robust_mean[offset:offset + vars_to_print] - robust_median[offset:offset + vars_to_print]))
        print("Error ||DP median - median||", np.linalg.norm(robust_median[offset:offset + vars_to_print] - var_predictions[offset:offset + vars_to_print]))

    return Predictions(center_predictions, std_predictions)

def _topk_method(input:AlgorithmParameters, rho, center_predictions, std): #Scale first, then clip to top-k
    n = len(input.data)

    # Re-center around (approximately) 0
    shifted_data = input.data - center_predictions

    # Scale
    std = np.clip(std, MIN_VALUE, None) # Defensive: make sure std isn't negative
    Delta = input.max_std_distance * std
    if input.regularization:
        Delta += np.linalg.norm(Delta, ord=1) / input.dimensions

    B = 1 / np.sqrt(Delta)
    scaled_data = shifted_data * B # Multiply each row vector with individual scalar values

    # Find top-k quantile
    top_k_budget = input.top_k_budget_proportion*rho
    norms = np.linalg.norm(scaled_data, axis=1)
    universe_upper_clipping = np.sqrt(np.log(input.dimensions)*np.log(1/input.beta)*np.linalg.norm(std, ord=1))
    universe_lower_clipping = 0

    # Calculate where to clip
    if(input.k > 0):
        k = input.k
    else:
        T = np.log(input.universe[1]-input.universe[0])
        if DEBUG: print('T:', T)
        k = ((np.sqrt(n)+(np.sqrt(T/2 * np.log(T/input.beta)/top_k_budget)))/n)

    if DEBUG: print('k=', k)
    if input.kaplan:
        if DEBUG: print('\n\nClipping at {k}\nn={n}, rho={rho}, d={d} \n\n'.format(k=(1-k), n=n, rho=top_k_budget, d=input.dimensions))
        top_k_length = np.apply_along_axis(lambda x: kaplan_quantile(x, (universe_lower_clipping,universe_upper_clipping), (1-k), top_k_budget, None), 0, norms)
    else:
        top_k_length = quantile_binary_search(norms, (1-k)*n, universe_upper_clipping, top_k_budget, l=universe_lower_clipping, T=input.step)
    rho = rho-top_k_budget

    # Clip x to fit in ellipsoid, based on top-k quantile
    clipped_data = clip_to_ellipsoid(scaled_data, top_k_length)

    # Calculate noisy mean
    means = np.mean(clipped_data,axis=0)
    noise = np.random.normal(loc=0, scale=np.sqrt(2*top_k_length**2/rho) * 1/n, size=input.dimensions) # Gaussian noise scaled to sum to 1
    noisy_means = means + noise
    unclipped_noisy_means = np.mean(shifted_data * B,axis=0) + noise

    # Scale back
    mean_estimations = (noisy_means * 1 / B) + center_predictions
    unclipped_mean_estimations = (unclipped_noisy_means * 1 / B) + center_predictions
    return Results(mean_estimations, unclipped_mean_estimations)

def clip_to_vector(inputs, vector, scale):
    inputs = np.clip(inputs, -vector * scale, vector * scale)
    return inputs

def clip_to_ellipsoid(inputs, C):
    x_norm = np.linalg.norm(inputs, axis=1)
    scale = C/x_norm
    scale[scale > 1.0] = 1.0 #Inplace adjust scale, i.e. min(C/x_norm,1.0)
    return np.apply_along_axis(lambda x: scale*x, 0, inputs) #Scale each dimension accordingly


def binary_mean_only(data, N, D, rho):
    from collections import Counter

    # compute ground truth means
    c = Counter()
    for x in data:
        c.update(x)

    gt_p = np.array([c[i] / N for i in range(D)])
    B = np.ones(shape=D)

    mean_budget = rho * 1/D
    noise = np.random.normal(scale=np.sqrt(1/(2 * mean_budget)) * 1/N, size=D)


def plan_binary(data, N, D, rho, prop=0.1, method="paper"):
    from collections import Counter

    # compute ground truth means
    c = Counter()
    for x in data:
        c.update(x)

    gt_p = np.array([c[i] / N for i in range(D)])
    B = np.ones(shape=D)

    clipping_prop = 1/D

    rho1 = prop * rho
    rho -= rho1
    rho2 = clipping_prop * rho
    rho3 = (1-clipping_prop) * rho


    # compute noisy means and set up budget
    if method == "oracle":
        var = (gt_p) * (1 - gt_p)
        std = np.sqrt(var)
        std += np.linalg.norm(std, ord=1) / D
        var = std**2
        B = var**(-1/3)
    elif method == "paper":
        mean_budget = rho1/D
        noise = np.random.normal(scale=np.sqrt(1/(2 * mean_budget)) * 1/N, size=D)
        noisy_means = np.clip(gt_p + noise, 1/D**.4, 1-1/D**.4)
        var = ((1 - noisy_means) * noisy_means)#**.5
        std = np.sqrt(var)
        if DEBUG:
            print("Estimation error:", np.linalg.norm(std -
                np.clip(np.sqrt(gt_p * (1-gt_p)), 1/D**.4, 1-1/D**.4)))
        std += np.linalg.norm(std, ord=1) / D
        B = var**(-1/3)
    elif method == "noscale":
        B = np.ones(shape=D) # repeating to be explicit
    else:
        raise NotImplementedError()

    # scale each element so that x_i \in {0, B_i}
    vecs = [[(i, B[i]) for i in vec] for vec in data]

    # compute norms and select quantile
    norms = []
    for vec in vecs:
        norm = 0
        for i, val in vec:
            norm += val**2
        norms.append(norm**.5)

    n = len(norms)
    T = 10
    quantile = (n - n**.5  - (T * np.log(T)/rho2)**.5)/n
    if DEBUG: print(quantile, np.quantile(norms, quantile))
    C = quantile_binary_search(norms, quantile * n, u=np.linalg.norm(B), l=0, p=rho2)
    if DEBUG: print("Chose", C, " as clipping threshold")

    # normalize data
    normalized_data = []
    for i, vec in enumerate(vecs):
        s = 1
        if norms[i] > C:
            s = C / norms[i]
        normalized_data.append([(j, val * s) for (j, val) in vec])

    # compute means
    means = np.zeros(shape=D)

    for vec in normalized_data:
        for i, val in vec:
            means[i] += val

    # noise means and scale back
    noise = np.random.normal(loc=0., scale = np.sqrt(2 * C**2 / rho3), size=D)
    output = means + noise
    output = output / (B * N)

    unclipped_means = np.zeros(shape=D)

    for vec in vecs:
        for i, val in vec:
            unclipped_means[i] += val

    unclipped_output = (unclipped_means + noise) / (B * N)

    return Results(output, unclipped_output)

