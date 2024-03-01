import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import sys
import json
sys.path.append(".")
sys.path.append("..")
import os
from plan.mean_estimation import clip_to_vector
from quantile_binary_search.method import quantile_binary_search_mp, random_rotation_mean
from kaplan_et_al.single_quantile_algo import kaplan_quantile
from util.parameters import Results


def only_binary_quantile(data, dimensions, universe, rho, steps):
    n = len(data)
    lower_bound = universe[0]
    upper_bound = universe[1]
    center_predictions = quantile_binary_search_mp(data,dimensions, 0.5*n, upper_bound, rho/dimensions, T=steps, l=lower_bound)
    return Results(np.array(center_predictions), np.empty(0))

def only_approximate_quantile(data, dimensions, universe, rho):
    center_predictions = np.apply_along_axis(lambda x: kaplan_quantile(x, universe, 0.5, rho/dimensions, None), 0, data)
    return Results(np.array(center_predictions), np.empty(0))

def naive_global_sensitivity(data, dimensions, universe, rho, cov, max_std_distance, T=10,prop=0.25):
    n = len(data)
    bin_search_budget_allocation = prop*rho/dimensions
    lower_bound = universe[0]
    upper_bound = universe[1]

    # Re-center around (approximatley) 0
    center_predictions = quantile_binary_search_mp(data,dimensions, 0.5*n, upper_bound, bin_search_budget_allocation, T=T, l=lower_bound)
    shifted_data = data - center_predictions
    
    # Adjust remaining budget
    rho = (1 - prop) * rho
    std = np.sqrt(cov.diagonal())

    clipped_data = clip_to_vector(shifted_data, std, max_std_distance)

    sensitivity = np.linalg.norm(2 * max_std_distance * std)**2

    means = np.mean(clipped_data,axis=0)
    # Gaussian noise scaled to universe
    noisy_means = means + np.random.normal(loc=0, scale=np.sqrt(sensitivity/(2*rho) *  1/n), size=dimensions)
    
    return Results(noisy_means, np.empty(0))

# FIXME: update to compare the settings.csv as well
def find_match(frame, key, universe, n, dimension, rho, mean, cov, run, r):
    if(len(frame)>0):
        return frame.loc[(frame["Algorithm"]==key) 
                            & (frame["u"] == str(universe))
                            & (frame["n"] == n)
                            & (frame["d"] == dimension)
                            & (frame["rho"] == rho)
                            & (frame["mean"] == mean)
                            & (frame["cov"] == str(cov))
                            & (frame["r"] == r)
                            & (frame["run"] == run)
                        ]
    else:
        return pd.DataFrame()

def output_to_dict(output):
    return {'Algorithm': output.label,
                'u': output.input.universe, 
                'n': len(output.input.data), 
                'd': output.input.dimensions, 
                'rho': output.input.rho, 
                'mean': output.mean, 
                'cov': output.cov_strategy, 
                'r': output.number_small_values,
                'run': output.run_index, 
                'error': output.error}


def plot_and_save(dataframe, plot_data, settings, folder, plot_name, csv_name, settings_name, x_label, x_scale, x_column, y_scale, title):
        # Plot and save
        plot_df = dataframe[["Algorithm", "d", "n", "rho", "mean", "error", "r"]].groupby(["Algorithm", "d", "n", "rho", "mean", "r"]).mean().reset_index()
        g = sns.lineplot(data=plot_df, x=x_column, y="error", hue="Algorithm", style="Algorithm", markers=True, dashes=False, ms=10)
        
        # Error terms aren't saved in the dataframe
        for key in plot_data.keys():
            x, y = zip(*plot_data[key])
            g = sns.lineplot(ax=g,x=x, y=y, markers=True, dashes=True, lw=5, label=key)

        g.set(xscale=x_scale)
        g.set(yscale=y_scale)
        g.set(xlabel=x_label)
        g.set(ylabel='$\ell_2$ error')
        g.set(title=title)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

        # Avoid empty folder by creating last, in case run is aborted or crashes
        if not os.path.exists(folder):
            os.makedirs(folder)
        print('Saving to folder {folder}'.format(folder=folder))

        g.figure.savefig(plot_name,bbox_inches='tight', pad_inches=0.5)
        g.figure.clear()
        dataframe.to_csv(csv_name)
        # Serialize data into file:
        json.dump(settings, open(settings_name, 'w'),cls=NumpyEncoder)

# In interval [0, d]
def random_skewed_elements(max:int):
    r = np.random.choice(range(0, max+1))
    return r_skewed_elements(max, r)

# r 1s, r-1 small
def r_skewed_elements(var, d, r):
    if d==r: # Guard against special case (would give division by zero)
        return np.repeat(var, r)
    return np.concatenate([np.repeat(var, r), np.repeat((var/(d-r))**2, d-r)])

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)