
import numpy as np

# Original source file from here:
# https://github.com/ShacharSchnapp/DP_AQ/blob/master/algorithms/single_quantile_algo.py

# Our wrapper method for the quantile estimation
def kaplan_quantile(sorted_array, bounds, quantile, rho, swap):
  sorted_array.sort() #Defensive: don't assume input is already sorted
  assert 0.0 <= quantile <= 1.0
  epsilon = np.sqrt(8*rho)
  return _single_quantile(sorted_array, bounds, quantile, epsilon, swap)

def racing_sample(log_terms):
    """Numerically stable method for sampling from an exponential distribution.
    Args:
      log_terms: Array of terms of form log(coefficient) - (exponent term).
    Returns:
      A sample from the exponential distribution determined by terms. See
      Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
      Utility Function Family for the Exponential Mechanism"
      (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
      analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithms.
    """
    return np.argmin(np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)

def _single_quantile(sorted_array, bounds, quantile, epsilon, swap):
    a, d = bounds
    n = len(sorted_array)
    sorted_array = np.clip(sorted_array, a, d)
    sorted_array = np.concatenate(([a], sorted_array, [d]))
    # intervals = np.diff(sorted_array)
    # Change from original source: We added clipping to avoid division by zero
    intervals = np.clip(np.diff(sorted_array), 1e-30, None)

    sensitivity = max(quantile, 1 - quantile)
    if swap:
        sensitivity = 1.0

    utility = -np.abs(np.arange(0, n + 1) - (quantile * n))
    idx_left = racing_sample((np.log(intervals) + (epsilon / (2.0 * sensitivity) * utility)))
    return np.random.uniform(sorted_array[idx_left], sorted_array[idx_left + 1])