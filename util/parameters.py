from dataclasses import dataclass
import numpy as np

@dataclass
class AlgorithmParameters:
    data: np.ndarray
    dimensions: int 
    universe: tuple 
    rho: float 
    alpha: float
    variance: float
    step: int
    k : float
    cov : np.ndarray
    recenter_and_std_budget_proportion: float
    recenter_budget_proportion: float
    top_k_budget_proportion: float
    fill: bool
    regularization: bool
    k_groups: int
    max_std_distance: float
    kaplan: bool
    chisquared: bool
    beta: float

@dataclass
class ErrorParameters:
    n: int
    dimensions: int
    rho: float
    sigmas: list

@dataclass
class Predictions:
    center_predictions: list
    std_predictions: np.ndarray

@dataclass
class Results:
    mean: np.ndarray
    unclipped_mean: np.ndarray

@dataclass
class ExperimentOutput:
    label: str
    input: AlgorithmParameters
    mean: float
    cov_strategy: str
    number_small_values: int
    run_index: int
    error: np.ndarray

@dataclass
class ExperimentSettings:
    step: int
    recenter_and_std_budget_proportion: float
    recenter_budget_proportion: float
    top_k_budget_proportion: float
    k: list
    preprocess_variance: float
    preprocess_alpha: float
    runs: int
    dimensions: list
    ns: list
    rhos: list
    beta: float
    universes: list
    strategies: list
    rs: list
    chisquared: bool
    statistical_means: list
    covariance_matrices: list
    correlation: float