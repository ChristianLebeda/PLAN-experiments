# PLAN experiments

This repository contains the source code used for the experiments in the paper:

Martin Aumüller, Christian Janos Lebeda, Boel Nelson, and Rasmus Pagh, "PLAN: Variance-Aware Differentially Private Mean Estimation".

## Dependencies

We re-use source code from two other projects.

Our quantile estimation is based on an algorithm from:  

H. Kaplan, S. Schnapp, U. Stemmer, "Differentially Private Approximate Quantiles".

We used their source code available here: https://github.com/ShacharSchnapp/DP_AQ/blob/master/algorithms/single_quantile_algo.py

We compare our mechanism with the mechanism from:

Ziyue Huang, Yuting Liang, Ke Yi, "Instance-optimal Mean Estimation Under Differential Privacy".

The source code is available as part of their supplementary material here: https://proceedings.neurips.cc/paper/2021/hash/da54dd5a0398011cdfa50d559c2c0ef8-Abstract.html

We include their implementation because we had to modify parts of their code as discussed in our paper.

To install dependencies using miniconda run 
```
$ install.sh
```

After installing use the following command
```
$ source miniconda/bin/activate plan
```

## Running experiments

Running an experiment using the commands below automatically creates an output folder. The result of the experiments are saved to a .csv file.

For the baseline experiments on Gaussian data, run 

```
$ python simulations/baseline_comparison.py experiment [algorithms ...]
````

Where the experiment is a string corresponding to one of the experiments used in the paper:  
GaussianA, GaussianB, GaussianC recreates the experiments from Figure 3.  
GaussianAlow, GaussianBlow, GaussianClow recreates the experiments from Figure 4 with small rho values.

[algorithms ...] should be replaced with a list of methods used for mean estimation. PLAN runs the standard algorithm as discussed in the experiments section of the paper. PLAN-noscale runs PLAN without the scaling step. PLAN-binsearch runs PLAN using private binary search instead of the exponential mechanism for PrivQuantile. The binsearch variant was not used for any of the final experiments in the paper. IOME runs the IOME baseline.

As an example, to run the experiment from Figure 8 (a) use

```
$ python simulations/baseline_comparison.py GaussianA PLAN PLAN-noscale IOME
````

For the experiments on binary data, run 

```
$ python simulations/binary_experiments.py [arguments]
```

Here arguments are of two types. The order of arguments does not matter.

The arguments specify the datasets used for experiments and should be one or more of kosarak, POS, or synthetic.  

The arguments also specify the algorithms used. PLAN runs the standard PLAN algorithm, PLAN-noscale runs PLAN without scaling as in Figure 9 from the paper. PLAN-oracle is a baseline that uses the empirical variances directly instead of privately estimating them. IOME is used for the IOME baseline.

To run all experiments for binary data from the paper use 
```
$ python simulations/binary_experiments.py kosarak POS synthetic PLAN PLAN-noscale IOME
```

Running our baseline IOME uses a lot of memory for some of the experiments. In particular, running IOME on the sparse binary datasets, requires a machine with at least 128 GB RAM. 
When running the binary experiments without IOME 8GB of RAM are sufficient.
All experiments for the variants of PLAN uses less than 10GB of RAM.

The number of repetitions can be changed by changing the 'runs' variable. This requires a change to the source files. The current setting is 50 repetitions.


## Producing plots

Plots are generated using the interactive jupyter notebook `eval/pretty_plot.ipynb`.

To use the notebook, load the environment and start jupyter:
```
$ source miniconda/bin/activate plan
$ jupyter notebook
```

These commands will open a browser tab at `http://localhost:8888/tree`. You can now navigate to the `pretty_plot.ipynb` which is in the folder `eval`. 


The notebook first outputs a plot for a selected Gaussian case, and then produces plots for the Binary case, signposted by the heading "The Binary Case". Given that `csv` files produced by the experiments are available in the expected locations, executing the entire notebook should produce plots paths on the format:
```
/experiments_results/{case}/{case}.pdf'
```

In the 4th notebook cell (marked with the comment `Edit me!`) it is possible to specify which plots to produce. By editing the value of the variable `case` the axes will be updated to the corresponding experiments for the Gaussian experiments. Supported values for `case` are:
```
'gaussianA'
'gaussianB'
'gaussianC'
```

Similiarly the Binary experiments are controlled by the variable `bincase`. Supported values:
```
'POS'
'kosarak'
```

Note that the notebook assumes that `.csv` files produced by experiments are stored in a folder `/experiments_results` as follows:
```
/experiments_results/{case}/dataframe.csv
```


