# pflacco: The R-package flacco in native Python code
For people who are not comfortable with R.

## Summary
Feature-based landscape analysis of continuous and constrained optimization problems is now available in Python as well.
This package provides a python interface to the R package [flacco](https://github.com/kerschke/flacco) by Pascal Kerschke in version v0.4.0.
And now it also provides a native Python implementation with additional features such as:
- [Features for exploiting black-box optimization problem structure](https://pure.itu.dk/ws/files/76529050/bbo_lion7.pdf).
- [Ruggedness, funnels and gradients in fitness landscapes and the effect on PSO performance](https://ieeexplore.ieee.org/abstract/document/6557671).
- [Global characterization of the CEC 2005 fitness landscapes using fitness-distance analysis](https://publications.mpi-cbg.de/Müller_2011_5158.pdf).
- [Analysing and characterising optimization problems using length scale](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.9948&rep=rep1&type=pdf).
- [Fitness Landscape Analysis Metrics based on Sobol Indices and Fitness-and State-Distributions.](https://ieeexplore.ieee.org/document/9185716).
- [Local optima networks for continuous fitness landscapes](https://dl.acm.org/doi/10.1145/3319619.3326852)

The following is the description of the original flacco package:
> flacco is a collection of features for Explorative Landscape Analysis (ELA) of single-objective, continuous (Black-Box-)Optimization Problems. It allows the user to quantify characteristics of an (unknown) optimization problem's landscape.
>
> Features, which used to be spread over different packages and platforms (R, Matlab, Python, etc.), are now combined within this single package. Amongst others, this package contains feature sets, such as ELA, Information Content, Dispersion, (General) Cell Mapping or Barrier Trees.
>
> Furthermore, the package provides a unified interface for all features -- using a so-called feature object and (if required) control arguments. In total, the current release (1.7) consists of 17 different feature sets, which sum up to approximately 300 features.
>
> In addition to the features themselves, this package also provides visualizations, e.g. of the cell mappings, barrier trees or information content

The calculation procedure and further background information of ELA features is given in [Comprehensive Feature-Based Landscape Analysis of Continuous and Constrained Optimization Problems Using the R-Package flacco](https://arxiv.org/abs/1708.05258).
## Prerequisites
For a stable (and tested) outcome, pflacco requires at least [Python>=3.8](https://www.python.org/downloads/release/python-364/)

## Setup
Easy as it usually is in Python:
```bash
python -m pip install pflacco
```

## Quickstart
```python
from pflacco.sampling import create_initial_sample

from pflacco.classical_ela_features import calculate_ela_distribution
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.local_optima_network_features import compute_local_optima_network, calculate_lon_features

# Arbitrary objective function
def objective_function(x):
    return x[0]**2 - x[1]**2

dim = 2
# Create inital sample using latin hyper cube sampling
X = create_initial_sample(dim, sample_type = 'lhs')
# Calculate the objective values of the initial sample using an arbitrary objective function (here y = x1^2 - x2^2)
y = X.apply(lambda x: objective_function(x), axis = 1)

# Compute an exemplary feature set from the convential ELA features of the R-package flacco
ela_distr = calculate_ela_distribution(X, y)
print(ela_distr)

# Compute an exemplary feature set from the novel features which are not part of the R-package flacco yet.
fdc = calculate_fitness_distance_correlation(X, y)
print(fdc)

# Compute a Local Optima Network (LON). From this network, LON features can be calculated.
nodes, edges = compute_local_optima_network(f=objective_function, dim=dim, lower_bound=0, upper_bound=1)
lon = calculate_lon_features(nodes, edges)
print(lon)

```

## Documentation
A comprehensive documentation can be found [here](https://pflacco.readthedocs.io/en/latest/index.html).

## Contact
I endorse and appreciate every comment and participation. Feel free to contact me under raphael.prager@uni-muenster.de
