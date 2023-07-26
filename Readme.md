![Github Status](https://github.com/reiyan/pflacco/actions/workflows/python-package.yml/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/pflacco/badge/?version=latest)](https://pflacco.readthedocs.io/en/latest/?badge=latest)
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

## FAQ
- For some (very few) features the values for the same sample differ between pflacco and flacco:
This is a known occurence. The differences can be traced back to the underlying methods to calculate the features. For example, ```ela_meta``` relies on linear models. The method to construct a linear model in R is based on qr (quantile regression) whereas the ```LinearModel()``` in scikit-learn uses the conventional OLS method. For a large enough sample, there is no statistical difference. However, to keep this consistent between programming language this issue will be addressed in future version.

- What is the difference between 0.* and 1.* version of pflacco?
The 0.* version of pflacco provided a simple interface to the programming language R and calculated any landscape features using the R-package flacco. While this is convenient for me as a developer, the downside is that the performance of pflacco is horrendous. Hence, the >=1.* releases of pflacco offer an implementation of almost all features of the R-package flacco in native python. Thereby, the calculation of features is expedited by an order of magnitude.

- Is it possible to calculate landscape features for CEC or Nevergrad?
Generally speaking, this is definitely possible. However, to the best of my knowledge, Nevergrad does not offer a dedicated API to query single objective functions and the CEC benchmarks are mostly written in C or Matlab.
Some CEC benchmarks have an unofficial Python wrapper (which is not kept up to date) like [CEC2017](https://github.com/lacerdamarcelo/cec17_python). These require additional compiling steps to run any of the functions.

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

It is also possible to include objective functions provided by other packages such as ```COCO``` and ```ioh```.

**Note that these packages do not always pandas dataframes as input. Hence, sometimes it is necessary to transform the initial sample X to a numpy array**

## COCO Example
In order for the following code snippet to work, you have install [coco](https://github.com/numbbo/coco) first (which is **not** possible via pip/conda).
This code snippet calculates the specified landscape features for the well-known single-objective noiseless Black-Box Optimization Benchmark (BBOB).
The optimization problems are comprised of all 24 functions in dimensions 2 and 3 for the first five instances.
```python
import cocoex
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample

features = []
# Get all 24 single-objective noiseless BBOB function in dimension 2 and 3 for the first five instances.
suite = cocoex.Suite("bbob", f"instances:1-5", f"function_indices:1-24 dimensions:2,3")
for problem in suite:
    dim = problem.dimension
    fid = problem.id_function
    iid = problem.id_instance

    # Create sample
    X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5)
    y = X.apply(lambda x: problem(x), axis = 1)

    # Calculate ELA features
    ela_meta = calculate_ela_meta(X, y)
    ela_distr = calculate_ela_distribution(X, y)
    nbc = calculate_nbc(X, y)
    disp = calculate_dispersion(X, y)
    ic = calculate_information_content(X, y, seed = 100)

    # Store results in pandas dataframe
    data = pd.DataFrame({**ic, **ela_meta, **ela_distr, **nbc, **disp, **{'fid': fid}, **{'dim': dim}, **{'iid': iid}}, index = [0])
    features.append(data)

features = pd.concat(features).reset_index(drop = True)
```

## IOH Example
Similar to the example above, this code snippet calculates the specified landscape features for the well-known single-objective noiseless Black-Box Optimization Benchmark (BBOB).
The optimization problems are comprised of all 24 functions in dimensions 2 and 3 for the first five instances.
In constrast to ```coco```, ```ioh``` can be installed via pip/conda and offers other benchmark problems. See the respective [documentation](https://iohprofiler.github.io/IOHexperimenter/python/problem.html) for more details.
```python
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample
from ioh import get_problem, ProblemType

features = []
# Get all 24 single-objective noiseless BBOB function in dimension 2 and 3 for the first five instances.
for fid in range(1,25):
    for dim in [2, 3]:
        for iid in range(1, 6):
            # Get optimization problem
            problem = get_problem(fid, iid, dim, problem_type = ProblemType.BBOB)

            # Create sample
            X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5)
            y = X.apply(lambda x: problem(x), axis = 1)

            # Calculate ELA features
            ela_meta = calculate_ela_meta(X, y)
            ela_distr = calculate_ela_distribution(X, y)
            ela_level = calculate_ela_level(X, y)
            nbc = calculate_nbc(X, y)
            disp = calculate_dispersion(X, y)
            ic = calculate_information_content(X, y, seed = 100)

            # Store results in pandas dataframe
            data = pd.DataFrame({**ic, **ela_meta, **ela_distr, **nbc, **disp, **{'fid': fid}, **{'dim': dim}, **{'iid': iid}}, index = [0])
            features.append(data)

features = pd.concat(features).reset_index(drop = True)
```

## Citation
If you are using pflacco in any capacity, I would appreciate a citation. You can use the following bibtex:
```
@article{10.1162/evco_a_00341,
    author = {Prager, Raphael Patrick and Trautmann, Heike},
    title = "{Pflacco: Feature-Based Landscape Analysis of Continuous and Constrained Optimization Problems in Python}",
    journal = {Evolutionary Computation},
    pages = {1-25},
    year = {2023},
    month = {07},
    abstract = "{The herein proposed Python package pflacco provides a set of numerical features to characterize single-objective continuous and constrained optimization problems. Thereby, pflacco addresses two major challenges in the area optimization. Firstly, it provides the means to develop an understanding of a given problem instance, which is crucial for designing, selecting, or configuring optimization algorithms in general. Secondly, these numerical features can be utilized in the research streams of automated algorithm selection and configuration. While the majority of these landscape features is already available in the R package flacco, our Python implementation offers these tools to an even wider audience and thereby promotes research interests and novel avenues in the area of optimization.}",
    issn = {1063-6560},
    doi = {10.1162/evco_a_00341},
    url = {https://doi.org/10.1162/evco\_a\_00341},
    eprint = {https://direct.mit.edu/evco/article-pdf/doi/10.1162/evco\_a\_00341/2148122/evco\_a\_00341.pdf},
}
```

## Documentation
A comprehensive documentation can be found [here](https://pflacco.readthedocs.io/en/latest/index.html).

## Contact
I endorse and appreciate every comment and participation. Feel free to open an issue here on GitHub or contact me under raphael.prager@uni-muenster.de
