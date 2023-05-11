.. pflacco documentation master file, created by
   sphinx-quickstart on Fri Aug 12 01:47:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of pflacco
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Feature-based landscape analysis of continuous and constrained optimization problems is now available in Python as well.
This package provides a Python interface to the R package flacco by Pascal Kerschke in version v0.4.0.
And now also a native Python implementation with additional features such as:

* `Features for exploiting black-box optimization problem structure <https://pure.itu.dk/ws/files/76529050/bbo_lion7.pdf>`_
* `Ruggedness, funnels and gradients in fitness landscapes and the effect on PSO performance <https://ieeexplore.ieee.org/abstract/document/6557671>`_
* `Global characterization of the CEC 2005 fitness landscapes using fitness-distance analysis <https://publications.mpi-cbg.de/Müller_2011_5158.pdf>`_
* `Analysing and characterising optimization problems using length scale <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.9948&rep=rep1&type=pdf>`_
* `Fitness Landscape Analysis Metrics based on Sobol Indices and Fitness-and State-Distributions <https://ieeexplore.ieee.org/document/9185716>`_
* `Local optima networks for continuous fitness landscapes <https://dl.acm.org/doi/10.1145/3319619.3326852>`_

The conventional features are identical or equivalent to the implementation of the R-package flacco.
To directly quote the documentation of flacco:

.. epigraph::

   flacco is a collection of features for Explorative Landscape Analysis (ELA) of single-objective, continuous (Black-Box-)Optimization Problems.
   It allows the user to quantify characteristics of an (unknown) optimization problem's landscape.
   Features, which used to be spread over different packages and platforms (R, Matlab, Python, etc.), are now combined within this single package.
   Amongst others, this package contains feature sets, such as ELA, Information Content, Dispersion, (General) Cell Mapping or Barrier Trees.
   Furthermore, the package provides a unified interface for all features -- using a so-called feature object and (if required) control arguments.
   In total, the current release (1.7) consists of 17 different feature sets, which sum up to approximately 300 features.
   In addition to the features themselves, this package also provides visualizations, e.g. of the cell mappings, barrier trees or information content
   The calculation procedure and further background information of ELA features is given in
   Comprehensive Feature-Based Landscape Analysis of Continuous and Constrained Optimization Problems Using the R-Package flacco.

FAQ
---
* For some (very few) features the values for the same sample differ between pflacco and flacco:
This is a known occurence. The differences can be traced back to the underlying methods to calculate the features. For example, ``ela_meta`` relies on linear models. The method to construct a linear model in R is based on qr (quantile regression) whereas the ``LinearModel()`` in scikit-learn uses the conventional OLS method. For a large enough sample, there is no statistical difference. However, to keep this consistent between programming language this issue will be addressed in future version.

* What is the difference between 0.* and 1.* version of pflacco?
The 0.* version of pflacco provided a simple interface to the programming language R and calculated any landscape features using the R-package flacco. While this is convenient for me as a developer, the downside is that the performance of pflacco is horrendous. Hence, the >=1.* releases of pflacco offer an implementation of almost all features of the R-package flacco in native python. Thereby, the calculation of features is expedited by an order of magnitude.

* Is it possible to calculate landscape features for CEC or Nevergrad?
Generally speaking, this is definitely possible. However, to the best of my knowledge, Nevergrad does not offer a dedicated API to query single objective functions and the CEC benchmarks are mostly written in C or Matlab.
Some CEC benchmarks have an unofficial Python wrapper (which is not kept up to date) like `CEC2017 <https://github.com/lacerdamarcelo/cec17_python>`_. These require additional compiling steps to run any of the functions.

Contents
--------

.. toctree::

   installation
   feature_sets
   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`