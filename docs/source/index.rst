.. pflacco documentation master file, created by
   sphinx-quickstart on Fri Aug 12 01:47:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of pflacco
===================================

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
* `Local optima networks for continuous fitness landscapes <https://dl.acm.org/doi/10.1145/3319619.3326852>`

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


Contents
--------

.. toctree::

   installation
   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`