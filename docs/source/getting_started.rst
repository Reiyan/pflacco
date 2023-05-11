Getting Started
===============

Pflacco offers a variety of different landscape features. These are compartmentalized into three submodules:

* :ref:`Conventional Exploratory Landscape Features <pflacco.classical_ela_features>`
* :ref:`Local Optima Networks <pflacco.local_optima_network_features>`
* :ref:`Miscellanous features <pflacco.misc_features>`, which are the product of recent advances in the research community.

The following code example provides a demonstration of the different submodules.

.. code-block:: python3

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
   # Calculate the objective values of the initial sample
   # using an arbitrary objective function (here y = x1^2 - x2^2)
   y = X.apply(lambda x: objective_function(x), axis = 1)

   # Compute an exemplary feature set from the convential ELA features
   # of the R-package flacco
   ela_distr = calculate_ela_distribution(X, y)
   print(ela_distr)

   # Compute an exemplary feature set from the novel features
   # which are not part of the R-package flacco yet.
   fdc = calculate_fitness_distance_correlation(X, y)
   print(fdc)

   # Compute a Local Optima Network (LON). From this network, LON features can be calculated.
   nodes, edges = compute_local_optima_network(f=objective_function, dim=dim, lower_bound=0, upper_bound=1)
   lon = calculate_lon_features(nodes, edges)
   print(lon)

It is also possible to include objective functions provided by other packages such as ``COCO`` and ``ioh``.

**Note that these packages do not always pandas dataframes as input. Hence, sometimes it is necessary to transform the initial sample X to a numpy array**

COCO Example
------------
In order for the following code snippet to work, you have install `coco <https://github.com/numbbo/coco>`_ first (which is **not** possible via pip/conda).
This code snippet calculates the specified landscape features for the well-known single-objective noiseless Black-Box Optimization Benchmark (BBOB).
The optimization problems are comprised of all 24 functions in dimensions 2 and 3 for the first five instances.

.. code-block:: python3

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


IOH Example
-----------
Similar to the example above, this code snippet calculates the specified landscape features for the well-known single-objective noiseless Black-Box Optimization Benchmark (BBOB).
The optimization problems are comprised of all 24 functions in dimensions 2 and 3 for the first five instances.
In constrast to ``coco``, ``ioh`` can be installed via pip/conda and offers other benchmark problems. See the respective `documentation <https://iohprofiler.github.io/IOHexperimenter/python/problem.html>`_ for more details.

.. code-block:: python3
   
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
