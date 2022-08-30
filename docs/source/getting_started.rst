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