Dispersion Features
===================
The dispersion features compare the dispersion, i.e. the (aggregated) pairwise distances, of all points in the initial design with the dispersion among the best points in the initial design.
Per default, this set of “best points” is based on the 2%, 5% and 10% quantile of the objectives. Those dispersions are then compared based on the ratio as well as on the difference.
For a complete overview of the features, please refer to the documentation of :func:`pflacco.classical_ela_features.calculate_dispersion` and the work of Lunacek and Whitley (2014) [#r1]_.

Below you find a code example.

.. code-block:: python3

   from pflacco.sampling import create_initial_sample
   from pflacco.classical_ela_features import calculate_dispersion

   # Arbitrary objective function
   def objective_function(x):
      return sum(x**2)

   dim = 3
   # Create inital sample using latin hyper cube sampling
   X = create_initial_sample(dim, sample_type = 'lhs')
   # Calculate the objective values of the initial sample
   # using an arbitrary objective function
   y = X.apply(lambda x: objective_function(x), axis = 1)

   # Compute disp feature set from the convential ELA features
   ic = calculate_dispersion(X, y)

.. rubric:: Literature Reference

.. [#r1] Lunacek, M. and Whitley, D. (2014), “The Dispersion Metric and the CMA Evolution Strategy”, in Proceedings of the 8th Annual Conference on Genetic and Evolutionary Computation, pp. 477—484, ACM (http://dx.doi.org/10.1145/1143997.1144085).