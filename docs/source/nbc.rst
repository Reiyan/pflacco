Nearest Better Features
#######################
The Nearest-Better Features — also called *Nearest-Better Clustering (NBC)* Features — are based on a heuristic, which recognizes single peaks in a multimodal landscape [#r1]_.
In general, the features are computed on two sets of distances: the distances from each point to its nearest neighbor and the distances from each point to its nearest-better neighbor.
Here, the latter one is the closest observation (w.r.t. the reference point) with a better objective value than the reference point.

Based on these two distance sets, this feature set computes five NBC features. A further motivation of these features can be found, amongst others, in Kerschke et al. (2015) [#r2]_
as well as in the technical documentation of :func:`pflacco.classical_ela_features.calculate_nbc`.
Below you find a code example.

.. code-block:: python3

   from pflacco.sampling import create_initial_sample
   from pflacco.classical_ela_features import calculate_nbc

   # Arbitrary objective function
   def objective_function(x):
      return sum(x**2)

   dim = 3
   # Create inital sample using latin hyper cube sampling
   X = create_initial_sample(dim, sample_type = 'lhs')
   # Calculate the objective values of the initial sample
   # using an arbitrary objective function
   y = X.apply(lambda x: objective_function(x), axis = 1)

   # Compute nbc feature set from the convential ELA features
   ic = calculate_nbc(X, y)

.. rubric:: Literature Reference

.. [#r1] Preuss, M. (2012), “Improved Topological Niching for Real-Valued Global Optimization”, in Applications of Evolutionary Computation, pp. 386—395, Springer (http://dx.doi.org/10.1007/978-3-642-29178-4_39).
.. [#r2] Kerschke, P. et al. (2015), “Detecting Funnel Structures by Means of Exploratory Landscape Analysis”, in Proceedings of the 17th Annual Conference on Genetic and Evolutionary Computation (GECCO ‘15), pp. 265-272, ACM (http://dx.doi.org/10.1145/2739480.2754642).