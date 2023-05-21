Principal Components Features
=============================
The features of this group try to describe the variable scaling of a continuous problem.
This is done by applying a Principal Component Analysis (either based on the covariance or the correlation matrix) to the data.
Here, “data” can either be the entire initial design or just the decision space (i.e. the initial design without the objective).

Based on these four approaches, the features describe the (relative) amount of principal components that are required to explain a certain amount of variability (the default is 0.9) of the problem.
The higher the ratio is, the more principal components are required. Apart from those features, the importance of the first principal component is measured as well.

For a complete overview of the features, please refer to the documentation of :func:`pflacco.classical_ela_features.calculate_pca`.