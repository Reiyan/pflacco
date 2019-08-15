# pflacco: A Python Interface of the R Package flacco
For people who are not comfortable with R.

## Summary
Feature-based landscape analysis of continuous and constrained optimization problems is now available in Python as well.
This package provides a python interface to the R package [flacco](https://github.com/kerschke/flacco) by Pascal Kerschke.
The following is the description of the original flacco package:
> flacco is a collection of features for Explorative Landscape Analysis (ELA) of single-objective, continuous (Black-Box-)Optimization Problems. It allows the user to quantify characteristics of an (unknown) optimization problem's landscape.
>
> Features, which used to be spread over different packages and platforms (R, Matlab, python, etc.), are now combined within this single package. Amongst others, this package contains feature sets, such as ELA, Information Content, Dispersion, (General) Cell Mapping or Barrier Trees.
>
> Furthermore, the package provides a unified interface for all features -- using a so-called feature object and (if required) control arguments. In total, the current release (1.7) consists of 17 different feature sets, which sum up to approximately 300 features.
>
> In addition to the features themselves, this package also provides visualizations, e.g. of the cell mappings, barrier trees or information content

## Setup
Easy as it usually is in Python:
```bash
python -m pip install flacco
```

## Quickstart
```python
from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features

# Arbitrary objective function
def objective_function(x, dim):
    return [entry[0]**2 - entry[1]**2 for entry in x]


# Create inital sample using latin hyper cube sampling
sample = create_initial_sample(100, 2, type = 'lhs')
# Calculate the objective values of the initial sample using an arbitrary objective function (here y = x1^2 - x2^2)
obj_values = objective_function(sample, 2)
# Create feature object
feat_object = create_feature_object(sample, obj_values, blocks=3)

# Calculate a single feature set
cm_angle_features = calculate_feature_set(feat_object, 'cm_angle')
print(cm_angle)

# Calculate all features
ela_features = calculate_features(feat_object)
print(ela_features)
```

