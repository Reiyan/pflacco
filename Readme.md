# pflacco: A Python Interface of the R Package flacco

## Summary
Feature-based landscape analysis of continuous and constrained optimization problems is now available in Python as well.


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
cm_angle = calculate_feature_set(feat_object, 'cm_angle')
print(cm_angle)

# Calculate all features
ela_features = calculate_features(feat_object)
print(ela_features)
```
## Setup


This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.