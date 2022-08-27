![svm](doc/fig/circles_margin.png)
# Motivation
An iterative algorithm that makes use of elementary geometry to find the maximal margin hyperplane between two classes.

- provides weight and bias that defines the optimum-hyperplane and also gives the support vectors.
- applicable only for linearly separable problem. i.e., not suitable for soft-marging classification
- if the data is not separable, the algorithm quickly figure out this and stops further iterations. 
- currently support binary classification of labelled data with number of features greater than 1.

# Usage

```python
from gsvm.classifier import GeometricClassifier
clf = GeometricClassifier()
clf.fit(X, label)
```

`X` is a multi-dimensional array where rows indicating samples and columns indicating features. Each row (sample) is associated with a class-label. `label` can be either a list (of str, number, or bool) or an array with size equal to the number of rows in `X`. 

Optimal hyperplane (if exist) is uniquely determned by `<weight, x> + bias = 0`, where `<,>` denotes the scalar product and `x` is an arbitrary vector.

#### Useful attributes: 
```python
clf.coef_             # weight
clf.intercept_        # bias
clf.support_vectors_  # support vectors  i.e., x satisfying  <weight, x> + bias = 1 or -1
clf.support_          # indices of the support vectors in X
clf.width_            # width of the margin
clf.message           # string message indicating success or failure of the computation.
```

[Sample output](sample_output.md)
