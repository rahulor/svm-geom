import numpy as np
from itertools import combinations
from itertools import product

def label_convertion_to_integer(label):
    """
    convert label into an array of +1 and -1. 
    If number of species (classes) is not exacly 2, raise error.
    
    Parameters
    ----------
    label : list or ndarray (1D)
        Only two kinds of labels are allowed. 
        For example, {'flower', 'fruit'} or {4, 6}. 
        Not allowed if labels = {'hi', 'hello', 'yes'} or {False}

    Returns
    -------
    y : int array (1D) where elements drawsn from {-1, 1}

    """
    if isinstance(label, (list, tuple)):
        label = np.array(label)
    if not isinstance(label, np.ndarray) :
        raise TypeError("label must be a list or numpy.ndarray")
    elif not (label.ndim == 1):
        raise ValueError("label must be 1 dimensional")
    elif not len(list(set(label))) == 2:
        raise ValueError("number of label(names) must be exactly two")
    else:
        pass
    label_names = list(set(label))
    y = np.array([1 if item==label_names[0] else -1 for item in label])
    return y

def check_argument_consistency(X, label):
    y = label_convertion_to_integer(label)
    if not isinstance(X, np.ndarray) :
        raise TypeError("X must be numpy.ndarray")
    elif not X.ndim == 2:
        raise ValueError("X must be a 2D-array")
    elif not len(X) == len(y):
        raise ValueError("length of X and y do not match")
    elif not np.sum(np.isreal(X))==X.size:
        raise ValueError("X must be real")
    else:
        pass
    return y

def get_combinations_of_three(vertices_positive, vertices_negative):
    if not all(isinstance(item, list) for item in [vertices_positive, vertices_negative]):
        raise TypeError("vertices_positive and vertices_negative must be a list")
    elif not (len(vertices_positive)>0 and len(vertices_negative)>0):
        raise ValueError("vertices cannot be empty")
    elif not (len(vertices_positive+vertices_negative)>2):
        raise ValueError("total number of vertices must be at least three")
    else:
        pass
    all_comb = []
    # two positive and one negative
    comb_positive = combinations(vertices_positive, 2)
    for ij in list(comb_positive): 
        i, j = ij
        for k in vertices_negative:
            all_comb.append([i, j, k])
    # two negative and one positive 
    comb_negative = combinations(vertices_negative, 2)
    for ij in list(comb_negative): 
        i, j = ij
        for k in vertices_positive:
            all_comb.append([i, j, k])
    return all_comb

def get_combinations_of_two(vertices_positive, vertices_negative):
    if not all(isinstance(item, list) for item in [vertices_positive, vertices_negative]):
        raise TypeError("vertices_positive and vertices_negative must be a list")
    elif not (len(vertices_positive)>0 and len(vertices_negative)>0):
        raise ValueError("vertices cannot be empty")
    else:
        pass
    all_comb = product(vertices_positive, vertices_negative)
    return all_comb

if __name__ == '__main__':
    pass