import numpy as np
from gsvm.helper import check_argument_consistency
from gsvm.hyperplane import HyperPlane
from gsvm.helper import get_combinations_of_three
from gsvm.helper import get_combinations_of_two
class GeometricClassifier():
    
    def __init__(self):
        self.flag_non_separable = False # will be set True during the run, if data is non-separable.
        self.success = False    # will be set True during the run, if optimal hyper-plane is reached.
        self.vertices_positive, self.vertices_negative = [], [] # vertices of each convex hull.
        self.current_vertices = []
        self.number_current_vertices = 0 # If no increment during iteration, set self.flag_optimal_hp = True.
        self.hp = HyperPlane()
    def store_indices(self):
        self.indices = np.arange(len(self.y))
        self.index_negative  = self.indices[self.y == -1]
        self.index_positive  = self.indices[self.y ==  1]
    def fit(self, X, label):
        y = check_argument_consistency(X, label) # if no Exception, good to go..
        self.X = X
        self.y = y
        self.store_indices()
        weight, bias = self.two_vector_classifier() # case wherein two vectors alone define the hyperplance. 
        # This is an opening test for the simplest case wherein number of vertices = 2.
        # Move on to more complicated iterations..
        self.current_vertices = []
        while not (self.success or self.flag_non_separable):
            # make hyperplanes from all the available combinations of vertices. Also get the cost function.
            cost_list, weight_bias_list, comb_list = self.try_combinations_from_vertices() 
            if not len(cost_list): # cost_list is empty
                self.message = '\nFail! Non-separable data.'
                self.flag_non_separable = True 
            else:
                ind = np.argmin(cost_list)
                weight, bias = weight_bias_list[ind]
                comb = comb_list[ind]
                self.vertices_positive, self.vertices_negative = [], []
                for i in comb:
                    if self.y[i] == 1:
                        self.vertices_positive.append(i)
                    else:
                        self.vertices_negative.append(i)
                self.append_convex_hull_vertices(weight, bias)

        if self.success == True:
            self.store_attributes(weight, bias)
        #print(self.message)
        return

    def initialize_convex_hull_vertices(self):
        # consider a pair: one from each group, and construct hyperplane. 
        for j in self.index_negative:
            for k in self.index_positive:
                weight, bias = self.get_hyperplane_supported_by_indices(j, k) # weight is None if pairs overlapping.
                if weight is not None:
                    self.append_convex_hull_vertices(weight, bias)
                    return
        self.flag_non_separable = True # arrive at this line if every pair is just overlapping. Rarest case though!
        self.message = '\nFail! Non-separable data.'
        return
    
    def grow_convex_hull_vertices(self):
        # at present self.number_current_vertices = 2. Find a hyperplane suppoerted by this pair
        j, k = self.vertices_positive[0], self.vertices_negative[0]
        weight, bias = self.get_hyperplane_supported_by_indices(j, k)
        if weight is not None:
            self.append_convex_hull_vertices(weight, bias)
        else:
            self.flag_non_separable = True
            self.message = '\nFail! Non-separable data.'
        return weight, bias

    def get_hyperplane_supported_by_indices(self, j, k):
        selected = [j, k] 
        X2 = self.X[selected]
        y2 = self.y[selected]
        weight, bias = self.hp.two_vectors(X2, y2)
        return weight, bias

    def append_convex_hull_vertices(self, weight, bias):
        # offset function (for positive labels) would be strictly > 1 for optimal hyper plane.
        offset_positive = self.hp.decision_function(weight, bias, self.X[self.index_positive]) 
        index_farthest = np.argmin(offset_positive) # there are points lying on the wrong side. pick the farthest
        self.vertices_positive.append(self.index_positive[index_farthest])
        # see, it is self.index_positive[index_farthest]. Not just index_farthest. i.e, index of index.
        #
        # offset function (for negative labels) would be strictly < -1 for optimal hyper plane.
        offset_negative = self.hp.decision_function(weight, bias, self.X[self.index_negative])
        index_farthest = np.argmax(offset_negative) # pick the farthest -- i.e., maximum. see the difference!
        self.vertices_negative.append(self.index_negative[index_farthest])
        # avoid repetation if any
        self.vertices_positive = list(set(self.vertices_positive))
        self.vertices_negative = list(set(self.vertices_negative))
        #store current vertices together in a list , if no change in current vertices, reached to optimal hp
        if sorted(self.current_vertices) == sorted(self.vertices_positive + self.vertices_negative):
            self.success = True
        self.current_vertices = self.vertices_positive + self.vertices_negative
        #print(self.current_vertices)
    
    def try_combinations_from_vertices(self):
        # Try all possible combinations (of 3) from vertices. Then get the hyperplane supported by them.
        cost_list, weight_bias_list, comb_list = [], [], []
        comb_three = get_combinations_of_three(self.vertices_positive, self.vertices_negative)
        for selected in comb_three:
            X3 = self.X[selected]
            y3 = self.y[selected]
            weight, bias = self.hp.three_vectors(X3, y3)
            if weight is None: # weights can be None if three points are collinear or two of them are identical
                continue
            # Now consider the subset of current vertices alone. i.e., self.X[self.current_vertices]
            # If the hyperplane has no misclassifications, store the cost function now. Pick the minimum later.
            offset_vertices = self.hp.decision_function(weight, bias, self.X[self.current_vertices])
            y_vertices = self.y[self.current_vertices] 
            # y*(wx+b) >= 1 only for perfect classification
            if (y_vertices*offset_vertices > 1.0-self.hp.epsilon).all(): 
                cost_list.append(0.5*np.sqrt(weight@weight))
                weight_bias_list.append((weight, bias))
                comb_list.append(selected)
        # Try possible combinations (of 2) from vertices. Then get the hyperplane supported by them
        comb_two = get_combinations_of_two(self.vertices_positive, self.vertices_negative)
        for j, k in comb_two:
            weight, bias = self.get_hyperplane_supported_by_indices(j, k) 
            if weight is not None: # weight is None if pairs are overlapping.
                continue
            offset_vertices = self.hp.decision_function(weight, bias, self.X[self.current_vertices])
            y_vertices = self.y[self.current_vertices] 
            # y*(wx+b) >= 1 only for perfect classification
            if (y_vertices*offset_vertices > 1.0-self.hp.epsilon).all(): 
                cost_list.append(0.5*np.sqrt(weight@weight))
                weight_bias_list.append((weight, bias))
                comb_list.append([j, k])
        return cost_list, weight_bias_list, comb_list
    
    def store_attributes(self, weight, bias):
        y_times_offset = self.y * self.hp.decision_function(weight, bias, self.X)
        # Sometimes there are many support vectors along the boundary line. get them all
        self.support_ = np.where(y_times_offset <(1.0+self.hp.epsilon))[0] # if  1-eps < y_times_offset < 1+eps
        self.support_vectors_= self.X[self.support_]
        self.coef_ = weight
        self.intercept_ = bias
        self.width_ = np.round(2/np.sqrt(weight@weight), 12)
        self.message = 'Success! Optimal hyperplane found.'

    def two_vector_classifier(self):
        self.initialize_convex_hull_vertices()  # after this line self.number_current_vertices will be 2.
        if self.flag_non_separable == False:
            weight, bias = self.grow_convex_hull_vertices()#after this line self.number_current_vertices will be >= 2.
            if self.flag_non_separable == False:
                if len(self.current_vertices) == 2:
                    self.success = True
                    return weight, bias
        return None, None
    
if __name__ == "__main__":
    pass