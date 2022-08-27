import numpy as np

class HyperPlane():
    def __init__(self):
        """
        epsilon: Checking for '< 1' in theory is replaced by '< 1-epsilon'.

        Returns
        -------
        None.

        """
        self.epsilon = 1e-8
    def three_vectors(self, X3, y3):
        """
        Generate the optimal-hyperplane (if exist) for 3 given vectors.

        Parameters
        ----------
        X3 : ndarray (2D) `float` type.
            Exactly 3 rows, i.e., 3 vectors.
        y3 : ndarray (1D) 'int'
            Exactly 3 elements, i.e., 3 labels from {-1,+1}. First two must be same. Thrid has the opposit label.
            
        Returns
        -------
        weight and bias (w and b) that defines the optimal-hyperplane.

        """
        # X3 -- consistency check
        if not isinstance(X3, np.ndarray) :
            raise TypeError("X3 must be numpy.ndarray")
        elif not (X3.ndim == 2 and len(X3) == 3):
            raise ValueError("X3 must be a 2D-array with exactly 3 rows")
        elif not np.sum(np.isreal(X3))==X3.size:
            raise ValueError("X3 must be real")
        else:
            pass
        # y3 -- consistency check
        if not isinstance(y3, np.ndarray):
            raise TypeError("y3 must be numpy.ndarray")
        elif not (y3.ndim == 1 and len(y3) == 3):
            raise ValueError("y3 must be a 1D-array with exactly 3 elements")
        elif not (y3[0]+y3[2]==0 and y3[1]+y3[2]==0 and y3[0]*y3[1]==1):
            raise ValueError("First two labels in y3 must be the same; and the thrid one has to be the opposit.")
        else:
            pass
        # extract elements 
        xi, xj, xk = X3[0], X3[1], X3[2]
        yi, yj, yk = y3[0], y3[1], y3[2] # yi is not necessary. May be used to confirm the result as yi =xi@w0+b0
        # compute perpendicular (w) to the hyperplane
        A = xi - xj
        B = xk - xj
        w = (A @ A)*B - (A @ B)*A
        # check the magnitude of w ?
        if w @ w < self.epsilon: # w might be zero (~ epsilon) if the points xi, xj, xk are collinear
            w0, b0 = None, None
        else:  # scale w to get the weight vector w0. Then compute bias b0.
            w0 = w * (yk-yj)/(w @ B)
            b0 = yk - xk @ w0 
        return w0, b0
    def two_vectors(self, X2, y2):
        """
        Generate the optimal-hyperplane (if exist) for 2 given vectors.

        Parameters
        ----------
        X2 : ndarray (2D) `float` type.
            Exactly 2 rows, i.e., 2 vectors.
        y2 : ndarray (1D) 'int'
            Exactly 2 elements, i.e., 2 labels from {-1,+1}. First two must be same. Thrid has the opposit label.
            
        Returns
        -------
        weight and bias (w and b) that defines the optimal-hyperplane.

        """
        # X2 -- consistency check
        if not isinstance(X2, np.ndarray) :
            raise TypeError("X2 must be numpy.ndarray")
        elif not (X2.ndim == 2 and len(X2) == 2):
            raise ValueError("X2 must be a 2D-array with exactly 2 rows")
        elif not np.sum(np.isreal(X2))==X2.size:
            raise ValueError("X2 must be real")
        else:
            pass
        # y2 -- consistency check
        if not isinstance(y2, np.ndarray):
            raise TypeError("y2 must be numpy.ndarray")
        elif not (y2.ndim == 1 and len(y2) == 2):
            raise ValueError("y2 must be a 1D-array with exactly 2 elements")
        elif not (y2[0]+y2[1]==0 and y2[0]*y2[1]==-1):
            raise ValueError("labels y2 drawn from {-1,+1} must be different.")
        else:
            pass
        #extract elements 
        xj, xk = X2[0], X2[1]
        yj, yk = y2[0], y2[1]
        # compute perpendicular (w) to the hyperplane
        w = xk - xj
        # check the magnitude of w ?
        if w @ w < self.epsilon: # w might be zero (~ epsilon) if xj = xk.
            w0, b0 = None, None
        else:  # scale w to get the weight vector w0. Then compute bias b0.
            w0 = w * (yk-yj)/(w @ w) 
            b0 = yk - w0 @ xk
        return w0, b0
    def predict(self, weight, bias, X):
        """
    
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        y_ = self.decision_function(weight, bias, X)
        y = [1 if v > 0 else -1 for v in y_]
        return np.array(y)
    def decision_function(self, weight, bias, X):
        return (X @ weight + bias)

if __name__ == '__main__':
    pass