import numpy as np
import unittest
from gsvm.classifier import GeometricClassifier

class TestGeometricClassifier(unittest.TestCase):
    def test_initializing(self):
        clf = GeometricClassifier()
        self.assertEqual(clf.success, False)
    def test_overlap_11(self):
        X = np.array([[1,0], [1,0]]) # identical
        label = [1, -1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertEqual(clf.success, False)
    def test_overlap_33(self):
        X = np.array([[1,0], [2,0], [3,0], [3,0], [4,0], [5,0]]) # thrid and fourth points are identical
        label = [1, 1, 1, -1, -1, -1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertEqual(clf.success, False)
    def test_separable_11(self):
        X = np.array([[1,0], [2,0]])
        label = [1, -1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertTupleEqual((clf.success, clf.width_), (True, 1.0))
    def test_separable_33(self):
        X = np.array([[1,0], [2,0], [3,0], [4,0], [5,0], [6,0]]) 
        label = [1, 1, 1, -1, -1, -1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertTupleEqual((clf.success, clf.width_), (True, 1.0))
    def test_separable_43(self):
        X = np.array([[1,0], [2,0], [3,0], [4,0], [4.2,0], [5,0], [6,0]]) 
        label = [1, 1, 1, 1, -1, -1, -1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertTupleEqual((clf.success, clf.width_), (True, 0.2))
    def test_separable_3D_4(self):
        X = np.array([[0,0,0], [5,0,0], [0,5,0], [-5,-5,-5]]) 
        label = [1, 1, 1, -1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertTupleEqual((clf.success, np.round(clf.width_,10)), (True, np.round(np.sqrt(75),10)))
    def test_separable_3D_6(self):
        X = np.array([[0,0,0], [5,0,0], [0,5,0], [0,0,5], [5,0,5], [0,5,5]]) 
        label = [1, 1, 1, -1,-1,-1]
        clf = GeometricClassifier()
        clf.fit(X, label)
        self.assertTupleEqual((clf.success, clf.width_), (True, 5.0))
if __name__ == '__main__':
    unittest.main(verbosity=2)
