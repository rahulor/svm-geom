import numpy as np
import unittest
from gsvm.hyperplane import HyperPlane

class TestHyperPlane(unittest.TestCase):
    def test_initializing(self):
        hp = HyperPlane()
        self.assertEqual(hp.epsilon, 1e-8)
       
    def test_three_vectors_good_args(self):
        # good arguments
        # no Errors expected
        X3 = np.random.uniform(size=(3,5))
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        try:
            hp.three_vectors(X3, y3)
        except Exception as the_exception:
            print('-'*50, '\n', the_exception)
            self.fail()
    #-------------------------------------------------------------------------
    def test_three_vectors_result_collinear(self):
        X3 = np.random.uniform(size=(3,5))
        X3[1] = X3[0]*7
        X3[2] = X3[0]*8
        # collinear
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        collineartext = 'collinear points should return None, None'
        self.assertTupleEqual(hp.three_vectors(X3, y3), (None, None), msg=collineartext)
    def test_three_vectors_result_identical_points(self):
        X3 = np.random.uniform(size=(3,5))
        X3[1] = X3[2]
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        collineartext = 'If two vectors are shame, w0 and b0 should be None, None'
        self.assertTupleEqual(hp.three_vectors(X3, y3), (None, None), msg=collineartext)
    def test_three_vectors_label_reconstruction(self):
        X3 = np.random.uniform(size=(3,5))
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        w0, b0 = hp.three_vectors(X3, y3)
        y3pred = np.around(X3 @ w0 + b0, 8)
        self.assertListEqual(list(y3pred), list(y3), msg='w@x+b=y')
    #-------------------------------------------------------------------------
    def three_vectors_X3_bad_type(self, X3):
        # bad X3 type
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        with self.assertRaises(TypeError) as e:
            hp.three_vectors(X3, y3)
        self.assertEqual(str(e.exception), "X3 must be numpy.ndarray")
    def test_three_vectors_X3_bad_type_a(self):
        # bad X3 type
        X3 = [i for i in range(10)]
        self.three_vectors_X3_bad_type(X3)
    def test_three_vectors_X3_bad_type_b(self):
        # bad X3 type
        X3 = 5.0
        self.three_vectors_X3_bad_type(X3)
    def test_three_vectors_X3_bad_type_c(self):
        # bad X3 type
        X3 = None
        self.three_vectors_X3_bad_type(X3)
    #-------------------------------------------------------------------------    
    def three_vectors_X3_bad_value(self, X3):
        # bad X3 value
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.three_vectors(X3, y3)
        self.assertEqual(str(e.exception), "X3 must be real")
    def test_three_vectors_X3_bad_value_a(self):
        # bad X3 value
        X3 = np.zeros((3,2),dtype=np.complex_)
        X3[0, 0] = 2+5j
        self.three_vectors_X3_bad_value(X3)
    
    #-------------------------------------------------------------------------   
    def three_vectors_X3_bad_shape(self, X3):
        # bad X3 shape
        y3 = np.array([1, 1, -1])
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.three_vectors(X3, y3)
        self.assertEqual(str(e.exception), "X3 must be a 2D-array with exactly 3 rows")
    def test_three_vectors_X3_bad_shape_a(self):
        # bad X3 shape
        X3 = np.random.uniform(size=(3,5,6))
        self.three_vectors_X3_bad_shape(X3)
    def test_three_vectors_X3_bad_shape_b(self):
        # bad X3 shape
        X3 = np.random.uniform(size=(2,5))
        self.three_vectors_X3_bad_shape(X3)
    def test_three_vectors_X3_bad_shape_c(self):
        # bad X3 shape
        X3 = np.random.uniform(size=(5))
        self.three_vectors_X3_bad_shape(X3)
    def test_three_vectors_X3_bad_shape_d(self):
        # bad X3 shape
        X3 = np.ones(5)
        self.three_vectors_X3_bad_shape(X3)
    def test_three_vectors_X3_bad_shape_e(self):
        # bad X3 shape
        X3 = np.zeros(5)
        self.three_vectors_X3_bad_shape(X3)
    def test_three_vectors_X3_bad_shape_f(self):
        # bad X3 shape
        X3 = np.random.uniform(size=(5,3))
        self.three_vectors_X3_bad_shape(X3)
       
#-------------------------------------------------------------------------
    def three_vectors_y3_bad_type(self, y3):
        # bad y3 type
        X3 = np.random.uniform(size=(3,5))
        hp = HyperPlane()
        with self.assertRaises(TypeError) as e:
            hp.three_vectors(X3, y3)
        self.assertEqual(str(e.exception), "y3 must be numpy.ndarray")
    def test_three_vectors_y3_bad_type_a(self):
        # bad y3 type
        y3 = [i for i in range(3)]
        self.three_vectors_y3_bad_type(y3)
    def test_three_vectors_y3_bad_type_b(self):
        # bad y3 type
        y3 = 3.0
        self.three_vectors_y3_bad_type(y3)
    def test_three_vectors_y3_bad_type_c(self):
        # bad y3 type
        y3 = None
        self.three_vectors_y3_bad_type(y3)
        
    #-------------------------------------------------------------------------   
    def three_vectors_y3_bad_shape(self, y3):
        # bad y3 shape
        X3 = np.random.uniform(size=(3,5))
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.three_vectors(X3, y3)
        self.assertEqual(str(e.exception), "y3 must be a 1D-array with exactly 3 elements")
    def test_three_vectors_y3_bad_shape_a(self):
        # bad X3 shape
        y3 = np.ones((3,5))
        self.three_vectors_y3_bad_shape(y3)    
    def test_three_vectors_y3_bad_shape_b(self):
        # bad X3 shape
        y3 = np.ones(2)
        self.three_vectors_y3_bad_shape(y3)
    
    #-------------------------------------------------------------------------   
    def three_vectors_y3_bad_value(self, y3):
        # bad y3 shape
        X3 = np.random.uniform(size=(3,5))
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.three_vectors(X3, y3)
        self.assertEqual(str(e.exception), "First two labels in y3 must be the same; and the thrid one has to be the opposit.")
    def test_three_vectors_y3_bad_value_a(self):
        # bad X3 shape
        y3 = np.ones(3)
        self.three_vectors_y3_bad_value(y3)    
    def test_three_vectors_y3_bad_value_b(self):
        # bad X3 shape
        y3 = np.ones(3)*-1
        self.three_vectors_y3_bad_value(y3)
    def test_three_vectors_y3_bad_value_c(self):
        # bad X3 shape
        y3 = np.array([1, -1, 1])
        self.three_vectors_y3_bad_value(y3)
    def test_three_vectors_y3_bad_value_d(self):
        # bad X3 shape
        y3 = np.array([0, 0, 1])
        self.three_vectors_y3_bad_value(y3) 
    #===========================================================================
    def test_two_vectors_good_args(self):
        # good arguments
        # no Errors expected
        X2 = np.random.uniform(size=(2,5))
        y2 = np.array([1,-1])
        hp = HyperPlane()
        try:
            hp.two_vectors(X2, y2)
        except Exception as the_exception:
            print('-'*50, '\n', the_exception)
            self.fail()
    def test_two_vectors_result_identical_points(self):
        X2 = np.random.uniform(size=(2,5))
        X2[1] = X2[0]
        y2 = np.array([1,-1])
        hp = HyperPlane()
        collineartext = 'If two vectors are shame, w0 and b0 should be None, None'
        self.assertTupleEqual(hp.two_vectors(X2, y2), (None, None), msg=collineartext)
        
    def test_two_vectors_label_reconstruction(self):
        X2 = np.random.uniform(size=(2,5))
        y2 = np.array([1,-1])
        hp = HyperPlane()
        w0, b0 = hp.two_vectors(X2, y2)
        y2pred = np.around(X2 @ w0 + b0, 8)
        self.assertListEqual(list(y2pred), list(y2), msg='w@x+b=y')

    #-------------------------------------------------------------------------
    def two_vectors_X2_bad_type(self, X2):
        # X2 bad type
        y2 = np.array([1, -1])
        hp = HyperPlane()
        with self.assertRaises(TypeError) as e:
            hp.two_vectors(X2, y2)
        self.assertEqual(str(e.exception), "X2 must be numpy.ndarray")
    def test_two_vectors_X2_bad_type_a(self):
        X2 = [i for i in range(10)]
        self.two_vectors_X2_bad_type(X2)
    def test_two_vectors_X2_bad_type_b(self):
        X2 = 5.0
        self.two_vectors_X2_bad_type(X2)
    def test_two_vectors_X2_bad_type_c(self):
        X2 = None
        self.two_vectors_X2_bad_type(X2)
    def test_two_vectors_X2_bad_type_d(self):
        X2 = 'some_string'
        self.two_vectors_X2_bad_type(X2)
#-------------------------------------------------------------------------    
    def two_vectors_X2_bad_value(self, X2):
        # bad X2 value
        y2 = np.array([1 -1])
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.two_vectors(X2, y2)
        self.assertEqual(str(e.exception), "X2 must be real")
    def test_three_vectors_X2_bad_value_a(self):
        X2 = np.zeros((2,2),dtype=np.complex_)
        X2[0, 0] = 2+5j
        self.two_vectors_X2_bad_value(X2)


     #-------------------------------------------------------------------------   
    def two_vectors_X2_bad_shape(self, X2):
        # bad X2 shape
        y2 = np.array([1, -1])
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.two_vectors(X2, y2)
        self.assertEqual(str(e.exception), "X2 must be a 2D-array with exactly 2 rows")
    def test_two_vectors_X2_bad_shape_a(self):
        # bad X2 shape
        X2 = np.random.uniform(size=(3,5,6))
        self.two_vectors_X2_bad_shape(X2)
    def test_two_vectors_X2_bad_shape_b(self):
        # bad X2 shape
        X2 = np.random.uniform(size=(1,5))
        self.two_vectors_X2_bad_shape(X2)
    def test_two_vectors_X2_bad_shape_c(self):
        # bad X2 shape
        X2 = np.random.uniform(size=(5))
        self.two_vectors_X2_bad_shape(X2)
    def test_two_vectors_X2_bad_shape_d(self):
        # bad X2 shape
        X2 = np.ones(5)
        self.two_vectors_X2_bad_shape(X2)
    def test_two_vectors_X2_bad_shape_e(self):
        # bad X2 shape
        X2 = np.zeros(5)
        self.two_vectors_X2_bad_shape(X2)
    def test_two_vectors_X2_bad_shape_f(self):
        # bad X2 shape
        X2 = np.random.uniform(size=(5,3))
        self.two_vectors_X2_bad_shape(X2)    
        
    #-------------------------------------------------------------------------
    def two_vectors_y2_bad_type(self, y2):
        # bad y2 type
        X2 = np.random.uniform(size=(2,5))
        hp = HyperPlane()
        with self.assertRaises(TypeError) as e:
            hp.two_vectors(X2, y2)
        self.assertEqual(str(e.exception), "y2 must be numpy.ndarray")
    def test_two_vectors_y2_bad_type_a(self):
        # bad y2 type
        y2 = [i for i in range(3)]
        self.two_vectors_y2_bad_type(y2)
    def test_two_vectors_y2_bad_type_b(self):
        # bad y2 type
        y2 = 3.0
        self.two_vectors_y2_bad_type(y2)
    def test_two_vectors_y2_bad_type_c(self):
        # bad y2 type
        y2 = None
        self.two_vectors_y2_bad_type(y2)
        
    #-------------------------------------------------------------------------   
    def two_vectors_y2_bad_shape(self, y2):
        # bad y2 shape
        X2 = np.random.uniform(size=(2,5))
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.two_vectors(X2, y2)
        self.assertEqual(str(e.exception), "y2 must be a 1D-array with exactly 2 elements")
    def test_two_vectors_y2_bad_shape_a(self):
        # bad X2 shape
        y2 = np.ones((3,5))
        self.two_vectors_y2_bad_shape(y2)    
    def test_two_vectors_y2_bad_shape_b(self):
        # bad X2 shape
        y2 = np.ones(3)
        self.two_vectors_y2_bad_shape(y2)
    
    #-------------------------------------------------------------------------   
    def two_vectors_y2_bad_value(self, y2):
        # bad y2 shape
        X2 = np.random.uniform(size=(2,5))
        hp = HyperPlane()
        with self.assertRaises(ValueError) as e:
            hp.two_vectors(X2, y2)
        self.assertEqual(str(e.exception), "labels y2 drawn from {-1,+1} must be different.")
    def test_two_vectors_y2_bad_value_a(self):
        # bad X2 shape
        y2 = np.ones(2)*0.5
        self.two_vectors_y2_bad_value(y2)    
    def test_two_vectors_y2_bad_value_b(self):
        # bad X2 shape
        y2 = np.ones(2)*-1
        self.two_vectors_y2_bad_value(y2)
    def test_two_vectors_y2_bad_value_c(self):
        # bad X2 shape
        y2 = np.array([1, 1])
        self.two_vectors_y2_bad_value(y2)
    def test_two_vectors_y2_bad_value_d(self):
        # bad X2 shape
        y2 = np.array([0, 1])
        self.two_vectors_y2_bad_value(y2)   

if __name__ == '__main__':
    unittest.main(verbosity=2)
