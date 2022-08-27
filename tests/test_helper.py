import numpy as np
import unittest
from gsvm.helper import label_convertion_to_integer
from gsvm.helper import check_argument_consistency
from gsvm.helper import get_combinations_of_three
class TestHerlper(unittest.TestCase):
    def test_label_conversion_list(self):
        m, n = 10, 5
        label = ['b']*m + ['a']*n
        if list(set(label))[0] == 'b':
            ypred = [+1]*m + [-1]*n
        else:
            ypred = [-1]*m + [+1]*n
        y = label_convertion_to_integer(label)
        self.assertListEqual(list(ypred), list(y), msg='compare')
    def test_label_conversion_tuplebool(self):
        label = True,True,True,False
        if list(set(label))[0] == True:
            ypred = np.array([1,1,1,-1])*1
        else:
            ypred = np.array([1,1,1,-1])*-1
        y = label_convertion_to_integer(label)
        self.assertListEqual(list(ypred), list(y), msg='compare')    
    def test_label_conversion_tuplenum(self):
        label = 5,5,7,7,7,7
        if list(set(label))[0] == 5:
            ypred = np.array([1,1,-1,-1,-1,-1])
        else:
            ypred = np.array([1,1,-1,-1,-1,-1])*-1
        y = label_convertion_to_integer(label)
        self.assertListEqual(list(ypred), list(y), msg='compare')           
    def test_label_conversion_mixed(self):
        m, n = 10, 5
        label = [int(5)]*m + ['b']*n
        if list(set(label))[0] == 'b':
            ypred = [+1]*m + [-1]*n
        else:
            ypred = [-1]*m + [+1]*n
            
        y = label_convertion_to_integer(label)
        self.assertListEqual(ypred, list(y), msg='compare')        
    #-------------------------------------------------------------------------
    def test_label_conversion_int(self):
        label = 5
        with self.assertRaises(TypeError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "label must be a list or numpy.ndarray")
    def test_label_conversion_float(self):
        label = 5.5
        with self.assertRaises(TypeError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "label must be a list or numpy.ndarray")
    def test_label_conversion_bool(self):
        label = True
        with self.assertRaises(TypeError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "label must be a list or numpy.ndarray")
    def test_label_conversion_str(self):
        label = 'okay'
        with self.assertRaises(TypeError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "label must be a list or numpy.ndarray")
    def _test_label_conversion_morelist(self):
        label = [5,5,5,5], [7,7] # are you mad ?
        print(label)
        with self.assertRaises(TypeError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "label must be a list or numpy.ndarray")
    #-------------------------------------------------------------------------
    def test_label_conversion_shape(self):
        label = np.ones((2,10))
        with self.assertRaises(ValueError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "label must be 1 dimensional")
    #-------------------------------------------------------------------------
    def test_label_conversion_single_label(self):
        label = [1,1,1,1,1,1,1,1,1,1]
        with self.assertRaises(ValueError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "number of label(names) must be exactly two")
    #-------------------------------------------------------------------------
    def test_label_conversion_many_label(self):
        label = [1,1,1,5,7,1,1,1,1]
        with self.assertRaises(ValueError) as e:
            label_convertion_to_integer(label)
        self.assertEqual(str(e.exception), "number of label(names) must be exactly two")  
        
        

    def test_Xy_consistency_y(self):
        label = 5.5
        X = np.random.uniform(size=(8,5))
        with self.assertRaises(TypeError) as e:
            check_argument_consistency(X, label)
        self.assertEqual(str(e.exception), "label must be a list or numpy.ndarray")
    def test_Xy_consistency_Xtype(self):
        label = [5,1,5,1,5,5,5]
        X = [7,8,7]
        with self.assertRaises(TypeError) as e:
            check_argument_consistency(X, label)
        self.assertEqual(str(e.exception),"X must be numpy.ndarray")
    def test_Xy_consistency_Xshape1(self):
        label = [5,1,5,1,5,5,5,5]
        X = np.arange(8)
        with self.assertRaises(ValueError) as e:
            check_argument_consistency(X, label)
        self.assertEqual(str(e.exception),"X must be a 2D-array")
    def test_Xy_consistency_Xshape2(self):
        label = [5,1,5,1,5,5,5,5]
        X = np.ones((8,2,5))
        with self.assertRaises(ValueError) as e:
            check_argument_consistency(X, label)
        self.assertEqual(str(e.exception),"X must be a 2D-array")
    def test_Xy_consistency_size_mismatch(self):
        label = [5,1,5,1,5,5,5,5]
        X = np.ones((9,3))
        with self.assertRaises(ValueError) as e:
            check_argument_consistency(X, label)
        self.assertEqual(str(e.exception),"length of X and y do not match")
    def test_Xy_consistency_size_Xnotreal(self):
        label = [5,1,5,1,5,5,5,5]
        X = np.ones((8,3),dtype=np.complex_) * 5+2j
        with self.assertRaises(ValueError) as e:
            check_argument_consistency(X, label)
        self.assertEqual(str(e.exception),"X must be real")
#--------------------------------------------------------------------------------- combinations ---
    def test_combinations_array(self):
        vertices_positive, vertices_negative = np.arange(4), np.arange(2)
        with self.assertRaises(TypeError) as e:
            get_combinations_of_three(vertices_positive, vertices_negative)
        self.assertEqual(str(e.exception), "vertices_positive and vertices_negative must be a list")
    def test_combinations_array_list(self):
        vertices_positive, vertices_negative = np.arange(4), [5,8]
        with self.assertRaises(TypeError) as e:
            get_combinations_of_three(vertices_positive, vertices_negative)
        self.assertEqual(str(e.exception), "vertices_positive and vertices_negative must be a list")
    def test_combinations_empty_list(self):
        vertices_positive, vertices_negative = [], [5,8,6]
        with self.assertRaises(ValueError) as e:
            get_combinations_of_three(vertices_positive, vertices_negative)
        self.assertEqual(str(e.exception), "vertices cannot be empty")
    def test_combinations_list_11(self):
        vertices_positive, vertices_negative = [5], [7]
        with self.assertRaises(ValueError) as e:
            get_combinations_of_three(vertices_positive, vertices_negative)
        self.assertEqual(str(e.exception), "total number of vertices must be at least three")
    
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
