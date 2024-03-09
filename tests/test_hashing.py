import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf





class TestHash(unittest.TestCase):
    
    
    def test_downconvert(self):
        # Test that 2 qubit downconvertion works
        test = np.arange(16)
        qubit_index_1 = np.array([1,2])

        self.assertTrue(np.all(sf.downconvert_data(qubit_index_1,test)==np.array([0,0,1,1,2,2,3,3,0,0,1,1,2,2,3,3])),"Downcovert does not mach ideal case.")
        qubit_index_2 = np.array([0,3])
        self.assertTrue(np.all(sf.downconvert_data(qubit_index_2,test)==np.array([0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3])),"Downcovert does not mach ideal case.")
        
    def test_downconvert_false(self):
        # Test that 2 qubit downconvertion works
        test = np.arange(16)
        qubit_index_1 = np.array([0,2])
        self.assertFalse(np.all(sf.downconvert_data(qubit_index_1,test)==np.array([0,0,1,1,2,2,3,3,0,0,1,1,2,2,3,3])), "Should be wrong")

    def test_downcovert_inverse_order(self):
        # Checks if inverse order matter
        test = np.arange(16)
        self.assertTrue(np.all(sf.downconvert_data(np.array([1,0]),test) == sf.downconvert_data(np.array([0,1]),test)),"Index order is not followed.")



    def test_single_calibration_states_generation(self):
        #This test ensures that the ordering of the hashing function is correct. 
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        one_qubit_calibration_states = np.array([sf.get_density_matrix_from_angles(angle) for angle in one_qubit_calibration_angles])
        hash = np.array([1,0])
        test = sf.generate_calibration_states_from_hash(hash,one_qubit_calibration_states)
        self.assertEqual(test[0,0,0],1)
        self.assertEqual(test[1,1,1],1)
        self.assertEqual(test[2,2,2],1)
        self.assertEqual(test[3,3,3],1)
        
        hash = np.array([0,1])
        test = sf.generate_calibration_states_from_hash(hash,one_qubit_calibration_states)
        self.assertEqual(test[0,0,0],1)
        self.assertEqual(test[1,2,2],1)
        self.assertEqual(test[2,1,1],1)
        self.assertEqual(test[3,3,3],1)
        
    def test_duplicated_calibration_state(self):
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        one_qubit_calibration_states = np.array([sf.get_density_matrix_from_angles(angle) for angle in one_qubit_calibration_angles])
        hash = np.array([1,1])
        test = sf.generate_calibration_states_from_hash(hash,one_qubit_calibration_states)
        self.assertEqual(test[0,0,0],1)
        self.assertEqual(test[1,0,0],1)
        self.assertEqual(test[2,3,3],1)
        self.assertEqual(test[3,3,3],1)
        
    def test_higher_order_calibration_states(self):
        # Test that multiple hashes work 
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        one_qubit_calibration_states = np.array([sf.get_density_matrix_from_angles(angle) for angle in one_qubit_calibration_angles])
        hash = np.array([[1,0,1],[1,0,1]])
        test = np.array([sf.generate_calibration_states_from_hash(function,one_qubit_calibration_states) for function in hash])
        self.assertEqual(test[0,0,0,0],1)
        self.assertEqual(test[0,1,2,2],1)
        self.assertEqual(test[0,2,5,5],1)
        self.assertEqual(test[0,3,7,7],1)
        self.assertEqual(test[1,0,0,0],1)
        self.assertEqual(test[1,1,2,2],1)
        
        # Test with more than two outputes
        hash = np.array([0,2])
        test = np.array(sf.generate_calibration_states_from_hash(hash,one_qubit_calibration_states))
        self.assertEqual(test[1,2,2],1)
        self.assertEqual(test[2,0,0],1)
        self.assertEqual(test[3,2,2],1)
        self.assertEqual(test[4,1,1],1)
        
        
        


if __name__ == '__main__':
    unittest.main()
