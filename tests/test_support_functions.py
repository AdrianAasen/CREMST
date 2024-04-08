import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf




class TestHash(unittest.TestCase):
    def test_frequency_donconvertion(self):
        subsystem_index = np.array([3,2])
        outcome_frequencies= np.arange(5*16).reshape(5,16)
        downconverted_freq = sf.downconvert_frequencies(subsystem_index,outcome_frequencies)
        self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,0,:,:])))
        self.assertTrue(np.all(downconverted_freq[0,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,1,:,:])))
        self.assertTrue(np.all(downconverted_freq[0,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,1,0,:,:])))
        self.assertTrue(np.all(downconverted_freq[1,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[1,1,1,:,:])))
        subsystem_index = np.array([2,0])
        outcome_frequencies= np.arange(5*16).reshape(5,16)
        #print(outcome_frequencies)
        downconverted_freq = sf.downconvert_frequencies(subsystem_index,outcome_frequencies)
        
        #print(np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,:,0,:,0]))
        self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,:,0,:,0])))
        self.assertTrue(np.all(downconverted_freq[0,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,:,1,:,0])))
        self.assertTrue(np.all(downconverted_freq[2,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,:,0,:,0])))
        self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,:,1,:,0])))
        
        subsystem_index = np.array([3,1])
        outcome_frequencies= np.arange(5*16).reshape(5,16)
        downconverted_freq = sf.downconvert_frequencies(subsystem_index,outcome_frequencies)
        self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,:,0,:])))
        self.assertTrue(np.all(downconverted_freq[3,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[3,0,:,1,:])))
        self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,1,:,0,:])))
        self.assertTrue(np.all(downconverted_freq[1,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[1,1,:,1,:])))
        
        subsystem_index = np.array([3,0])
        outcome_frequencies= np.arange(5*16).reshape(5,16)
        downconverted_freq = sf.downconvert_frequencies(subsystem_index,outcome_frequencies)
        
        self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,:,:,0])))
        self.assertTrue(np.all(downconverted_freq[3,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[3,0,:,:,1])))
        self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,1,:,:,0])))
        self.assertTrue(np.all(downconverted_freq[4,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[4,1,:,:,1])))
        
        # Test different size systems
    
        subsystem_index = np.array([3,0])
        outcome_frequencies= np.arange(5*32).reshape(5,32)
        downconverted_freq = sf.downconvert_frequencies(subsystem_index,outcome_frequencies)
        
        self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[0,:,0,:,:,0])))
        self.assertTrue(np.all(downconverted_freq[3,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[3,:,0,:,:,1])))
        self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[2,:,1,:,:,0])))
        self.assertTrue(np.all(downconverted_freq[4,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[4,:,1,:,:,1])))
                
        
    def test_get_traced_out_indicies(self):
        index_to_keep = np.array([0,3])
        traced_out = sf.get_traced_out_indicies(index_to_keep,4)
        self.assertTrue(np.all(traced_out == np.array([1,2])))
        
        traced_out = sf.get_traced_out_indicies(index_to_keep,5)
        self.assertTrue(np.all(traced_out == np.array([1,2,4])))
        
        index_to_keep = np.array([0])
        traced_out = sf.get_traced_out_indicies(index_to_keep,5)
        self.assertTrue(np.all(traced_out == np.array([1,2,3,4])))
        
        index_to_keep = np.array([1,2,3])
        traced_out = sf.get_traced_out_indicies(index_to_keep,3)
        self.assertTrue(np.all(traced_out == np.array([])))
        
        
        index_to_keep = np.array([2])
        traced_out = sf.get_traced_out_indicies(index_to_keep,3)
        self.assertTrue(np.all(traced_out == np.array([0,1])))
        
        traced_out = sf.get_traced_out_indicies(index_to_keep,11)
        self.assertTrue(np.all(traced_out == np.array([0,1,3,4,5,6,7,8,9,10])))
        
   
    def test_infidelity(self):
        # Test for the one_qubit_infidelity function
        rho = np.array([[1, 0], [0, 0]])  # Pure state |0⟩
        sigma = np.array([[0, 0], [0, 1]])  # Pure state |1⟩
        self.assertTrue(sf.one_qubit_infidelity(rho, sigma) == 1)
        
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])  # X state (|0⟩ + |1⟩)/√2
        sigma = np.array([[1, 0], [0, 0]])  # Pure state |0⟩
        self.assertTrue(sf.one_qubit_infidelity(rho, sigma) == 0.5)
        
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])  # X state
        sigma = np.array([[0.5, 0.5], [0.5, 0.5]])  # X state
        self.assertTrue(sf.one_qubit_infidelity(rho, sigma) == 0)
        
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])  # X state
        sigma = np.array([[0, 0], [0, 1]])  # Pure state |1⟩
        self.assertTrue(sf.one_qubit_infidelity(rho, sigma) == 0.5)
        
        rho = np.array([[1, 0], [0, 0]])  # Pure state |0⟩
        sigma = np.array([[1, 0], [0, 0]])  # Pure state |0⟩
        self.assertTrue(sf.one_qubit_infidelity(rho, sigma) == 0)


    def test_get_opposing_angles(self):
        angles = np.array([[0, 0], [np.pi/2, np.pi], [np.pi, 0]])
        # expected_result = np.array([[np.pi,np.pi ], [np.pi/2, 0], [0, np.pi]])
        density = np.array([sf.get_density_matrix_from_angles(np.array([angle])) for angle in angles])
        result = sf.get_opposing_angles(angles)
        result_density = np.array([sf.get_density_matrix_from_angles(np.array([angle])) for angle in result])
        # Check that the two states defined by the angles are in fact orhtogonal
        self.assertTrue(np.allclose(np.array([0,0,0]), np.einsum('ijk,ikj->i',density,result_density)))


        angles = np.array([[np.pi/4, np.pi/4], [np.pi/3, np.pi/6]])
        density = np.array([sf.get_density_matrix_from_angles(np.array([angle])) for angle in angles])
        # expected_result = np.array([[3*np.pi/4, 5*np.pi/4], [2*np.pi/3, 7*np.pi/6]])
        result = sf.get_opposing_angles(angles)
        result_density = np.array([sf.get_density_matrix_from_angles(np.array([angle])) for angle in result])
        self.assertTrue(np.allclose(np.array([0,0]), np.einsum('ijk,ikj->i',density,result_density)))

        angles = np.array([[np.pi/6, np.pi/3], [np.pi/2, np.pi/2], [np.pi/4, np.pi/4]])
        density = np.array([sf.get_density_matrix_from_angles(np.array([angle])) for angle in angles])
        # expected_result = np.array([[5*np.pi/6, 4*np.pi/3], [0, np.pi], [3*np.pi/4, 5*np.pi/4]])
        result = sf.get_opposing_angles(angles)
        result_density = np.array([sf.get_density_matrix_from_angles(np.array([angle])) for angle in result])
        self.assertTrue(np.allclose(np.array([0,0,0]), np.einsum('ijk,ikj->i',density,result_density)))
        
    def test_binary_to_decimal(self):
        # Test case 1
        a1 = np.array([[1, 0, 1]])
        expected1 = np.array([5])
        assert sf.binary_to_decimal(a1) == expected1

        # Test case 2
        a2 = np.array([1, 1, 0, 1])
        expected2 = 13
        assert sf.binary_to_decimal(a2) == expected2

        # Test case 3
        a3 = np.array([0, 0, 0, 0, 1])
        expected3 = 1
        assert sf.binary_to_decimal(a3) == expected3

        # Test case 4
        a4 = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
        expected4 = np.array([15, 0])
        assert np.all(sf.binary_to_decimal(a4) == expected4)

        # Test case 5
        a5 = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
        expected5 = np.array([21, 10])
        assert np.all(sf.binary_to_decimal(a5) == expected5)

        
        # Test case 5
        a6 = np.array([[[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],[[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]])
        expected6 = np.array([[21, 10],[21, 10]])
        assert np.all(sf.binary_to_decimal(a6) == expected6)
        # Add more test cases if needed
        
if __name__ == '__main__':
    unittest.main()

