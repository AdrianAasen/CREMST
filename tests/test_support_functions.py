import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf




class TestSupport(unittest.TestCase):

   
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

