import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib.povm import POVM

class testPOVM(unittest.TestCase):
    
    def test_getting(self):
        simple_POVM = np.array([[[1,0],[0,0]],[[0,0],[0,1]]])
        angles = np.array([[0,0],[np.pi/2,0]])
        test = POVM(simple_POVM,angles)
        self.assertTrue(np.all(simple_POVM == test.get_POVM()),"Setting and getting does not work.")
        
        self.assertTrue(np.all(test.get_angles()==angles))
        povm = POVM.computational_basis_POVM(1)
        self.assertTrue(np.all(povm[0].get_POVM()==simple_POVM))
    
    def test_angles(self):
        povm = POVM.computational_basis_POVM(1)[0]
        angles = np.array([[[0,0]],[[np.pi,0]]])
        angles_povm = povm.get_angles()
        self.assertTrue(np.all(angles == angles_povm))
    
    def test_pauli(self):
        pauli = POVM.generate_Pauli_POVM(1)
        comp_basis = POVM.computational_basis_POVM(1)[0]
        y = np.array([[[0.5,-0.5j],[0.5j,0.5]],[[0.5,0.5j],[-0.5j,0.5]]], dtype = complex)
        x = np.array([[[0.5,0.5],[0.5,0.5]],[[0.5,-0.5],[-0.5,0.5]]], dtype = complex)
        # Check z measurement is equal to comp. basis
        self.assertTrue(np.all(pauli[2].get_POVM() == comp_basis.get_POVM()))
        self.assertTrue(np.all(pauli[2].get_angles() == comp_basis.get_angles()))
        self.assertTrue(np.all(pauli[1].get_POVM() == y))
        self.assertTrue(np.all(pauli[0].get_POVM() == x))
        
        
        
        
        