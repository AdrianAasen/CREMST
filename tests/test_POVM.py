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
        
    def test_comp_to_pauli_conversion(self):
        comp = POVM.computational_basis_POVM(1)[0]
        comp_to_pauli = POVM.generate_Pauli_from_comp(comp)
        pauli = POVM.generate_Pauli_POVM(1)

        self.assertTrue(np.allclose(pauli[0].get_POVM(), comp_to_pauli[0].get_POVM()),"X rotation did not work.")
        self.assertTrue(np.allclose(pauli[1].get_POVM(), comp_to_pauli[1].get_POVM()),"Y rotation did not work.")
        self.assertTrue(np.allclose(pauli[2].get_POVM(), comp_to_pauli[2].get_POVM()),"Identity transfer did not work.")
        
        # Test that it works with larger matices
        comp = POVM.computational_basis_POVM(2)[0]
        comp_to_pauli = POVM.generate_Pauli_from_comp(comp)
        pauli = POVM.generate_Pauli_POVM(2)
        for i in range(3**2):
            self.assertTrue(np.allclose(pauli[i].get_POVM(), comp_to_pauli[i].get_POVM()),"X rotation did not work for 2 qubits.")
        # 3 qubits
        comp = POVM.computational_basis_POVM(3)[0]
        comp_to_pauli = POVM.generate_Pauli_from_comp(comp)
        pauli = POVM.generate_Pauli_POVM(3)
        for i in range(3**3):
            self.assertTrue(np.allclose(pauli[i].get_POVM(), comp_to_pauli[i].get_POVM()),"X rotation did not work for 2 qubits.")
        
    def test_hashed_POVM(self):
        test_hash = np.array([1,0])
        pauli = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_POVM(2)])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash)])
        self.assertTrue(np.all(pauli == povm), "Standard 2 qubit assignment failed.")
        test_hash = np.array([0,1])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash)])
        # Checks if the inverted order mathces (Write down xx xy etc..)
        self.assertTrue(np.all(pauli[1] == povm[3]))
        self.assertTrue(np.all(pauli[2] == povm[6]))
        self.assertTrue(np.all(pauli[3] == povm[1]))
        self.assertTrue(np.all(pauli[4] == povm[4]))
        self.assertTrue(np.all(pauli[5] == povm[7]))
        
        # Test long hash
        long_pauli = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_POVM(4)])
        test_hash = np.array([1,0,1,0])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash)])
        self.assertTrue(np.all(long_pauli[0] == povm[0]))
        self.assertTrue(np.all(long_pauli[10] == povm[1]))
        self.assertTrue(np.all(long_pauli[20] == povm[2]))
        self.assertTrue(np.all(long_pauli[30] == povm[3]))
        self.assertTrue(np.all(long_pauli[40] == povm[4]))

    # Test hash with more than two 2 lower qubits dims
        test_hash = np.array([1,0,1,2])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash)])
        self.assertTrue(np.all(long_pauli[0] == povm[0]))
        self.assertTrue(np.all(long_pauli[18] == povm[2]))
        self.assertTrue(np.all(long_pauli[30] == povm[3]))
        self.assertTrue(np.all(long_pauli[39] == povm[4]))
        self.assertTrue(np.all(long_pauli[78] == povm[8]))
        self.assertTrue(np.all(long_pauli[1] == povm[9]))
        
        
        
if __name__ == '__main__':
    unittest.main()
