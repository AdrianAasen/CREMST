import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib.povm import POVM
import EMQST_lib.povm as pv

class testPOVM(unittest.TestCase):
    
    def test_getting(self):
        simple_POVM = np.array([[[1,0],[0,0]],[[0,0],[0,1]]])
        angles = np.array([[0,0],[np.pi/2,0]])
        test = POVM(simple_POVM,angles)
        self.assertTrue(np.all(simple_POVM == test.get_POVM()),"Setting and getting does not work.")
        
        self.assertTrue(np.all(test.get_angles()==angles))
        povm = POVM.generate_computational_POVM(1)
        self.assertTrue(np.all(povm[0].get_POVM()==simple_POVM))
    
    def test_angles(self):
        povm = POVM.generate_computational_POVM(1)[0]
        angles = np.array([[[0,0]],[[np.pi,0]]])
        angles_povm = povm.get_angles()
        self.assertTrue(np.all(angles == angles_povm))
    
    def test_pauli(self):
        pauli = POVM.generate_Pauli_POVM(1)
        comp_basis = POVM.generate_computational_POVM(1)[0]
        y = np.array([[[0.5,-0.5j],[0.5j,0.5]],[[0.5,0.5j],[-0.5j,0.5]]], dtype = complex)
        x = np.array([[[0.5,0.5],[0.5,0.5]],[[0.5,-0.5],[-0.5,0.5]]], dtype = complex)
        # Check z measurement is equal to comp. basis
        self.assertTrue(np.all(pauli[2].get_POVM() == comp_basis.get_POVM()))
        self.assertTrue(np.all(pauli[2].get_angles() == comp_basis.get_angles()))
        self.assertTrue(np.all(pauli[1].get_POVM() == y))
        self.assertTrue(np.all(pauli[0].get_POVM() == x))
        
    def test_comp_to_pauli_conversion(self):
        comp = POVM.generate_computational_POVM(1)[0]
        comp_to_pauli = POVM.generate_Pauli_from_comp(comp)
        pauli = POVM.generate_Pauli_POVM(1)

        self.assertTrue(np.allclose(pauli[0].get_POVM(), comp_to_pauli[0].get_POVM()),"X rotation did not work.")
        self.assertTrue(np.allclose(pauli[1].get_POVM(), comp_to_pauli[1].get_POVM()),"Y rotation did not work.")
        self.assertTrue(np.allclose(pauli[2].get_POVM(), comp_to_pauli[2].get_POVM()),"Identity transfer did not work.")
        
        # Test that it works with larger matices
        comp = POVM.generate_computational_POVM(2)[0]
        comp_to_pauli = POVM.generate_Pauli_from_comp(comp)
        pauli = POVM.generate_Pauli_POVM(2)
        for i in range(3**2):
            self.assertTrue(np.allclose(pauli[i].get_POVM(), comp_to_pauli[i].get_POVM()),"X rotation did not work for 2 qubits.")
        # 3 qubits
        comp = POVM.generate_computational_POVM(3)[0]
        comp_to_pauli = POVM.generate_Pauli_from_comp(comp)
        pauli = POVM.generate_Pauli_POVM(3)
        for i in range(3**3):
            self.assertTrue(np.allclose(pauli[i].get_POVM(), comp_to_pauli[i].get_POVM()),"X rotation did not work for 2 qubits.")
        
    def test_hashed_POVM(self):
        test_hash = np.array([1,0])
        n_hash_sybols = 2
        pauli = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_POVM(2)])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash, n_hash_sybols)])
        self.assertTrue(np.all(pauli == povm), "Standard 2 qubit assignment failed.")
        test_hash = np.array([0,1])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash, n_hash_sybols)])
        # Checks if the inverted order mathces (Write down xx xy etc..)
        self.assertTrue(np.all(pauli[1] == povm[3]))
        self.assertTrue(np.all(pauli[2] == povm[6]))
        self.assertTrue(np.all(pauli[3] == povm[1]))
        self.assertTrue(np.all(pauli[4] == povm[4]))
        self.assertTrue(np.all(pauli[5] == povm[7]))
        
        # Test long hash
        long_pauli = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_POVM(4)])
        test_hash = np.array([1,0,1,0])
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash, n_hash_sybols)])
        self.assertTrue(np.all(long_pauli[0] == povm[0]))
        self.assertTrue(np.all(long_pauli[10] == povm[1]))
        self.assertTrue(np.all(long_pauli[20] == povm[2]))
        self.assertTrue(np.all(long_pauli[30] == povm[3]))
        self.assertTrue(np.all(long_pauli[40] == povm[4]))

    # Test hash with more than two 2 lower qubits dims
        test_hash = np.array([1,0,1,2])
        n_hash_sybols = 3
        povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_from_hash(test_hash, n_hash_sybols)])
        self.assertTrue(np.all(long_pauli[0] == povm[0]))
        self.assertTrue(np.all(long_pauli[18] == povm[2]))
        self.assertTrue(np.all(long_pauli[30] == povm[3]))
        self.assertTrue(np.all(long_pauli[39] == povm[4]))
        self.assertTrue(np.all(long_pauli[78] == povm[8]))
        self.assertTrue(np.all(long_pauli[1] == povm[9]))
        
    def test_computational_basis_POVM(self):
        # Test normalization
        basis = POVM.generate_computational_POVM(1)[0]
        norm = np.sum(basis.get_POVM(),axis = 0)
        
        self.assertTrue(np.all(norm == np.eye(2)),"Normalization failed.")
        
        
        
        
    def test_trace_down_POVM(self):
        # Test case 1: Two qubit POVM
        povm = POVM.generate_computational_POVM(2)[0]
        rho = 1/2*np.array([[1,0],[0,1]]) # Thermals state
        traced_down_povm = povm.reduce_POVM_two_to_one(rho)
        expected1 = POVM.generate_computational_POVM(1)[0] # Single qubit POVM
        self.assertTrue(traced_down_povm == expected1)

        # Test case 2: Invalid POVM
        povm = POVM.generate_computational_POVM(4)[0]
        traced_down_povm = povm.reduce_POVM_two_to_one(rho)
        self.assertIsNone(traced_down_povm)

        # Test case 3: Test Pauli POVM
        povm_array = POVM.generate_Pauli_POVM(2)
        traced_down_povm =  np.array([povm.reduce_POVM_two_to_one(rho) for povm in povm_array])
        expected3 = POVM.generate_Pauli_POVM(1)  # Single qubit POVM
        self.assertTrue(all(expected3[0] == povm for povm in traced_down_povm[:3]))   
        self.assertTrue(all(expected3[1] == povm for povm in traced_down_povm[3:6]))     
        self.assertTrue(all(expected3[2] == povm for povm in traced_down_povm[9:]))        
        
        
    def test_POVM_equality(self):
        povm1 = POVM.generate_computational_POVM(1)[0]
        povm2 = POVM.generate_computational_POVM(1)[0]
        self.assertTrue(povm1 == povm2)
        povm2 = POVM.generate_computational_POVM(2)[0]
        self.assertFalse(povm1 == povm2)
        self.assertFalse(povm1 == 1)
        
        

    def test_get_classical_correlation_coefficient(self):
        # Test case 1: Two qubit POVM
        povm = POVM.generate_computational_POVM(2)[0]
        c = povm.get_classical_correlation_coefficient()
        c_ac = povm.get_classical_correlation_coefficient("AC" )
        self.assertTrue(np.all(np.isclose(c, np.array([0,0]))))
        self.assertTrue(np.all(np.isclose(c_ac, np.array([0,0]))))
        
        noisy_povm = POVM.generate_noisy_POVM(povm, 4)
        c = noisy_povm.get_classical_correlation_coefficient()
        self.assertTrue(np.all(np.isclose(c, np.array([0,0]))))

        
        
    def test_get_quantum_correlation_coefficient(self):
        povm = POVM.generate_computational_POVM(2)[0]
        c = povm.get_quantum_correlation_coefficient()
        
        self.assertTrue(np.all(np.isclose(c,np.array([0,0]))))
        # Check uncorrelated noise
        noisy_povm = POVM.generate_noisy_POVM(povm, 4)
        c = noisy_povm.get_quantum_correlation_coefficient()

        self.assertTrue(np.all(np.isclose(c, np.array([0,0]))))
        np.random.seed(1)
        # Check quantum noise larger than classical noise.
        # Also checks that pv call is equivalent to class call. 
        for i in range(7):
            mode = "WC"
            noisy_povm = POVM.generate_noisy_POVM(povm, i+1)
            c = noisy_povm.get_quantum_correlation_coefficient(mode)
            classical = noisy_povm.get_classical_correlation_coefficient(mode)
            povm_array = noisy_povm.get_POVM()
            classical_from_array = pv.get_classical_correlation_coefficient(povm_array, mode)
            print(classical, classical_from_array)
            self.assertTrue(np.all(classical == classical_from_array))
            print(i,c,classical)
            self.assertTrue(np.all(c>=classical) or np.all(np.isclose(classical-c, np.array([0,0]))))
            
        print("AC")
        for i in range(7):
            mode = "AC"
            noisy_povm = POVM.generate_noisy_POVM(povm, i+1)
            c_ac = noisy_povm.get_quantum_correlation_coefficient(mode)
            classical_ac = noisy_povm.get_classical_correlation_coefficient(mode)
            noisy_povm_array = noisy_povm.get_POVM()
            classical_from_array = pv.get_classical_correlation_coefficient(noisy_povm_array, mode)
            c_from_array = pv.get_quantum_correlation_coefficient(noisy_povm_array, mode)
            self.assertTrue(np.all(c_ac == c_from_array))
            self.assertTrue(np.all(classical_ac == classical_from_array))
            print(i,c_ac ,classical_ac)
            self.assertTrue(np.all(c_ac>=classical_ac) or np.all(np.isclose(classical_ac-c_ac, np.array([0,0]))))
            
        for _ in range(10): # Testing random noise in WC mode
            mode = "WC"
            noisy_povm = POVM.generate_random_POVM(4,4)
            noisy_povm_array = noisy_povm.get_POVM()
            c = noisy_povm.get_quantum_correlation_coefficient(mode)
            c_from_array = pv.get_quantum_correlation_coefficient(noisy_povm_array, mode)
            classical = noisy_povm.get_classical_correlation_coefficient(mode)
            classical_from_array = pv.get_classical_correlation_coefficient(noisy_povm_array, mode)
            self.assertTrue(np.all(classical == classical_from_array))
            self.assertTrue(np.all(c == c_from_array))
            self.assertTrue(np.all(c>=classical) or np.all(np.isclose(classical-c, np.array([0,0]))))
        
        for _ in range(10): # Testing random noise in AC mode
            mode = "AC"
            noisy_povm = POVM.generate_random_POVM(4,4)
            c = noisy_povm.get_quantum_correlation_coefficient(mode)
            classical = noisy_povm.get_classical_correlation_coefficient(mode)
            #print(c,classical)
            self.assertTrue(np.all(c>=classical) or np.all(np.isclose(classical-c, np.array([0,0]))))
            
            
    def test_get_classical_POVM(self):
        # Create a POVM with off-diagonal elements
        povm = POVM(np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]]))
        
        # Turn the POVM into a classical POVM
        classical_povm = povm.get_classical_POVM()
        
        # Check that the off-diagonal elements are removed
        expected_povm = POVM(np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]]))
        self.assertTrue(np.array_equal(classical_povm.get_POVM(), expected_povm.get_POVM()))
        # Check that pauli off-diagonal elements are removed. 
        povm_list = POVM.generate_Pauli_POVM(1)
        expected_results = np.array([[[[0.5, 0], [0, 0.5]],[[0.5, 0], [0, 0.5]]],
                                     [[[0.5, 0], [0, 0.5]],[[0.5, 0], [0, 0.5]]],
                                     [[[1,0],[0,0]],[[0,0],[0,1]]]])
        for povm, true_povm in zip(povm_list,expected_results):
            classical_povm = povm.get_classical_POVM()
            self.assertTrue(np.array_equal(np.real(classical_povm.get_POVM()),true_povm))
            
    def test_get_coherent_error(self):
        povm = POVM.generate_computational_POVM(2)[0]
        coherent_error = povm.get_coherent_error()
        self.assertEqual(coherent_error, 0)
        
        noisy_povm = POVM.generate_noisy_POVM(povm, 1)
        coherent_error = noisy_povm.get_coherent_error()
        self.assertEqual(coherent_error, 0)
        
        noisy_povm = POVM.generate_noisy_POVM(povm, 2)
        coherent_error = noisy_povm.get_coherent_error()
        self.assertEqual(coherent_error, 0)
        
        noisy_povm = POVM.generate_noisy_POVM(povm, 3)
        coherent_error = noisy_povm.get_coherent_error()
        self.assertGreater(coherent_error, 0)
        
        # Check single qubit. 
        povm = POVM.generate_computational_POVM(1)[0]
        noisy_povm = POVM.generate_noisy_POVM(povm, 4)
        coherent_error = noisy_povm.get_coherent_error()
        self.assertGreater(coherent_error, 0)

    def test_POVM_from_angles(self):
        X = np.array([np.pi/2,0])
        Y = np.array([np.pi/2,np.pi/2])
        Z = np.array([0,0])
        n_qubits = 2
        povm_angles = np.array([[X,X], [X,Y], [X,Z],[Y,X], [Y,Y], [Y,Z], [Z,X], [Z,Y], [Z,Z]])
        
        pauli_povm = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_POVM(n_qubits)])
        # xx, xy, xz...
        
        POVM_list = [POVM.POVM_from_angles(angles) for angles in povm_angles]
        for i in range(len(povm_angles)):
            self.assertTrue(np.allclose(POVM_list[i].get_POVM(), pauli_povm[i]))


    def test_rotate_POVM_to_computational_basis(self):
        n_qubits = 2
        pauli_povm = POVM.generate_Pauli_POVM(n_qubits)
        correct_povm = POVM.generate_computational_POVM(n_qubits)[0].get_POVM()
        pauli_xx = pauli_povm[0].get_POVM()
        pauli_xy = pauli_povm[1].get_POVM()
        pauli_yy = pauli_povm[4].get_POVM()
        pauli_zz = pauli_povm[8].get_POVM()
        self.assertTrue(np.allclose(pv.rotate_POVM_to_computational_basis(pauli_xx,"XX"), correct_povm))
        self.assertTrue(np.allclose(pv.rotate_POVM_to_computational_basis(pauli_xy,"XY"), correct_povm))
        self.assertTrue(np.allclose(pv.rotate_POVM_to_computational_basis(pauli_yy,"YY"), correct_povm))
        self.assertTrue(np.allclose(pv.rotate_POVM_to_computational_basis(pauli_zz,"ZZ"), correct_povm))
        
        # Test 3 qubits
        n_qubits = 3
        pauli_povm = POVM.generate_Pauli_POVM(n_qubits)
        pauli_xxx = pauli_povm[0].get_POVM()
        correct_povm = POVM.generate_computational_POVM(n_qubits)[0].get_POVM()
        self.assertTrue(np.allclose(pv.rotate_POVM_to_computational_basis(pauli_xxx,"XXX"), correct_povm))
        
    def test_tensor_POVM(self):
        # Check that order of operations does not matter. 
        povm_a = POVM.generate_random_POVM(2,2)
        angle = np.array([['0','1']])
        povm_a.set_angles(angle)
        povm_b = POVM.generate_random_POVM(2,2)
        povm_b.set_angles(angle)
        povm_c = POVM.generate_random_POVM(2,2)
        povm_c.set_angles(angle)
        povm_ab = POVM.tensor_POVM(povm_a,povm_b)[0]
        povm_abc = POVM.tensor_POVM(povm_ab,povm_c)[0]
        povm_bc = POVM.tensor_POVM(povm_b,povm_c)[0]
        povm_abc2 = POVM.tensor_POVM(povm_a,povm_bc)[0]
        self.assertTrue(np.allclose(povm_abc.get_POVM(), povm_abc2.get_POVM()))
        
        
if __name__ == '__main__':
    unittest.main()
