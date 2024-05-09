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
        traced_down_povm = povm.partial_trace(rho)
        expected1 = POVM.generate_computational_POVM(1)[0] # Single qubit POVM
        self.assertTrue(traced_down_povm == expected1)

        # Test case 2: Invalid POVM
        povm = POVM.generate_computational_POVM(4)[0]
        traced_down_povm = povm.partial_trace(rho)
        self.assertIsNone(traced_down_povm)

        # Test case 3: Test Pauli POVM
        povm_array = POVM.generate_Pauli_POVM(2)
        traced_down_povm =  np.array([povm.partial_trace(rho) for povm in povm_array])
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
        
        self.assertTrue(np.all(np.isclose(c, np.array([0,0]))))

        noisy_povm = noisy_povm = POVM.generate_noisy_POVM(povm, 4)
        c = noisy_povm.get_classical_correlation_coefficient()
        self.assertTrue(np.all(np.isclose(c, np.array([0,0]))))

        #noisy_povm = noisy_povm = POVM.generate_noisy_POVM(povm, 3)
        #c = noisy_povm.get_classical_correlation_coefficient()
        #print(c)
        
        #self.assertTrue(np.all(np.isclose(c, np.array([0,0]))))
        # # Test case 2: Non-two qubit POVM
        # POVM_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # p = povm.POVM(POVM_list)
        # c = p.get_classical_correlation_coefficient()
        # self.assertIsNone(c)

        # # Test case 3: Custom two qubit POVM
        # POVM_list = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
        # p = povm.POVM(POVM_list)
        # c = p.get_classical_correlation_coefficient()
        # self.assertEqual(c, 1)
        
        
        
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
        for i in range(7):
            noisy_povm = POVM.generate_noisy_POVM(povm, i+1)
            c = noisy_povm.get_quantum_correlation_coefficient()
            classical = noisy_povm.get_classical_correlation_coefficient()
            print(i,c,classical)
            self.assertTrue(np.all(c>=classical) or np.all(np.isclose(classical-c, np.array([0,0]))))
            
        for _ in range(10): # Testing random noise
            noisy_povm = POVM.generate_random_POVM(4,4)
            c = noisy_povm.get_quantum_correlation_coefficient()
            classical = noisy_povm.get_classical_correlation_coefficient()
            #print(c,classical)
            self.assertTrue(np.all(c>=classical) or np.all(np.isclose(classical-c, np.array([0,0]))))
    # def test_noise_POVM(self):
    #     np.random.seed(0)
    #     POVMset=1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)#np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex)#
    #     iniPOVM=POVM(POVMset,np.array([[[0,0],[np.pi,0]]]))
    #     bool_exp_meaurement=False
    #     expDict={}
    #     calibration_angles=np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]],
    #                         [[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]],
    #                         [[0,0]],[[np.pi,0]]])
    #     calibration_states=np.array([sf.get_density_matrix_from_angles(angle) for angle in calibration_angles])



    #     nShots=10**4
    #     start = time.time()

    #     for i in range(4):
    #         noise_mode=i+1
    #         noisy_POVM=POVM.generate_noisy_POVM(iniPOVM,noise_mode)
    #         #print(noisy_POVM.get_POVM())
    #         corrPOVM=dt.device_tomography(1, nShots, noisy_POVM,calibration_states,bool_exp_meaurement,expDict,iniPOVM)
    #         #print(corrPOVM.get_POVM())
    #         print(f'Distance between reconstructed and noisy POVM: {sf.POVM_distance(corrPOVM.get_POVM(),noisy_POVM.get_POVM())}')
    #         #print(1/np.sqrt(nShots))
    #         #print(np.allclose(corrPOVM.get_POVM(),noisy_POVM.get_POVM(),atol=1/np.sqrt(nShots)))
    #         assert np.allclose(corrPOVM.get_POVM(),noisy_POVM.get_POVM(),atol=1/np.sqrt(nShots))
        
        
        
        
if __name__ == '__main__':
    unittest.main()
