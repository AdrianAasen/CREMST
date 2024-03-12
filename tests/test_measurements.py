import unittest
import numpy as np
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import measurement_functions as mf
from EMQST_lib.povm import POVM
from EMQST_lib import support_functions as sf

class TestMesh(unittest.TestCase):
    def test_simulated_measurement(self):
        rho = np.array([[1,0],[0,0]])
        n_shots = 100

        comp_povm = POVM.computational_basis_POVM(1)[0]
        outcomes = mf.simulated_measurement(n_shots,comp_povm,rho)
        true = np.array([0]*n_shots)
        self.assertTrue(np.all(outcomes == true),"Computational basis measurements are not correct.")
        
        
        np.random.seed(0) # Set random seed
        iniPOVM1 = POVM.computational_basis_POVM(1)[0]
        POVMset2 = 1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)
        #iniPOVM1 = POVM(POVMset1,np.array([[[0,0],[np.pi,0]]]))
        iniPOVM2 = POVM(POVMset2,np.array([[[np.pi/2,np.pi/2],[np.pi/2,3*np.pi/2]]]))   

        # Prepare x state
        rho = np.array([[1/2,1/2],[1/2,1/2]],dtype=complex)
        outcome1 = mf.simulated_measurement(n_shots,iniPOVM1,rho)

        self.assertTrue(np.array_equal(outcome1,np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                                                0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
                                                0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                                                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                                                1, 0, 1, 0],dtype=float)))
        outcome2=mf.simulated_measurement(n_shots,iniPOVM2,rho)

        self.assertTrue(np.array_equal(outcome2,np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
                                                1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0,
                                                0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                                                0, 0, 0, 0],dtype=float)))
        
        
        
    def test_simulated_frequencies(self): 
        comp_povm = POVM.computational_basis_POVM(1)[0]
        n_shots = 100
        return_frequencies = True
        rho = np.array([[1,0],[0,0]])
        
        outcome_frequencies = mf.simulated_measurement(n_shots,comp_povm,rho,return_frequencies)
        self.assertEqual(outcome_frequencies,np.array([n_shots]))
        
        rho = np.array([[0,0],[0,1]])
        outcome_frequencies = mf.simulated_measurement(n_shots,comp_povm,rho,return_frequencies)
        self.assertEqual(outcome_frequencies,np.array([n_shots]))
        
        np.random.seed(0) # Set random seed. 
        rho = 1/2 * np.array([[1,1],[1,1]])
        outcome_frequencies = mf.simulated_measurement(n_shots,comp_povm,rho,return_frequencies)
        self.assertTrue(np.all(outcome_frequencies == np.array([51,49])), 'x-state not sampled correctly.')
        
    def test_random_Pauli_6_measurements(self):
        np.random.seed(0)
        n_qubits=1

        n_states=10
        rho=np.array([sf.generate_random_pure_state(1) for _ in range(n_states)])#np.array([[1/2,1/2],[1/2,1/2]],dtype=complex)
        n_shots=10**6
        povm_mesh=POVM.generate_Pauli_POVM(n_qubits)
        outcome=np.zeros((3,10))
        
        for i in range (3):
            for j in range(n_states):
                out_temp=mf.measurement(n_shots,povm_mesh[i],rho[j],False,{})            
                outcome[i,j]=np.sum(out_temp)/n_shots

        
        self.assertTrue(np.array_equal(outcome,np.array([[0.086265, 0.100177, 0.681203, 0.454587, 0.564957, 0.084083, 0.102104, 0.167757,
        0.735205, 0.185308],
        [0.517672, 0.69213,  0.281892, 0.463051 ,0.199737, 0.423756, 0.611773 ,0.81554,
        0.833552, 0.502384],
        [0.220338, 0.270147, 0.088223, 0.003455, 0.10485,  0.767328, 0.218644, 0.699056,
        0.788635, 0.111963]])))