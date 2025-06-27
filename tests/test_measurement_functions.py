import unittest
import numpy as np
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import measurement_functions as mf
from EMQST_lib.povm import POVM
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot


class TestOutcomesToFrequencies(unittest.TestCase):
    def test_outcomes_to_frequencies(self):
        # Test case 1: Single outcome
        outcomes = np.array([1])
        min_length = 3
        expected_result = np.array([0, 1, 0])
        self.assertTrue(np.all(mf.outcomes_to_frequencies(outcomes, min_length) == expected_result))

        # Test case 2: Multiple outcomes with repetitions
        outcomes = np.array([1, 2, 1, 3, 2, 2])
        min_length = 4
        expected_result = np.array([0, 2, 3, 1])
        self.assertTrue(np.all(mf.outcomes_to_frequencies(outcomes, min_length) == expected_result))

        # Test case 3: All outcomes present
        outcomes = np.array([0, 1, 2])
        min_length = 3
        expected_result = np.array([1, 1, 1])
        self.assertTrue(np.all(mf.outcomes_to_frequencies(outcomes, min_length) == expected_result))

        # Test case 4: No outcomes
        outcomes = np.array([])
        min_length = 2
        expected_result = np.array([0, 0])
        self.assertTrue(np.all(mf.outcomes_to_frequencies(outcomes, min_length) == expected_result))

        # Test case 5: Empty array
        outcomes = np.array([])
        min_length = 0
        expected_result = np.array([])
        self.assertTrue(np.all(mf.outcomes_to_frequencies(outcomes, min_length) == expected_result))
        
        
    def test_simulated_measurement(self):
        rho = np.array([[1,0],[0,0]])
        n_shots = 100

        comp_povm = POVM.generate_computational_POVM(1)[0]
        outcomes = mf.simulated_measurement(n_shots,comp_povm,rho)
        true = np.array([0]*n_shots)
        self.assertTrue(np.all(outcomes == true),"Computational basis measurements are not correct.")
        
        
        np.random.seed(0) # Set random seed
        iniPOVM1 = POVM.generate_computational_POVM(1)[0]
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
        comp_povm = POVM.generate_computational_POVM(1)[0]
        n_shots = 100
        return_frequencies = True
        rho = np.array([[1,0],[0,0]])
        
        outcome_frequencies = mf.simulated_measurement(n_shots,comp_povm,rho,return_frequencies)
        self.assertTrue(np.all(outcome_frequencies == np.array([n_shots,0])))
        
        rho = np.array([[0,0],[0,1]])
        outcome_frequencies = mf.simulated_measurement(n_shots,comp_povm,rho,return_frequencies)
        self.assertTrue(np.all(outcome_frequencies == np.array([0,n_shots])))
        
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
        0.788635, 0.111963]])),'Average value for single qubit Pauli-6 was not measured correctly.')
        
        
    def test_higher_qubit_pauli_6_measurements(self):
        np.random.seed(0)
        n_qubits=6
        n_shots = 100
        povm_mesh=POVM.generate_Pauli_POVM(n_qubits)
        rho = sf.generate_random_pure_state(n_qubits)
        results = np.array([mf.simulated_measurement(n_shots,povm,rho) for povm in povm_mesh])
        self.assertTrue(np.all(results.shape == (3**n_qubits,100)), 'Pauli-6 measurements not generated correctly for 6 qubits.')
        
        
    def test_measure_separable_state(self):
        n_shots = 4
        povm_list = np.array([POVM.generate_computational_POVM(1)[0], POVM.generate_computational_POVM(1)[0]])
        rho_list = np.array([[[1,0],[0,0]],[[0,0],[0,1]]])
        
        outcomes = mf.measure_separable_state(n_shots, povm_list, rho_list)
 
        expected_outcomes = np.array([[0, 1], [0,1],[0,1], [0,1]])
        self.assertTrue(np.all(outcomes == expected_outcomes))
        
        povm_list = np.array([POVM.generate_computational_POVM(1)[0], POVM.generate_computational_POVM(1)[0], POVM.generate_computational_POVM(1)[0]])
        rho_list = np.array([[[1,0],[0,0]], [[0,0],[0,1]], [[1/2,1/2],[1/2,1/2]]])
        np.random.seed(0)
        n_shots = 5
        outcomes = mf.measure_separable_state(n_shots, povm_list, rho_list)
        expected_outcomes = np.array([[0, 1, 1], [0, 1, 1] , [0, 1, 1], [0, 1, 1], [0,1,0]])
        self.assertTrue(np.all(outcomes == expected_outcomes))


    def test_measure_hashed_chunk_QST(self):
        # Test different chunk and state configureations
        # Does not checks, only checks that runs terminate without errors
        n_shots = 100
        chunk_size = 2
        # 6 qubit POVM 
        povm_size_array = np.array([1,1,1,1,2])
        state_size_array = np.array([2,2,1,1])	
        n_qubits = np.sum(state_size_array)
        povm_array = [POVM.generate_computational_POVM(size)[0] for size in povm_size_array]
        state_array = [sf.generate_random_pure_state(size) for size in state_size_array]

        hash_function = ot.create_2RDM_hash(n_qubits)
        possible_QST_instructions = np.array(['X', 'Y', 'Z'])
        n_hash_symbols = 2
        hashed_QST_instructions = ot.create_hashed_instructions(hash_function, possible_QST_instructions, n_hash_symbols)
        hashed_outcomes = mf.measure_hashed_chunk_QST(n_shots, chunk_size, povm_array, povm_size_array, state_array, state_size_array,hashed_QST_instructions) 
        # Chunksize 4
        chunk_size = 4
        povm_size_array = np.array([4, 3,1, 2,2])
        state_size_array = np.array([2,2, 1,1,1,1, 3,1])	
        n_qubits = np.sum(state_size_array)
        povm_array = [POVM.generate_computational_POVM(size)[0] for size in povm_size_array]
        state_array = [sf.generate_random_pure_state(size) for size in state_size_array]

        hash_function = ot.create_2RDM_hash(n_qubits)
        possible_QST_instructions = np.array(['X', 'Y', 'Z'])
        n_hash_symbols = 2
        hashed_QST_instructions = ot.create_hashed_instructions(hash_function, possible_QST_instructions, n_hash_symbols)
        hashed_outcomes = mf.measure_hashed_chunk_QST(n_shots, chunk_size, povm_array, povm_size_array, state_array, state_size_array,hashed_QST_instructions) 
        
        # Chunksize 8
        chunk_size = 8
        povm_size_array = np.array([4,3,1])
        state_size_array = np.array([2,2,1,1,1,1])	
        n_qubits = np.sum(state_size_array)
        povm_array = [POVM.generate_computational_POVM(size)[0] for size in povm_size_array]
        state_array = [sf.generate_random_pure_state(size) for size in state_size_array]

        hash_function = ot.create_2RDM_hash(n_qubits)
        possible_QST_instructions = np.array(['X', 'Y', 'Z'])
        n_hash_symbols = 2
        hashed_QST_instructions = ot.create_hashed_instructions(hash_function, possible_QST_instructions, n_hash_symbols)
        hashed_outcomes = mf.measure_hashed_chunk_QST(n_shots, chunk_size, povm_array, povm_size_array, state_array, state_size_array,hashed_QST_instructions) 
        
if __name__ == '__main__':
    unittest.main()