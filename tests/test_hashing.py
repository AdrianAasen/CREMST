import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot
from EMQST_lib.povm import POVM


class TestHash(unittest.TestCase):
    
    
    
    def test_trace_out(self):
        # Test case 1
        qubit_to_keep_labels = np.array([0, 2])
        qubit_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        expected_result = np.array([[[1, 3],[4,6]],[ [7, 9],[10, 12]]])
        result = ot.trace_out(qubit_to_keep_labels, qubit_array)
        self.assertTrue(np.allclose(result, expected_result))

        # Test case 2
        qubit_to_keep_labels = [1]
        qubit_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        expected_result = np.array([[[2], [5]],[ [8], [11]]])
        result = ot.trace_out(qubit_to_keep_labels, qubit_array)
        self.assertTrue(np.allclose(result, expected_result))

        # Test case 3
        qubit_to_keep_labels = [2, 0]
        qubit_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        expected_result = np.array([[[1, 3],[4,6]],[ [7, 9],[10, 12]]])
        result = ot.trace_out(qubit_to_keep_labels, qubit_array)
        self.assertTrue(np.allclose(result, expected_result))
        
        # Test case 4
        qubit_to_keep_labels = [0]
        qubit_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        expected_result = np.array([[[3],[6]],[ [9],[12]]])
        result = ot.trace_out(qubit_to_keep_labels, qubit_array)
        self.assertTrue(np.allclose(result, expected_result))
    
    
    # def test_downconvert(self):
    #     # Test that 2 qubit downconvertion works
    #     test = np.arange(16).reshape(4,4)
    #     qubit_index_1 = np.array([1,2])

    #     self.assertTrue(np.all(sf.trace_out_outcomes(qubit_index_1,test)==np.array([0,0,1,1,2,2,3,3,0,0,1,1,2,2,3,3])),"Downcovert does not mach ideal case.")
    #     qubit_index_2 = np.array([0,3])
    #     self.assertTrue(np.all(sf.trace_out_outcomes(qubit_index_2,test)==np.array([0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3])),"Downcovert does not mach ideal case.")
        
    # def test_downconvert_false(self):
    #     # Test that 2 qubit downconvertion works
    #     test = np.arange(16)
    #     qubit_index_1 = np.array([0,2])
    #     self.assertFalse(np.all(sf.trace_out_outcomes(qubit_index_1,test)==np.array([0,0,1,1,2,2,3,3,0,0,1,1,2,2,3,3])), "Should be wrong")

    # def test_downcovert_inverse_order(self):
    #     # Checks if inverse order matter
    #     test = np.arange(16)
    #     self.assertTrue(np.all(sf.trace_out_outcomes(np.array([1,0]),test) == sf.trace_out_outcomes(np.array([0,1]),test)),"Index order is not followed.")



    def test_single_calibration_states_generation(self):
        # This test ensures that the ordering of the instructions are correct. 
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        one_qubit_calibration_states = np.array([sf.get_density_matrix_from_angles(angle) for angle in one_qubit_calibration_angles])
        instruction = np.array([1,0])
        test = ot.calibration_states_from_instruction(instruction,one_qubit_calibration_states)
        self.assertEqual(test[1,0,0],1)
        self.assertEqual(test[0,1,1],1)
        
        instruction = np.array([0,1])
        test = ot.calibration_states_from_instruction(instruction,one_qubit_calibration_states)
        self.assertEqual(test[0,0,0],1)
        self.assertEqual(test[1,1,1],1)

        
    def test_duplicated_calibration_state(self):
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        one_qubit_calibration_states = np.array([sf.get_density_matrix_from_angles(angle) for angle in one_qubit_calibration_angles])
        instruction = np.array([1,1])
        test = ot.calibration_states_from_instruction(instruction,one_qubit_calibration_states)
        self.assertEqual(test[0,1,1],1)
        self.assertEqual(test[1,1,1],1)

        
        # Test duplicate calibration, but more symbols
        instruction = np.array([0,0])
        test = ot.calibration_states_from_instruction(instruction, one_qubit_calibration_states)
        self.assertEqual(test[0,0,0],1)
        self.assertEqual(test[1,0,0],1)
        
    def test_tensor_product(self):
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        one_qubit_calibration_states = np.array([sf.get_density_matrix_from_angles(angle) for angle in one_qubit_calibration_angles])
        instruction = np.array([1,0])
        test = ot.calibration_states_from_instruction(instruction, one_qubit_calibration_states,True)
        self.assertEqual(test[2,2],1)
        
        instruction = np.array([0,1,1])
        test = ot.calibration_states_from_instruction(instruction, one_qubit_calibration_states,True)
        self.assertEqual(test[3,3],1)
        
        instruction = np.array([1,1,1])
        test = ot.calibration_states_from_instruction(instruction, one_qubit_calibration_states,True)
        self.assertEqual(test[7,7],1)
        


    def test_create_unique_combinations(self):
        elements = [1, 2, 3]
        n_repeat = 2
        expected_result = np.array([ [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]])
        result = ot.create_unique_combinations(elements, n_repeat)
        self.assertTrue(np.array_equal(result, expected_result))

        elements = ['A', 'B', 'C']
        n_repeat = 3
        expected_result = np.array([['A', 'A', 'B'], ['A', 'A', 'C'], ['A', 'B', 'A'], ['A', 'B', 'B'], ['A', 'B', 'C'],
                                    ['A', 'C', 'A'], ['A', 'C', 'B'], ['A', 'C', 'C'], ['B', 'A', 'A'], ['B', 'A', 'B'],
                                    ['B', 'A', 'C'], ['B', 'B', 'A'], ['B', 'B', 'C'], ['B', 'C', 'A'], ['B', 'C', 'B'],
                                    ['B', 'C', 'C'], ['C', 'A', 'A'], ['C', 'A', 'B'], ['C', 'A', 'C'], ['C', 'B', 'A'],
                                    ['C', 'B', 'B'], ['C', 'B', 'C'], ['C', 'C', 'A'], ['C', 'C', 'B']])
        result = ot.create_unique_combinations(elements, n_repeat)
        self.assertTrue(np.array_equal(result, expected_result))

        elements = [0, 1]
        n_repeat = 4
        expected_result = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0],
                                    [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0],
                                    [1, 1, 0, 1], [1, 1, 1, 0]])
        result = ot.create_unique_combinations(elements, n_repeat)
        self.assertTrue(np.array_equal(result, expected_result))

        elements = [[0,0], [1,1]]
        n_repeat = 2
        expected_result = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]])
        result = ot.create_unique_combinations(elements, n_repeat)
        self.assertTrue(np.array_equal(result, expected_result))
        
        
        elements = ['A', 'B', 'C','D', 'E', 'F']
        n_repeat = 2
        expected_result = np.array([['A', 'B'], ['A', 'C'], ['A', 'D'], ['A', 'E'], ['A', 'F'], ['B', 'A'], ['B', 'C'],
                                    ['B', 'D'], ['B', 'E'], ['B', 'F'], ['C', 'A'], ['C', 'B'], ['C', 'D'], ['C', 'E'],
                                    ['C', 'F'], ['D', 'A'], ['D', 'B'], ['D', 'C'], ['D', 'E'], ['D', 'F'], ['E', 'A'],
                                    ['E', 'B'], ['E', 'C'], ['E', 'D'], ['E', 'F'], ['F', 'A'], ['F', 'B'], ['F', 'C'],
                                    ['F', 'D'], ['F', 'E']])
        result = ot.create_unique_combinations(elements, n_repeat)
        self.assertTrue(np.array_equal(result, expected_result))
        
        
        
    def test_qubit_label_to_list_index(self):
        self.assertEqual(ot.qubit_label_to_list_index(3, 5), 1)
        
        hash_1 = np.array([0, 0, 1, 1])
        n_hash_symbols = 2
        self.assertTrue(np.array_equal(ot.qubit_label_to_list_index(hash_1, n_hash_symbols), np.array([1, 1, 0, 0])))
        
        hash_2 = np.array([0, 1, 0, 1])
        n_hash_symbols = 2
        self.assertTrue(np.array_equal(ot.qubit_label_to_list_index(hash_2, n_hash_symbols), np.array([1, 0, 1, 0])))
        
        hash_3 = np.array([0, 1, 1, 0])
        n_hash_symbols = 10
        self.assertTrue(np.array_equal(ot.qubit_label_to_list_index(hash_3, n_hash_symbols), np.array([9, 8, 8, 9])))
        
        hash_4 = np.array([0, 1, 2, 3])
        n_hash_symbols = 4
        self.assertTrue(np.array_equal(ot.qubit_label_to_list_index(hash_4, n_hash_symbols), np.array([3, 2, 1, 0])))
        
        hash_5 = np.array([0, 1, 2, 3, 4, 5])
        n_hash_symbols = 6
        self.assertTrue(np.array_equal(ot.qubit_label_to_list_index(hash_5, n_hash_symbols), np.array([5, 4, 3, 2, 1, 0])))
        
        
        
    def test_hash_to_instruction(self):
        hash_function = np.array([0, 1, 0])
        instruction_list =np.array( ['A', 'B', 'C'])
        n_hash_symbols = 2
        expected_result = np.array([['B', 'A', 'B'], ['C', 'A', 'C'], ['A', 'B', 'A'], ['C', 'B', 'C'], ['A', 'C', 'A'], ['B', 'C', 'B']])
        result = ot.hash_to_instruction(hash_function, instruction_list, n_hash_symbols)
        self.assertTrue(np.all(result == expected_result))  
        
        hash_function = np.array([0, 2, 1])
        instruction_list =np.array( ['A', 'B'])
        n_hash_symbols = 3
        expected_result = np.array([['B', 'A', 'A'], ['A', 'A', 'B'], ['B', 'A', 'B'], ['A', 'B', 'A'], ['B', 'B', 'A'], ['A', 'B', 'B']])
        result = ot.hash_to_instruction(hash_function, instruction_list, n_hash_symbols)
        self.assertTrue(np.all(result == expected_result))  
        
        
    def test_instruction_equivalence(self):
        # Define the possible instructions and their equivalence
        possible_instructions = ['A', 'B', 'C']
        instruction_equivalence = ['X', 'Y', 'Z']

        # Test case 1: Single instruction
        instruction = ['A']
        expected_output = np.array(['X'])
        self.assertTrue(np.array_equal(ot.instruction_equivalence(instruction, possible_instructions, instruction_equivalence), expected_output))

        # Test case 2: Multiple instructions
        instruction = ['A', 'B', 'C']
        expected_output = np.array(['X', 'Y', 'Z'])
        self.assertTrue(np.array_equal(ot.instruction_equivalence(instruction, possible_instructions, instruction_equivalence), expected_output))


        # Test case 3: Multiple instructions
        instruction_equivalence = [0, 1, 2]
        instruction = ['A', 'B', 'C']
        expected_output = np.array([0, 1, 2])
        self.assertTrue(np.array_equal(ot.instruction_equivalence(instruction, possible_instructions, instruction_equivalence), expected_output))
        
        
        
        # Test matrix instructions
        instruction = ['A', 'B', 'C']
        instruction_equivalence =[np.eye(2), np.ones((2,2)), np.array([[1,0],[0,-1]])] 
        #instruction_equivalence = [0, 1, 2]
        expected_output = instruction_equivalence
        self.assertTrue(np.array_equal(ot.instruction_equivalence(instruction, possible_instructions, instruction_equivalence), expected_output))
        
    
    # def test_frequency_donconvertion(self):
    #     subsystem_index = np.array([3,2])
    #     outcome_frequencies= np.arange(5*16).reshape(5,16)
    #     downconverted_freq = ot.downconvert_frequencies(subsystem_index,outcome_frequencies)
    #     self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,0,:,:])))
    #     self.assertTrue(np.all(downconverted_freq[0,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,1,:,:])))
    #     self.assertTrue(np.all(downconverted_freq[0,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,1,0,:,:])))
    #     self.assertTrue(np.all(downconverted_freq[1,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[1,1,1,:,:])))
    #     subsystem_index = np.array([2,0])
    #     outcome_frequencies= np.arange(5*16).reshape(5,16)
    #     #print(outcome_frequencies)
    #     downconverted_freq = ot.downconvert_frequencies(subsystem_index,outcome_frequencies)
        
    #     #print(np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,:,0,:,0]))
    #     self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,:,0,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[0,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,:,1,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[2,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,:,0,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,:,1,:,0])))
        
    #     subsystem_index = np.array([3,1])
    #     outcome_frequencies= np.arange(5*16).reshape(5,16)
    #     downconverted_freq = ot.downconvert_frequencies(subsystem_index,outcome_frequencies)
    #     self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,:,0,:])))
    #     self.assertTrue(np.all(downconverted_freq[3,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[3,0,:,1,:])))
    #     self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,1,:,0,:])))
    #     self.assertTrue(np.all(downconverted_freq[1,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[1,1,:,1,:])))
        
    #     subsystem_index = np.array([3,0])
    #     outcome_frequencies= np.arange(5*16).reshape(5,16)
    #     downconverted_freq = ot.downconvert_frequencies(subsystem_index,outcome_frequencies)
        
    #     self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[0,0,:,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[3,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[3,0,:,:,1])))
    #     self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[2,1,:,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[4,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2)[4,1,:,:,1])))
        
    #     # Test different size systems
    
    #     subsystem_index = np.array([3,0])
    #     outcome_frequencies= np.arange(5*32).reshape(5,32)
    #     downconverted_freq = ot.downconvert_frequencies(subsystem_index,outcome_frequencies)
        
    #     self.assertTrue(np.all(downconverted_freq[0,0] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[0,:,0,:,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[3,1] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[3,:,0,:,:,1])))
    #     self.assertTrue(np.all(downconverted_freq[2,2] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[2,:,1,:,:,0])))
    #     self.assertTrue(np.all(downconverted_freq[4,3] == np.sum(outcome_frequencies.reshape(5,2,2,2,2,2)[4,:,1,:,:,1])))
                
        
    def test_get_traced_out_indicies(self):
        index_to_keep = np.array([0,3])
        traced_out = ot.get_traced_out_indicies(index_to_keep,4)
        self.assertTrue(np.all(traced_out == np.array([1,2])))
        
        traced_out = ot.get_traced_out_indicies(index_to_keep,5)
        self.assertTrue(np.all(traced_out == np.array([1,2,4])))
        
        index_to_keep = np.array([0])
        traced_out = ot.get_traced_out_indicies(index_to_keep,5)
        self.assertTrue(np.all(traced_out == np.array([1,2,3,4])))
        
        index_to_keep = np.array([1,2,3])
        traced_out = ot.get_traced_out_indicies(index_to_keep,3)
        self.assertTrue(np.all(traced_out == np.array([])))
        
        
        index_to_keep = np.array([2])
        traced_out = ot.get_traced_out_indicies(index_to_keep,3)
        self.assertTrue(np.all(traced_out == np.array([0,1])))
        
        traced_out = ot.get_traced_out_indicies(index_to_keep,11)
        self.assertTrue(np.all(traced_out == np.array([0,1,3,4,5,6,7,8,9,10])))
        
    def test_create_2RDM_hash(self):
        # Test case 1: n_total_qubits = 2
        n_total_qubits = 2
        expected_hash_family = np.array([[0, 1]])
        self.assertTrue(np.array_equal(ot.create_2RDM_hash(n_total_qubits), expected_hash_family))       
        
        
        # Test case 2: n_total_qubits = 3
        n_total_qubits = 3
        expected_hash_family = np.array([[0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.array_equal(ot.create_2RDM_hash(n_total_qubits), expected_hash_family))

        # Test case 3: n_total_qubits = 4
        n_total_qubits = 4
        expected_hash_family = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        self.assertTrue(np.array_equal(ot.create_2RDM_hash(n_total_qubits), expected_hash_family))

        # Test case 4: n_total_qubits = 5
        n_total_qubits = 5
        expected_hash_family = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]])
        self.assertTrue(np.array_equal(ot.create_2RDM_hash(n_total_qubits), expected_hash_family))



    def test_check_qubit_pairs(self):
       
        subsystem_labels1 = np.array([[0, 1], [2, 3], [4, 5]])
        expected1 = np.array([[0, 1], [2, 3], [0, 1]])
        n_total_qubits = 4
        self.assertTrue(np.all(ot.check_qubit_pairs(subsystem_labels1, n_total_qubits)== expected1))

 
        subsystem_labels2 = np.array([[0, 0], [1, 1], [3, 4]])
        expected2 = np.array([[0, 1], [1, 2], [3, 0]])
        self.assertTrue(np.all(ot.check_qubit_pairs(subsystem_labels2,n_total_qubits)== expected2))


        subsystem_labels3 = np.array([[0, 1], [1, 1], [2, 3], [3, 3]])
        expected3 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        self.assertTrue(np.all(ot.check_qubit_pairs(subsystem_labels3, n_total_qubits)== expected3))


        # Test case 4: Empty input
        subsystem_labels4 = np.array([])
        expected4 = np.array([])
        self.assertTrue(np.all(ot.check_qubit_pairs(subsystem_labels4, n_total_qubits) ==  expected4))
    
    
    def test_find_2PC_cluster(self):
        # Test case 1
        subsystem_labels = np.array([[0, 1], [1, 0], [2, 3], [3, 4], [0,2],[2,0],[0,4],[1,3],[2,4]])
        quantum_correlation_array = np.array([0.8, 0.6, 0.4, 0.2, 0.9,1,0.2,0.3,1])
        two_point_qubit_labels = np.array([[0, 1],[1,2]])
        max_clusters = 3
        expected1 = np.array([[0, 1, 2],[1,2,4]])
        #print(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters))
        self.assertTrue(np.array_equal(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters), expected1))

            

            
            
        # Test case 2
        subsystem_labels = np.array([[0, 1], [1, 2], [0,2], [1, 3], [1, 4],[0,5], [5, 4],[0,3],[0,4]])
        quantum_correlation_array = np.array([0.05, 0.9, 0.9, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        two_point_qubit_labels = np.array([[0, 1]])
        max_clusters = 4
        expected2 = np.array([[0, 1, 2 ,4]])
        #print(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters))
        self.assertTrue(np.array_equal(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters), expected2))

        # Test case 3
        subsystem_labels = np.array([[0, 1], [1, 2], [0,2], [1, 3], [1, 4],[0,5], [5, 4],[0,3],[0,4]])
        quantum_correlation_array = np.array([0.05, 0.9, 0.9, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        two_point_qubit_labels = np.array([[0, 1]])
        max_clusters = 2
        expected3 = np.array([[0, 1]])
        self.assertTrue(np.array_equal(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters), expected3))
        
        
        # the the order of qubits in the subsystem labels should not matter
        subsystem_labels = np.array([[0, 1], [1,2],[3,1],[1,4], [0,4]])
        quantum_correlation_array = np.array([0.05, 0.1, 0.9, 0.3,0.1])
        two_point_qubit_labels = np.array([[0, 1]])
        max_clusters = 3
        expected4 = np.array([[0, 1, 3]])
        #print(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters))
        
        self.assertTrue(np.array_equal(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters), expected4)) 

    def test_conditioned_trace_out_POVM(self):
        # Test tracing down two qubit to one. 
        povm = POVM.generate_computational_POVM(2)[0]
        single_qubit_povm = POVM.generate_computational_POVM(1)[0].get_POVM()
        trace_out = np.array([0])
        new_povm = ot.conditioned_trace_out_POVM(povm,trace_out)
        self.assertTrue(np.all(new_povm[0] == single_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[1] == single_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[2] == single_qubit_povm[1]))
        self.assertTrue(np.all(new_povm[3] == single_qubit_povm[1]))
        
        trace_out = np.array([1])
        new_povm = ot.conditioned_trace_out_POVM(povm,trace_out)
        self.assertTrue(np.all(new_povm[0] == single_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[1] == single_qubit_povm[1]))
        self.assertTrue(np.all(new_povm[2] == single_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[3] == single_qubit_povm[1]))
        
        
        # Trace down 3 to 2
        povm = POVM.generate_computational_POVM(3)[0]
        two_qubit_povm = POVM.generate_computational_POVM(2)[0].get_POVM()
        trace_out = np.array([0])
        new_povm = ot.conditioned_trace_out_POVM(povm,trace_out)
        self.assertTrue(np.all(new_povm[0] == two_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[1] == two_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[2] == two_qubit_povm[1]))
        self.assertTrue(np.all(new_povm[3] == two_qubit_povm[1]))
        
        # Test tracing out more than one dimension
        povm = POVM.generate_computational_POVM(4)[0]
        two_qubit_povm = POVM.generate_computational_POVM(2)[0].get_POVM()
        trace_out = np.array([0,1])
        new_povm = ot.conditioned_trace_out_POVM(povm,trace_out)
        self.assertTrue(np.all(new_povm[0] == two_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[1] == two_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[2] == two_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[3] == two_qubit_povm[0]))
        # Check deep 
        self.assertTrue(np.all(new_povm[4] == two_qubit_povm[1]))
        self.assertTrue(np.all(new_povm[8] == two_qubit_povm[2]))
        self.assertTrue(np.all(new_povm[12] == two_qubit_povm[3]))
        
        # Chanee order 
        trace_out = np.array([3,2])
        new_povm = ot.conditioned_trace_out_POVM(povm,trace_out)
        self.assertTrue(np.all(new_povm[0] == two_qubit_povm[0]))
        self.assertTrue(np.all(new_povm[1] == two_qubit_povm[1]))
        self.assertTrue(np.all(new_povm[2] == two_qubit_povm[2]))
        self.assertTrue(np.all(new_povm[3] == two_qubit_povm[3]))
        

    def test_get_cluster_index_from_correlator_labels(self):
        two_point_corr = [[11,7],[3,11],[7,0],[0,11],[3,4]]  
        cluster_labels = [[11, 10, 9], [8, 7, 6, 5], [4, 3], [2, 1, 0]]
        expected_results = [[0,1], [2,0],[1,3], [3,0], [2]]
        cluster_index = [ot.get_cluster_index_from_correlator_labels(cluster_labels,two_point) for two_point in two_point_corr]
        print(cluster_index)
        self.assertTrue(expected_results == cluster_index)
        
        # Check that it is false
        two_point_corr = [6,7]
        expected_results = [0,1]
        cluster_index = ot.get_cluster_index_from_correlator_labels(cluster_labels,two_point_corr)
        self.assertFalse(expected_results == cluster_index)
    
    
    
    def test_reduce_cluster_POVMs(self):
        
        
        povm_A = POVM.generate_computational_POVM(3)[0]
        povm_B = POVM.generate_computational_POVM(2)[0]
        povm_list = [povm_A,povm_B]
        subsystem_label_list = [[3,1,0], [4,2]]
        correlator = [0,4]

        a = ot.reduce_cluster_POVMs(povm_list,subsystem_label_list,correlator)
        povm_A_reduced = a[0]
        povm_B_reduced = a[1]
        expected_results = POVM.generate_computational_POVM(1)[0].get_POVM()
        for i in range(4): # We expect 0 to oscillate between 0 and 1
            self.assertTrue(np.all(povm_A_reduced[2*i] == expected_results[0]))
            self.assertTrue(np.all(povm_A_reduced[2*i+1] == expected_results[1]))
        # We expect it to stay constant for 2
        self.assertTrue(np.all(povm_B_reduced[0] == expected_results[0]))
        self.assertTrue(np.all(povm_B_reduced[1] == expected_results[0]))
        self.assertTrue(np.all(povm_B_reduced[2] == expected_results[1]))
        self.assertTrue(np.all(povm_B_reduced[3] == expected_results[1]))
        
        
        # Check case where both correlators are in the same cluster, expect a 2 qubit POVM. 
        subsystem_label_list = [[3,1,0]]
        correlator = [3,0]
        povm_list = [povm_A]
        expected_results = POVM.generate_computational_POVM(2)[0].get_POVM()
        expected_2_qubit_order=[0,1,0,1,2,3,2,3]
        povm =ot.reduce_cluster_POVMs(povm_list,subsystem_label_list,correlator)[0]
        #print(povm)
        for i in range(len(povm)):
            self.assertTrue(np.all(povm[i] == expected_results[expected_2_qubit_order[i]]))
            
        # # Test failsafe
        # subsystem_label_list = [[3,0]]
        # correlator = [3,0,1]
        # povm_list = [povm_B]
        # povm = ot.reduce_cluster_POVMs(povm_list,subsystem_label_list,correlator)
        # self.assertIsNone(povm)   
        
    def test_create_chunk_index_array(self):
        size_array = np.array([2,2,2])
        chunk_size = 2
        index_array = ot.create_chunk_index_array(size_array, chunk_size)
        true_array = np.array([0,1,2,3])

        self.assertTrue(np.all(index_array == true_array))
        
        chunk_size = 3
        size_array = np.array([1,2,2,1,3])
        true_array = np.array([0,2,4,5])
        index_array = ot.create_chunk_index_array(size_array, chunk_size)
        self.assertTrue(np.all(index_array == true_array))
        
        chunk_size = 6
        size_array = np.array([4,2,2,2,2,6,5,1])
        true_array = np.array([0,2,5,6,8])
        index_array = ot.create_chunk_index_array(size_array, chunk_size)
        self.assertTrue(np.all(index_array == true_array))
        
        
        chunk_size = 8 
        size_array = np.array([4,3,1,8,7,1])
        index_array = ot.create_chunk_index_array(size_array, chunk_size)
        true_array = np.array([0,3,4,6])
        self.assertTrue(np.all(index_array == true_array))


    def test_swap_qubits(self):
        # check basic swap
        base_rho = np.array([sf.generate_random_pure_state(1) for _ in range(2)])
        qubit_labels = np.array([0,1])
        swap_labels = np.array([1,0])
        rho = reduce(np.kron, base_rho)
        swapped_rho = reduce(np.kron,base_rho[::-1])
        test_rho = ot.swap_qubits(rho, qubit_labels, swap_labels)
        self.assertTrue(np.allclose(swapped_rho, test_rho))
        
        # Check larger qubit size
        n_qubits = 5
        base_rho = np.array([sf.generate_random_pure_state(1) for _ in range(n_qubits)])
        qubit_labels = np.array([5,1,2,4,8])
        swap_labels = np.array([5,8])
        rho = reduce(np.kron, base_rho)
        swapped_rho = reduce(np.kron,base_rho[[4,1,2,3,0]])
        test_rho = ot.swap_qubits(rho, qubit_labels, swap_labels)
        self.assertTrue(np.allclose(swapped_rho, test_rho))
        
        # Test non-edge states:
        n_qubits = 5
        base_rho = np.array([sf.generate_random_pure_state(1) for _ in range(n_qubits)])
        qubit_labels = np.array([5,1,2,4,8])
        swap_labels = np.array([1,2])
        rho = reduce(np.kron, base_rho)
        swapped_rho = reduce(np.kron,base_rho[[0,2,1,3,4]])
        test_rho = ot.swap_qubits(rho, qubit_labels, swap_labels)
        self.assertTrue(np.allclose(swapped_rho, test_rho))
        
        # Swap swap order
        swap_labels = np.array([8,5])
        swapped_rho = reduce(np.kron,base_rho[[4,1,2,3,0]])
        test_rho = ot.swap_qubits(rho, qubit_labels, swap_labels)
        self.assertTrue(np.allclose(swapped_rho, test_rho))
        
        # Test double swap
        n_qubits = 5
        base_rho = np.array([sf.generate_random_pure_state(1) for _ in range(n_qubits)])
        qubit_labels = np.array([5,1,2,4,8])
        swap_labels = np.array([5,8])
        rho = reduce(np.kron, base_rho)
        test_rho = ot.swap_qubits(rho, qubit_labels, swap_labels)
        test_rho = ot.swap_qubits(test_rho, qubit_labels, swap_labels)
        self.assertTrue(np.allclose(rho, test_rho))
        
        # Test with entangled states
        rho = np.kron(sf.generate_random_pure_state(1), sf.generate_random_pure_state(2))
        qubit_labels = np.array([2,1,0])
        swap_labels = np.array([0,2])
        swapped_rho = ot.swap_qubits(rho, qubit_labels, swap_labels)
        traced_down_swapped_rho = ot.trace_down_qubit_state(swapped_rho,qubit_labels,np.array([1,0]))
        true_traced_down_rho = ot.trace_down_qubit_state(rho,qubit_labels, np.array([2,1]))
        self.assertTrue(np.allclose(traced_down_swapped_rho, true_traced_down_rho))
        
        
    def test_POVM_sort(self):
        povm = POVM.generate_computational_POVM(2)[0]
        swap_order = np.array([1,0])
        swapped_POVM = ot.POVM_sort(povm,swap_order)[0]
        povm_array = povm.get_POVM()
        swapped_array = swapped_POVM.get_POVM()
        self.assertTrue(np.all(povm_array == swapped_array))
        # Modify some entries
        povm_array[1,0,0] = 100
        povm_array[1,1,1] = 1000
        swapped_POVM = ot.POVM_sort(POVM(povm_array),swap_order)[0]
        swapped_array = swapped_POVM.get_POVM()
        self.assertTrue(np.real(swapped_array[2,0,0]) == 100 )
        self.assertTrue(np.real(swapped_array[2,2,2]) == 1000)
        
        povm_a = POVM.generate_random_POVM(2,2)
        povm_b = POVM.generate_random_POVM(2,2)
        povm_ab = POVM.tensor_POVM(povm_a,povm_b)[0]
        povm_ba = POVM.tensor_POVM(povm_b,povm_a)[0]
        sortin_index = np.array([1,0])
        swapped_ba = ot.POVM_sort(povm_ab,sortin_index)[0]
        self.assertTrue(np.all(povm_ba.get_POVM() == swapped_ba.get_POVM()))
        
        # Make sure swapped orderd does not work
        sortin_index = np.array([0,1])
        swapped_ba = ot.POVM_sort(povm_ab,sortin_index)[0]
        self.assertFalse(np.all(povm_ba.get_POVM() == swapped_ba.get_POVM()))
        
        # check tensoring
        povm_c = POVM.generate_random_POVM(2,2)
        povm_abc = POVM.tensor_POVM(povm_ab,povm_c)[0]      
        povm_bc = POVM.tensor_POVM(povm_b,povm_c)[0]
        povm_abc_2 = POVM.tensor_POVM(povm_a,povm_bc)[0]
        self.assertTrue(np.allclose(povm_abc.get_POVM(), povm_abc_2.get_POVM()))
        
        # 3 Qubit
        povm_cba = POVM.tensor_POVM(povm_c,povm_ba)[0]
        povm_cab = POVM.tensor_POVM(povm_c,povm_ab)[0]
        sortin_index = np.array([0,2,1])
        swapped_POVM = ot.POVM_sort(povm_cab,sortin_index)[0]
        self.assertTrue(np.allclose(povm_cba.get_POVM(), swapped_POVM.get_POVM()))
        
        
        sortin_index = np.array([2,0,1])
        swapped_POVM = ot.POVM_sort(povm_cab,sortin_index)[0]
        povm_bca = POVM.tensor_POVM(povm_bc,povm_a)[0]
        self.assertTrue(np.allclose(povm_bca.get_POVM(), swapped_POVM.get_POVM()))
        
        # Double swap will generally not work, execpt for some instances where the swap is cyclic. 
        # Here just the two first indecies are swapped. 
        sortin_index = np.array([1,0,2])
        swapped_POVM = ot.POVM_sort(povm_abc,sortin_index)[0]
        swapped_POVM = ot.POVM_sort(swapped_POVM,sortin_index)[0]
        self.assertTrue(np.allclose(povm_abc.get_POVM(), swapped_POVM.get_POVM()))
        
        # Make sure correct ordere does nothing
        sortin_index = np.array([0,1,2])
        swapped_POVM = ot.POVM_sort(povm_abc,sortin_index)[0]
        self.assertTrue(np.allclose(povm_abc.get_POVM(), swapped_POVM.get_POVM()))
        
        # 4 qubit test
        povm_d = POVM.generate_random_POVM(2,2)
        povm_abcd = POVM.tensor_POVM(povm_abc,povm_d)[0]
        sortin_index = np.array([3,2,0,1])
        povm_dcab = POVM.tensor_POVM(povm_d,povm_cab)[0]
        swapped_POVM = ot.POVM_sort(povm_abcd,sortin_index)[0]
        self.assertTrue(np.allclose(povm_dcab.get_POVM(), swapped_POVM.get_POVM()))
        #swap again
        swapped_POVM = ot.POVM_sort(swapped_POVM,sortin_index)[0]
        povm_bad = POVM.tensor_POVM(povm_ba,povm_d)[0]
        povm_badc = POVM.tensor_POVM(povm_bad,povm_c)[0]
        self.assertTrue(np.allclose(povm_badc.get_POVM(), swapped_POVM.get_POVM()))


    def test_tensor_chunk_states(self):
        rho_true_list = [sf.generate_random_pure_state(2) for _ in range(2)]
        rho_labels = [[0,1],[2,3]]
        povm_labels = [[0,1],[2,3]]
        two_point = [[0,2]]
        true_state = reduce(np.kron, rho_true_list)
        rho, _ = ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        self.assertTrue(np.allclose(true_state, rho[0]))
        
        
        two_point = [[0,1]]
        true_state = rho_true_list[0]
        rho, _ = ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        self.assertTrue(np.allclose(true_state, rho[0]))
        
        rho_true_list = [sf.generate_random_pure_state(3) for _ in range(3)]
        
        rho_labels = [[0,1,2],[3,4,5],[6,7,8]]
        povm_labels = [[0,1,2,3,4,5,6,7,8]]
        two_point = [[0,1],[1,2],[0,2],[4,6]]
        true_state = reduce(np.kron, rho_true_list)
        rho, _ = ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        for rho_sample in rho:
            self.assertTrue(np.allclose(true_state, rho_sample))
            
        povm_labels = [[0],[1],[2],[3],[4],[5],[6],[7],[8]]
        two_point = [[0,1]]
        rho, _ = ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        self.assertTrue(np.allclose(rho_true_list[0], rho[0]))
        
        two_point = [[2,3]]
        rho , _= ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        expected_rho = np.kron(rho_true_list[0],rho_true_list[1])
        self.assertTrue(np.allclose(expected_rho, rho[0]))
        
        two_point = [[4,6]]
        rho, _ = ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        expected_rho = np.kron(rho_true_list[1],rho_true_list[2])
        self.assertTrue(np.allclose(expected_rho, rho[0]))
        
        povm_labels = [[0,1,2,3,4],[5],[6],[7],[8]]
        two_point = [[0,1]]
        rho, _ = ot.tensor_chunk_states(rho_true_list, rho_labels, povm_labels, two_point)
        expected_rho = np.kron(rho_true_list[0],rho_true_list[1])
        self.assertTrue(np.allclose(expected_rho, rho[0]))
        
        

        
if __name__ == '__main__':
    unittest.main()
