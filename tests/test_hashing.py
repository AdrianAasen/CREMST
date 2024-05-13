import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot




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
            expected4 = np.array([[0, 1, 4]])
            #print(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters))
            self.assertTrue(np.array_equal(ot.find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters), expected4)) 
if __name__ == '__main__':
    unittest.main()
