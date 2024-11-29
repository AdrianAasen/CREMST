import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot




class TestCluster(unittest.TestCase):

    def test_is_pair_in_more_than_one_cluster(self):
        pair_label = [1, 2]
        clusters = [[1, 2], [3,4], [5,6 ]]
        result = ot.is_pair_in_more_than_one_cluster(pair_label, clusters)
        self.assertFalse(result)

        pair_label = [3, 4]
        result = ot.is_pair_in_more_than_one_cluster(pair_label, clusters)
        self.assertFalse(result)

        pair_label = [1, 3]
        result = ot.is_pair_in_more_than_one_cluster(pair_label, clusters)
        print(result)
        self.assertTrue(result)

        pair_label = [1, 6]
        result = ot.is_pair_in_more_than_one_cluster(pair_label, clusters)
        self.assertTrue(result)
        
    def test_find_clusters_from_correlator_labels(self):
        pair_label = [[4,2],[2,1]]
        clusters = [[4, 3], [2, 1, 0], [], [11, 10, 9], [], [8, 7, 6, 5]]
        expected_result = [[[4,3],[2,1,0]], [[2,1,0]]]
        result = ot.find_clusters_from_correlator_labels(pair_label,clusters)
        self.assertTrue(expected_result==result)
        
        pair_label = [[10,0],[0,10]] # Check that order does not matter
        expected_result = [ [[2,1,0], [11,10,9]], [[2,1,0], [11,10,9]]]
        result = ot.find_clusters_from_correlator_labels(pair_label,clusters)
        self.assertTrue(expected_result==result)
        
        
    def test_trace_down_qubit_state(self):
        n_qubits = 4
        np.random.seed(1)
        qubit_states  = [sf.generate_random_pure_state(1) for _ in range(n_qubits)]
        full_state = reduce(np.kron, qubit_states)
        state_labels = [0,1,2,3]
        trace_out_labels = [0,1]
        expected_state = np.kron(qubit_states[0], qubit_states[1])
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        self.assertTrue(np.allclose(expected_state, traced_down_state))
        
        # Change order to check if it works
        trace_out_labels = [1,0]
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        self.assertTrue(np.allclose(expected_state, traced_down_state))
        # Check state order matters
        state_labels = [1,3,0,2]
        #trace_out_labels = [0,2]
        #expected_state = np.kron(qubit_states[], qubit_states[3])
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        self.assertTrue(np.allclose(expected_state, traced_down_state))
        
        
        # Check mixed state labels
        state_labels = [0,1,10,3]
        trace_out_labels = [0,1]
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        expected_state = np.kron(qubit_states[0], qubit_states[1])
        self.assertTrue(np.allclose(expected_state, traced_down_state))
        
        # Check multiple labels to trace out
        state_labels = [0,1,10,3]
        trace_out_labels = [0,1,3]
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        expected_state = qubit_states[0]
        self.assertTrue(np.allclose(expected_state, traced_down_state))
        
        state_labels = [0,1,10,3]
        trace_out_labels = [10]
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        expected_state = reduce(np.kron, [qubit_states[1], qubit_states[2], qubit_states[3]])
        self.assertTrue(np.allclose(expected_state, traced_down_state))
        
        # Check empty trace
        
        trace_out_labels = []
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        self.assertTrue(np.allclose(full_state, traced_down_state))
        
        # Test tracing out non-existing labels
        
        state_labels = [0,1,10,3]
        trace_out_labels = [11]
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        expected_state = reduce(np.kron, [qubit_states[1], qubit_states[2], qubit_states[3]])
        self.assertTrue(np.allclose(full_state, traced_down_state))
        
        state_labels = [0,1,10,3]
        trace_out_labels = [11,1]
        traced_down_state = ot.trace_down_qubit_state(full_state, state_labels, trace_out_labels)
        expected_state = reduce(np.kron, [qubit_states[0], qubit_states[1], qubit_states[3]])
        self.assertTrue(np.allclose(expected_state, traced_down_state))


    def test_are_sublists_equal(self):
        # Test case 1: Identical lists
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[1, 2], [3, 4], [5, 6]]
        self.assertTrue(ot.are_sublists_equal(list1, list2))

        # Test case 2: Lists with same elements but different order
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[5, 6], [1, 2], [3, 4]]
        self.assertTrue(ot.are_sublists_equal(list1, list2))

        # Test case 3: Lists with different elements
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[1, 2], [3, 4], [7, 8]]
        self.assertFalse(ot.are_sublists_equal(list1, list2))

        # Test case 4: Lists with different lengths
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[1, 2], [3, 4]]
        self.assertFalse(ot.are_sublists_equal(list1, list2))

        # Test case 5: Empty lists
        list1 = []
        list2 = []
        self.assertTrue(ot.are_sublists_equal(list1, list2))

        # Test case 6: One empty list and one non-empty list
        list1 = []
        list2 = [[1, 2]]
        self.assertFalse(ot.are_sublists_equal(list1, list2))
        
        list1 = [[1, 2]]
        list2 = [[0,3], [4,1], [2,5]]
        self.assertFalse(ot.are_sublists_equal(list1, list2))
        
    def test_find_all_args_of_label_single_occurrence(self):
        corr_labels = np.array([[1, 2], [3, 4], [5, 6]])
        label_to_find = 3
        expected_result = np.array([1])
        result = ot.find_all_args_of_label(corr_labels, label_to_find)
        print(result)
        self.assertTrue(np.all(result==expected_result))

    def test_find_all_args_of_label_multiple_occurrences(self):
        corr_labels = np.array([[1, 2], [3, 4], [5, 6], [3, 7], [8, 3]])
        label_to_find = 3
        expected_result = [1, 3, 4]
        result = ot.find_all_args_of_label(corr_labels, label_to_find)
        self.assertTrue(np.all(result == expected_result))

    def test_find_all_args_of_label_no_occurrence(self):
        corr_labels = np.array([[1, 2], [3, 4], [5, 6]])
        label_to_find = 7
        expected_result = []
        result = ot.find_all_args_of_label(corr_labels, label_to_find)
        self.assertTrue(np.all(result == expected_result))

    def test_find_all_args_of_label_empty_list(self):
        corr_labels = np.array([])
        label_to_find = 1
        expected_result = []
        result = ot.find_all_args_of_label(corr_labels, label_to_find)
        self.assertTrue(np.all(result == expected_result))

    def test_find_all_args_of_label_single_element_sublists(self):
        corr_labels = np.array([[1], [2], [3], [1]])
        label_to_find = 1
        expected_result = [0, 3]
        result = ot.find_all_args_of_label(corr_labels, label_to_find)
        print(result)
        self.assertTrue(np.all(result == expected_result))
