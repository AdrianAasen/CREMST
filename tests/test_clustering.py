import unittest
import numpy as np
from functools import reduce
import sys
sys.path.append('../') # Adding path to library
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot
from EMQST_lib import clustering as cl



class TestCluster(unittest.TestCase):

    def test_is_pair_in_more_than_one_cluster(self):
        pair_label = [1, 2]
        clusters = [[1, 2], [3,4], [5,6 ]]
        result = cl.is_pair_in_more_than_one_cluster(pair_label, clusters)
        self.assertFalse(result)

        pair_label = [3, 4]
        result = cl.is_pair_in_more_than_one_cluster(pair_label, clusters)
        self.assertFalse(result)

        pair_label = [1, 3]
        result = cl.is_pair_in_more_than_one_cluster(pair_label, clusters)
        print(result)
        self.assertTrue(result)

        pair_label = [1, 6]
        result = cl.is_pair_in_more_than_one_cluster(pair_label, clusters)
        self.assertTrue(result)
        
    def test_find_clusters_from_correlator_labels(self):
        pair_label = [[4,2],[2,1]]
        clusters = [[4, 3], [2, 1, 0], [], [11, 10, 9], [], [8, 7, 6, 5]]
        expected_result = [[[4,3],[2,1,0]], [[2,1,0]]]
        result = cl.find_clusters_from_correlator_labels(pair_label,clusters)
        self.assertTrue(expected_result==result)
        
        pair_label = [[10,0],[0,10]] # Check that order does not matter
        expected_result = [ [[2,1,0], [11,10,9]], [[2,1,0], [11,10,9]]]
        result = cl.find_clusters_from_correlator_labels(pair_label,clusters)
        self.assertTrue(expected_result==result)
        

    def test_are_sublists_equal(self):
        # Test case 1: Identical lists
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[1, 2], [3, 4], [5, 6]]
        self.assertTrue(cl.are_sublists_equal(list1, list2))

        # Test case 2: Lists with same elements but different order
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[5, 6], [1, 2], [3, 4]]
        self.assertTrue(cl.are_sublists_equal(list1, list2))

        # Test case 3: Lists with different elements
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[1, 2], [3, 4], [7, 8]]
        self.assertFalse(cl.are_sublists_equal(list1, list2))

        # Test case 4: Lists with different lengths
        list1 = [[1, 2], [3, 4], [5, 6]]
        list2 = [[1, 2], [3, 4]]
        self.assertFalse(cl.are_sublists_equal(list1, list2))

        # Test case 5: Empty lists
        list1 = []
        list2 = []
        self.assertTrue(cl.are_sublists_equal(list1, list2))

        # Test case 6: One empty list and one non-empty list
        list1 = []
        list2 = [[1, 2]]
        self.assertFalse(cl.are_sublists_equal(list1, list2))
        
        list1 = [[1, 2]]
        list2 = [[0,3], [4,1], [2,5]]
        self.assertFalse(cl.are_sublists_equal(list1, list2))
        
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
