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
        
        

