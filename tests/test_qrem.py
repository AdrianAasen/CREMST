import unittest
import numpy as np
import scipy as sp
import sys
sys.path.append('../')
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot
from EMQST_lib import clustering as cl
from EMQST_lib.qrem import QREM
from EMQST_lib.povm import POVM


class TestQREM(unittest.TestCase):
    def test_set_coherent_POVM_array(self):
        base_path = 'QDOT_results/iswap_example'
        data_path = sf.generate_data_folder(base_path)

        sim_dict ={
            'n_qubits': 16,
            'n_QST_shots_total': 10**3,
            'n_QDT_shots': 10**4,
            'n_QDT_hash_symbols': 4,
            'n_QST_hash_symbols': 4,
            'n_cores': 7,
            'max_cluster_size': 3,
            'data_path': data_path,
        }
        cluster_size = np.array([1,2,1,3,1,4], dtype=int)
        qrem = QREM(sim_dict)
        qrem.set_initial_cluster_size(cluster_size)
        noise = np.pi/10
        qrem.set_coherent_POVM_array(angle = noise)
        for i in range(len(qrem._povm_array)): # Check that dimensions are correct
            self.assertTrue(qrem._povm_array[i].get_POVM().shape[0] == 2**cluster_size[i] )

        # Check that first POVM is simple X rotation
        rot_matrix = sf.rot_about_collective_X(noise, 1)
        comp_povm = POVM.generate_computational_POVM(1)[0].get_POVM()
        true_povm = np.einsum("ij,kjl,lm->kim",rot_matrix,comp_povm,rot_matrix.conj().T) # We use the inverse of the rotation matrix on POVMs as opposed to states.
        self.assertTrue(np.allclose(qrem._povm_array[0].get_POVM(), true_povm))
        
        # Check the 3 qubit case.
        X = np.array([[0,1],[1,0]], dtype=complex)
        Id = np.eye(2, dtype=complex)
        H = np.kron(np.kron(X,X),Id) + np.kron(np.kron(Id,X),X)
        rot_matrix = sp.linalg.expm(-1j*noise*H/2)
        #rot_matrix = sf.rot_about_collective_X(noise, 2)

        comp_povm = POVM.generate_computational_POVM(3)[0].get_POVM()
        true_povm = np.einsum("ij,kjl,lm->kim",rot_matrix,comp_povm,rot_matrix.conj().T) 
        self.assertTrue(np.allclose(qrem._povm_array[3].get_POVM(), true_povm))