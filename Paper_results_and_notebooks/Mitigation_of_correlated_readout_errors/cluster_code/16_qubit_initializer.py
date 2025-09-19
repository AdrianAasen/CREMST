import numpy as np
from EMQST_lib import support_functions as sf
from EMQST_lib.qrem import QREM
import pickle
import sys
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot

np.set_printoptions(precision=3)


# Initialize run parameters
# Generate new dictionary for current run
base_path = 'QDOT_results/16_exp_example'
data_path = sf.generate_data_folder(base_path)
n_cores = 9
# Updates passed core count if availabe
if len(sys.argv) > 1:
    new_n_cores = int(sys.argv[1])
    if new_n_cores > n_cores:
        n_cores = new_n_cores
        print(f'Updated core count to {new_n_cores}.')


sim_dict ={
    'n_qubits': 16,
    'n_QST_shots_total': 10**5, # This is the number of shots for each two-point correlator QST.
    'n_QDT_shots': 10**4,
    'n_QDT_hash_symbols': 3,
    'n_QST_hash_symbols': 2, # We still supply this, but it is no longer used. 
    'n_cores': 9,
    'max_cluster_size': 3, # Have to limit cluster size in these simulations due to 4 times 4 clusters occurs which can't be efficiently computed
    'data_path': data_path,
}
 
# Load random parameters
selected_two_point_corr_labels = np.array([[0,1], [4,8], [6,11], [7,13]])

random_two_pont_array = ot.generate_random_pairs_of_qubits(sim_dict["n_qubits"], 16)
total_two_point_corr_labels = np.concatenate((selected_two_point_corr_labels, random_two_pont_array), axis=0)

#total_two_point_corr_labels = selected_two_point_corr_labels
# Ensure that cluster_sizes is a numpy array of integers
cluster_size = np.array([4, 1, 3, 2, 2, 1, 1, 2], dtype=int)
qrem_default = QREM(sim_dict, two_point_corr_labels = total_two_point_corr_labels)
n_state_averages = 10
chunk_size = 4
qrem_default.set_chunked_true_states(n_averages=n_state_averages, chunk_size=chunk_size)
qrem_default.set_initial_cluster_size(cluster_size)
povm_mode = 'strong'
qrem_default.set_exp_POVM_array(noise_mode = povm_mode)


with open('16_qubit_example_qrem_default.pkl', 'wb') as f:
    pickle.dump(qrem_default, f)
print(f'Initialized qrem object saved to qrem_default.pkl')