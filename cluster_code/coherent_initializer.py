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
base_path = 'QDOT_results/coherent_example'
data_path = sf.generate_data_folder(base_path)
n_cores = 9
# Updates passed core count if availabe
if len(sys.argv) > 1:
    new_n_cores = int(sys.argv[1])
    if new_n_cores > n_cores:
        n_cores = new_n_cores
        print(f'Updated core count to {new_n_cores}.')

rot_angle_array = np.array([0,10,20,30])*np.pi/100
sim_dict ={
    'n_qubits': 16,
    'n_QST_shots_total': 10**5, # This is the number of shots for each two-point correlator QST.
    'n_QDT_shots': 10**4,
    'n_QDT_hash_symbols': 3,
    'n_QST_hash_symbols': 2, # We still supply this, but it is no longer used. 
    'n_cores': n_cores,
    'max_cluster_size': 3, # Have to limit cluster size in these simulations due to 4 times 4 clusters occurs which can't be efficiently computed
    'data_path': data_path,
    'rot_angle_array': rot_angle_array,
}
 
# Load random parameters
total_two_point_corr_labels = ot.generate_random_pairs_of_qubits(sim_dict["n_qubits"], 20)
#total_two_point_corr_labels= np.array([[0,1], [1,10], [6,5], [9,1], [14,2],[3,15]])

# Ensure that cluster_sizes is a numpy array of integers
cluster_size = np.array([4, 1, 3, 2, 2, 1, 1, 2], dtype=int)

n_state_averages = 10
chunk_size = 4
qrem_default = QREM(sim_dict, two_point_corr_labels = total_two_point_corr_labels)
qrem_default.set_chunked_true_states(n_averages=n_state_averages, chunk_size=chunk_size)
qrem_array = [QREM(sim_dict, two_point_corr_labels = total_two_point_corr_labels) for _ in range(len(rot_angle_array))]

for qrem, angle in zip(qrem_array, rot_angle_array):
    qrem.copy_chunked_true_states(qrem_default) # Make sure every run has the same true states
    qrem.set_initial_cluster_size(cluster_size)
    qrem.set_coherent_POVM_array(angle)
    qrem.save_initialization()
    
with open('coherent_qrem_default.pkl', 'wb') as f:
    pickle.dump(qrem_array, f)
print(f'Initialized qrem object saved to coherent_qrem_default.pkl')