import numpy as np
import os
import time as t
import pickle
import uuid
import sys
from EMQST_lib.qrem import QREM
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot
from EMQST_lib import clustering as cl
#from EMQST_lib import qrem as qrem
from EMQST_lib.povm import POVM
np.set_printoptions(precision=3)

first_start_time = t.time()
start_time = t.time()

with open('16_qubit_example_qrem_default.pkl', "rb") as f:
    qrem = pickle.load(f)

# Updates passed core count if availabe
if len(sys.argv) > 1:
    new_n_cores = int(sys.argv[1])
    if new_n_cores > qrem._n_cores:
        qrem._n_cores = new_n_cores
        print(f'Updated core count to {new_n_cores}.')

qrem.print_current_state()

# qrem.set_initial_cluster_size(cluster_size)
print('Starting QDT measurements')
qrem.perform_QDT_measurements()
#qrem.copy_chunked_true_states(qrem_default)
qrem.save_initialization()

print(f'QDT measurements and initialization took {t.time() - start_time:.2f} seconds')

# Compute clusters and save dendrogram data
start_time = t.time()
print('Starting clustering')
qrem.perform_clustering()
print(f'Clustering took {t.time() - start_time:.2f} seconds')

start_time = t.time()
qrem.reconstruct_all_one_qubit_POVMs()
print(f'Reconstructing all one-qubit POVMs took {t.time() - start_time:.2f} seconds')

start_time = t.time()
qrem.reconstruct_cluster_POVMs()
print(f'Reconstructing cluster POVMs took {t.time() - start_time:.2f} seconds')

start_time = t.time()
qrem.compute_correlator_true_states()
print(f'Computing correlator true states and two-point POVMs took {t.time() - start_time:.2f} seconds')



print('Starting parallel correlator QST')
start_time = t.time()
comparison_methods = [0,1,2,3]
result_dict = qrem.perform_correlated_QREM_comparison(comparison_methods)
# Extend the save_path such that we can do statistical averaging with error bars. 
save_path = f'{qrem.data_path}/{str(uuid.uuid4())}'
os.mkdir(save_path)
with open(f'{save_path}/result_QST.pkl', 'wb') as f:
    print(f'Saving results to {save_path}/result_QST.pkl.')	
    pickle.dump(result_dict, f)
print(f'Parallel correlator QST took {t.time() - start_time:.2f} seconds')
print(f'Total run time: {t.time() - first_start_time:.2f} seconds')
