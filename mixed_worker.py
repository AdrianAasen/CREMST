import os
import atexit
import shutil
import tempfile

# === TEMP DIRECTORY SETUP  ===
# Setup temp directories to avoid joblib disk space issues
temp_dir = os.path.expanduser("~/tmp")
joblib_temp_dir = os.path.join(temp_dir, f"joblib_{os.getpid()}")

# Create directories and set environment variables
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(joblib_temp_dir, exist_ok=True)

env_vars = {
    'TMPDIR': temp_dir,
    'TMP': temp_dir,
    'TEMP': temp_dir,
    'JOBLIB_TEMP_FOLDER': joblib_temp_dir,
    'JOBLIB_MULTIPROCESSING': '0',
}

for key, value in env_vars.items():
    os.environ[key] = value

tempfile.tempdir = temp_dir

# Clean up on exit
def cleanup_temp():
    try:
        if os.path.exists(joblib_temp_dir):
            shutil.rmtree(joblib_temp_dir, ignore_errors=True)
    except:
        pass

atexit.register(cleanup_temp)

# Basic imports
import numpy as np 
from datetime import datetime
import uuid
import sys

# Import EMQST modules
from EMQST_lib import adaptive_functions as ad
from EMQST_lib.qst import QST  
from EMQST_lib import measurement_functions as mf
from EMQST_lib import visualization as vis
from EMQST_lib import support_functions as sf
from EMQST_lib.povm import POVM

# === MAIN SCRIPT PARAMETERS ===
n_cores = 6
# Updates passed core count if available
if len(sys.argv) > 1:
    new_n_cores = int(sys.argv[1])
    if new_n_cores > n_cores:
        n_cores = new_n_cores
        print(f'Updated core count to {new_n_cores}.')

path = "adaptive_results/mixed_state_100k" 
now=datetime.now()
now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
dir_name= now_string+str(uuid.uuid4())
data_path=f'{path}/{dir_name}'



n_shots_total = 10**5
n_qubits = 1
n_shots = n_shots_total//3**n_qubits # In the qst code it is assumed that each single qubit measurement is a Pauli-basis, hence 3^n_qubits total measurement settings.
n_averages = 100
adaptive_burnin = 30
compute_uncertainty = True
print(f'Starting adaptive QST with {n_shots_total} shots, {n_qubits} qubits, {n_averages} averages, and adaptive burnin of {adaptive_burnin}.')

true_states = np.array([sf.generate_random_Hilbert_Schmidt_mixed_state(n_qubits) for _ in range(n_averages)])
povm = POVM.generate_Pauli_POVM(n_qubits)
decompiled_array = np.array([povm[i].get_POVM() for i in range(len(povm))])
pauli_6_array = 1/3**(n_qubits)*np.array(decompiled_array.reshape(-1,decompiled_array.shape[-2],decompiled_array.shape[-1]))
test_POVM = POVM(pauli_6_array)
qst_adaptive = QST(povm, true_states, n_shots, n_qubits, False,{}, n_cores=n_cores)

noise_levels = [0,0.05,0.15,0.25]
n_steps = len(noise_levels)
infidelity_container_nonadaptive = []
infidelity_container_adaptive = []
infidelity_container_grid = []
uncertainty_container_adaptive = []
uncertainty_container_nonadaptive = []
qst_array = []
noise_strengths = []
setting_array = []

start_time = datetime.now()
print(f'Total shots for the whole run: {n_shots_total * n_averages*n_steps}')
print(f'Starting adaptive QST with {n_shots_total} shots, {n_qubits} qubits, \n{n_steps} of noise runs, with {n_averages} averages each, and {n_shots} shots per measurement setting for adaptive QST.')
print(f'Start time: {now_string}.')

print(f"=== Pre-QST Check ===")
print(f"JOBLIB_TEMP_FOLDER: {os.environ.get('JOBLIB_TEMP_FOLDER')}")
print(f"Available space: {os.statvfs(temp_dir).f_bavail * os.statvfs(temp_dir).f_frsize / (1024**3):.2f} GB")
print("=== End Pre-QST Check ===")
for i in range(n_steps):
    # NonAdaptive QST
    
    print(f'Starting nonadaptive QST run {i+1}/{n_steps}')
    depolarizing_strength = noise_levels[i] 
    start_time = datetime.now()
    depolarized_pauli_6_array = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in pauli_6_array])
    noisy_povm = POVM(depolarized_pauli_6_array)
    # # Note that we use n_shots*3**n_qubits here, since nonadaptive QST does not split the shots between different measurement settings. 
    # # We also multiply rather than use n_shots total since there could be rounding difference between the adaptive and non-daptive QST. 
    qst_nonadaptive = QST([noisy_povm], true_states, n_shots*3**n_qubits, n_qubits, False,{}, n_cores=n_cores) # They are initalized the same
    qst_nonadaptive.generate_data()
    qst_nonadaptive.perform_BME(compute_uncertainty = compute_uncertainty)
    infidelity_container_nonadaptive.append(qst_nonadaptive.get_infidelity())
    uncertainty_container_nonadaptive.append(qst_nonadaptive.get_uncertainty())
    print(f'Nonadaptive QST run {i+1}/{n_steps} took {datetime.now()-start_time}')
    non_ada_state = qst_nonadaptive.get_rho_estm()
    # Clean up memory after nonadaptive run
    del qst_nonadaptive
    import gc
    gc.collect()

    print(f'Starting grid adaptive QST run {i+1}/{n_steps}')
    start_time = datetime.now()
    qst_adaptive = QST(povm, true_states, n_shots, n_qubits, False,{}, n_cores=n_cores)
    qst_adaptive.perform_random_adaptive_BME(depolarizing_strength = depolarizing_strength,
                            adaptive_burnin_steps = adaptive_burnin,
                            compute_uncertainty = compute_uncertainty)

    noise_strengths.append(depolarizing_strength)
    qst_array.append(qst_adaptive)
    infidelity_container_adaptive.append(qst_adaptive.get_infidelity())
    uncertainty_container_adaptive.append(qst_adaptive.get_uncertainty())
    print(f'Adaptive QST run {i+1}/{n_steps} took {datetime.now()-start_time}')
    rho_estm_adaptive = qst_adaptive.get_rho_estm()
    # Clean up memory after grid adaptive run
    #del qst_adaptive
    #gc.collect()

    # print(f'Starting adaptive QST run {i+1}/{n_steps}')
    # start_time = datetime.now()
    # qst_adaptive = QST(povm, true_states, n_shots, n_qubits, False,{}, n_cores=n_cores)
    # qst_adaptive.perform_adaptive_BME(depolarizing_strength = depolarizing_strength,
    #                         adaptive_burnin_steps = adaptive_burnin,
    #                         compute_uncertainty = compute_uncertainty)

    # noise_strengths.append(depolarizing_strength)
    # qst_array.append(qst_adaptive)
    # infidelity_container_adaptive.append(qst_adaptive.get_infidelity())
    # uncertainty_container_adaptive.append(qst_adaptive.get_uncertainty())
    # print(f'Adaptive QST run {i+1}/{n_steps} took {datetime.now()-start_time}')
    # # For non_adaptive QST we need to supply it with the noisy POVM separatly.

    # # Memory cleanup between major iterations
    # gc.collect()
    
    # Store reconstructed states before cleanup
    
    
    settings = {
        'n_qubits': n_qubits,
        'n_averages': n_averages,
        'adaptive_burnin': adaptive_burnin,
        'noise_strengths': noise_strengths,
        'true_states': true_states,
        'reconstructed_states_adaptive': rho_estm_adaptive,
        'reconstructed_states_nonadaptive': non_ada_state
    }
    setting_array.append(settings)

container_dict = {
    'adaptive_infidelity_container': infidelity_container_adaptive,
    'nonadaptive_infidelity_container': infidelity_container_nonadaptive,
    'adaptive_uncertainty_container': uncertainty_container_adaptive,
    'nonadaptive_uncertainty_container': uncertainty_container_nonadaptive,
    'grid_infidelity_container': infidelity_container_grid
}


os.mkdir(data_path)
with open(f'{data_path}/infidelity_container.npy', 'wb') as f:
    np.save(f, container_dict)
with open(f'{data_path}/settings.npy', 'wb') as f:    
    np.save(f, setting_array)
print(f'Saved to {data_path}')