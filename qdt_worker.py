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



import numpy as np 
from datetime import datetime
import os
import uuid
import sys
from EMQST_lib import adaptive_functions as ad
from EMQST_lib.qst import QST  
from EMQST_lib import measurement_functions as mf
from EMQST_lib import visualization as vis
from EMQST_lib import support_functions as sf
from EMQST_lib.povm import POVM
from EMQST_lib import dt as dt

n_cores = 6
# Updates passed core count if availabe
if len(sys.argv) > 1:
    new_n_cores = int(sys.argv[1])
    if new_n_cores > n_cores:
        n_cores = new_n_cores
        print(f'Updated core count to {new_n_cores}.')


path = "adaptive_results/qdt_100k_calib_100k"
now=datetime.now()
now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")



n_qst_shots_total = 10**5
n_qdt_shots_total = 10**4
n_qubits = 1
n_qst_shots = n_qst_shots_total//(3**n_qubits) # In the qst code it is assumed that each single qubit measurement is a Pauli-basis, hence 3^n_qubits total measurement settings.
n_averages = 50
adaptive_burnin = 0
compute_uncertainty = True
print(f'Starting adaptive QST with {n_qst_shots_total} shots, {n_qubits} qubits, {n_averages} averages, and adaptive burnin of {adaptive_burnin}.')

true_states = np.array([sf.generate_random_pure_state(n_qubits) for _ in range(n_averages)])
#povm = POVM.generate_Pauli_POVM(n_qubits)
#decompiled_array = np.array([povm[i].get_POVM() for i in range(len(povm))])

# comp_basis_povm = POVM.generate_computational_POVM(n_qubits)
# comp_basis_array = comp_basis_povm[0].get_POVM()

# pauli_6 = POVM.generate_Pauli_POVM(n_qubits)
# pauli_6_array = np.array([pauli_6[i].get_POVM() for i in range(len(pauli_6))])
# decomp_pauli_6_array = 1/3**(n_qubits)*np.array(pauli_6_array.reshape(-1,pauli_6_array.shape[-2],pauli_6_array.shape[-1]))

#pauli_6_array = 1/3**(n_qubits)*np.array(decompiled_array.reshape(-1,decompiled_array.shape[-2],decompiled_array.shape[-1]))
#test_POVM = POVM(decomp_pauli_6_array)
#qst_adaptive = QST(povm, true_states, n_qst_shots_total, n_qubits, False,{}, n_cores=n_cores)


noise_levels = [0.15]
qdt_multipliers = [10]
n_steps = len(qdt_multipliers)
infidelity_container_nonadaptive = []
infidelity_container_adaptive = []
infidelity_container_grid = []
uncertainty_container_adaptive = []
uncertainty_container_nonadaptive = []
qst_array = []
noise_strengths = []
setting_array = []
recon_POVM_list = []
noisy_POVM_list = []
start_time = datetime.now()

print(f'Total shots for the whole run: {n_qst_shots_total * n_averages*n_steps}')
print(f'Starting adaptive QST with {n_qst_shots_total} shots, {n_qubits} qubits, \n{n_steps} of noise runs, with {n_averages} averages each, and {n_qst_shots} shots per measurement setting for adaptive QST.')
print(f'Start time: {now_string}.')

# Perform all QDT up front


for i in range(n_steps):
    # QDT step
    depolarizing_strength = noise_levels[0]
    
    diag_recon_comp_list = []
    for _ in range(n_averages):
        comp_basis_povm = POVM.generate_computational_POVM(n_qubits)
        comp_basis_array = comp_basis_povm[0].get_POVM()
        depolarized_comp_array = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in comp_basis_array])
        noisy_comp_POVM = [POVM(depolarized_comp_array)]

        calibration_states,calibration_angles=sf.get_calibration_states(n_qubits,"SIC")
        n_calibration_shots_each =(qdt_multipliers[i] * n_qdt_shots_total)//(len(calibration_states))

        reconstructed_comp_povm = dt.device_tomography(n_qubits,n_calibration_shots_each,noisy_comp_POVM,calibration_states,n_cores=n_cores, initial_guess_POVM =comp_basis_povm )
        diag_recon_comp = np.array([np.diag(np.diagonal(element)) for element in reconstructed_comp_povm[0].get_POVM()]) # Makes recon classical only
        #POVM.generate_random_POVM(2,2).get_POVM()
        #diag_recon_comp = np.array([sf.depolarizing_channel(element, 0.2) for element in comp_basis_array])
        #print(diag_recon_comp)
        diag_recon_comp_list.append(diag_recon_comp)             
    # depolarized_comp_array = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in comp_basis_array])
    # #print('Depolarized comp basis array shape:', depolarized_comp_array)
    # noisy_comp_POVM = [POVM(depolarized_comp_array)]
    # calibration_states,calibration_angles=sf.get_calibration_states(n_qubits,"SIC")
    
    # n_calibration_shots_each = qdt_multipliers[i] * (n_qdt_shots_total//(len(calibration_states)))
    # print(n_calibration_shots_each)
    # for _ in range(10):
    #     reconstructed_comp_povm = dt.device_tomography(n_qubits,n_calibration_shots_each,noisy_comp_POVM,calibration_states,n_cores=n_cores, initial_guess_POVM =comp_basis_povm )

    #     # Create Pauli-6 from reconstructed comp povm
    #     print(f'Distance between true comp povm and reconstructed comp povm: {sf.ac_POVM_distance(reconstructed_comp_povm[0].get_POVM(),depolarized_comp_array)}')
    # reconstructed_pauli_6_array = POVM.generate_Pauli_from_comp(reconstructed_comp_povm[0])
    # decompiled_pauli_6 = 1/3**n_qubits*np.array([povm.get_POVM() for povm in reconstructed_pauli_6_array]) # Normalizsation factor 1/3^n_qubits when recombining the POVM. 
    # print(decompiled_pauli_6)
    # decompiled_pauli_6 =  np.reshape(decompiled_pauli_6, (-1, *decompiled_pauli_6.shape[2:])) # Reshape operators to be on the same level.
    
    # #print(decompiled_pauli_6.shape)
    # #print('Reconstructed pauli 6 array shape:', reconstructed_pauli_6_array)
    # recon_pauli_6_POVM = POVM(decompiled_pauli_6)
    # recon_POVM_list.append(recon_pauli_6_POVM)
    print(f'Starting nonadaptive QST run {i+1}/{n_steps}')
    start_time = datetime.now()

    # Note that we use n_shots*3**n_qubits here, since nonadaptive QST does not split the shots between different measurement settings. 
    # We also multiply rather than use n_shots total since there could be rounding difference between the adaptive and non-daptive QST. 
    # Create noisey_pauli-6 POVM
    # depolarized_pauli_6_array = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in decomp_pauli_6_array])
    # noisy_pauli_6_POVM = POVM(depolarized_pauli_6_array)



    qst_nonadaptive = QST(POVM.generate_computational_POVM(n_qubits), true_states,n_qst_shots*3**n_qubits, n_qubits, False,{}, n_cores=n_cores) # They are initalized the same
    # Update mh steps:
    #qst_nonadaptive._MH_steps = 100
    qst_nonadaptive.n_bank = 2000
    #qst_nonadaptive.generate_data()
    pauli_6 = POVM.generate_Pauli_POVM(n_qubits)
    decompiled_pauli_6 = np.array([povm.get_POVM() for povm in pauli_6]) # Normalization factor 1/3^n_qubits when recombining the POVM. 
    decompiled_pauli_6 =  1/3**n_qubits*np.reshape(decompiled_pauli_6, (-1, *decompiled_pauli_6.shape[2:])) # Reshape operators to be on the same level.
    decompiled_pauli_6 = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in decompiled_pauli_6])
    pauli_6_POVM = POVM(decompiled_pauli_6)  
    qst_nonadaptive = QST([pauli_6_POVM], true_states, n_qst_shots*3**n_qubits, n_qubits, False,{}, n_cores=n_cores) # They are initalized the same
    qst_nonadaptive.generate_data()
    
    reconstructed_pauli_6_array = POVM.generate_Pauli_from_comp(POVM(diag_recon_comp))
    #1/3**self.n_qubits*
    decompiled_pauli_6 = np.array([povm.get_POVM() for povm in reconstructed_pauli_6_array]) # Normalization factor 1/3^n_qubits when recombining the POVM. 
    decompiled_pauli_6 =  1/3**n_qubits* np.reshape(decompiled_pauli_6, (-1, *decompiled_pauli_6.shape[2:]))
    qst_nonadaptive.perform_BME(compute_uncertainty = compute_uncertainty,
                                override_POVM_list= [POVM(decompiled_pauli_6)])
    # qst_nonadaptive.perform_random_adaptive_BME(depolarizing_strength = depolarizing_strength,
    #                         adaptive_burnin_steps = n_qst_shots_total*2, recon_comp_list= diag_recon_comp_list,
    #                         compute_uncertainty = compute_uncertainty)
    uncertainty_container_nonadaptive.append(qst_nonadaptive.get_uncertainty())
    infidelity_container_nonadaptive.append(qst_nonadaptive.get_infidelity())
    print(f'Nonadaptive QST run {i+1}/{n_steps} took {datetime.now()-start_time}')

    # Adaptive QST
    start_time = datetime.now()
    print(f'Starting adaptive QST run {i+1}/{n_steps}')
    qst_adaptive = QST(POVM.generate_computational_POVM(n_qubits), true_states,n_qst_shots*3**n_qubits, n_qubits, False,{}, n_cores=n_cores)
    #qst_adaptive._MH_steps = 100
    qst_adaptive.n_bank = 2000
    qst_adaptive.perform_random_adaptive_BME(depolarizing_strength = depolarizing_strength,
                            adaptive_burnin_steps = adaptive_burnin, recon_comp_list = diag_recon_comp_list,
                            compute_uncertainty = compute_uncertainty)

    noise_strengths.append(depolarizing_strength)
    qst_array.append(qst_adaptive)
    infidelity_container_adaptive.append(qst_adaptive.get_infidelity())
    uncertainty_container_adaptive.append(qst_adaptive.get_uncertainty())
    print(f'Adaptive QST run {i+1}/{n_steps} took {datetime.now()-start_time}')
    # For non_adaptive QST we need to supply it with the noisy POVM separatly.


    # DT_settings={
    #     "n_qubits": n_qubits,
    #     #"calibration_states": calibration_states,
    #     "n_calibration_shots": n_qdt_shots_total//(len(calibration_states)),
    #    #"initial_POVM": POVM_list,
    #    "reconstructed_POVM_list": recon_POVM_list,
    #     "noisy_POVM_list" : noisy_POVM_list,
    #    "reconstructed_POVM_matrix":  np.array([povm.get_POVM() for povm in recon_POVM_list])
    # }
    
    settings = {
        'n_qst_shots_total': n_qst_shots_total,
        'n_qdt_shots_total': n_qdt_shots_total,
        'n_qubits': n_qubits,
        'n_averages': n_averages,
        'adaptive_burnin': adaptive_burnin,
        'noise_strengths': depolarizing_strength,
        'qdt_multipliers': qdt_multipliers,
        'true_states': true_states,
        'reconstructed_states_adaptive': qst_adaptive.get_rho_estm(),
        'reconstructed_states_nonadaptive': qst_nonadaptive.get_rho_estm(),
        'qst_bank_size': qst_adaptive.n_bank,
        'qst_nonadaptive_bank_size': qst_nonadaptive.n_bank
    }
    setting_array.append(settings)

container_dict = {
    'adaptive_infidelity_container': infidelity_container_adaptive,
    'nonadaptive_infidelity_container': infidelity_container_nonadaptive,
    'adaptive_uncertainty_container': uncertainty_container_adaptive,
    'nonadaptive_uncertainty_container': uncertainty_container_nonadaptive,
    'grid_infidelity_container': infidelity_container_grid
}

print(f'Total run time: {datetime.now()-now}')
dir_name= now_string+str(uuid.uuid4())
data_path=f'{path}/{dir_name}'
os.mkdir(data_path)

with open(f'{data_path}/infidelity_container.npy', 'wb') as f:
    np.save(f, container_dict)
with open(f'{data_path}/settings.npy', 'wb') as f:    
    np.save(f, setting_array)
print(f'Saved to {data_path}')