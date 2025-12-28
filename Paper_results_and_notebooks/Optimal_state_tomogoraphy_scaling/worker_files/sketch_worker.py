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

n_cores = 6
# Updates passed core count if availabe
if len(sys.argv) > 1:
    new_n_cores = int(sys.argv[1])
    if new_n_cores > n_cores:
        n_cores = new_n_cores
        print(f'Updated core count to {new_n_cores}.')


path = "adaptive_results/nonadaptive_sketch"
now=datetime.now()
now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
dir_name= now_string+str(uuid.uuid4())
data_path=f'{path}/{dir_name}'
os.mkdir(data_path)


n_shots_total = 10**4
n_qubits = 1
n_shots = n_shots_total//3**n_qubits # In the qst code it is assumed that each single qubit measurement is a Pauli-basis, hence 3^n_qubits total measurement settings.
n_averages = 200

print(f'Starting nonadaptive QST with {n_shots_total} shots, {n_qubits} qubits, {n_averages} averages.')

true_states = np.array([sf.generate_random_pure_state(n_qubits) for _ in range(n_averages)])
povm = POVM.generate_Pauli_POVM(n_qubits)
decompiled_array = np.array([povm[i].get_POVM() for i in range(len(povm))])
pauli_6_array = 1/3**(n_qubits)*np.array(decompiled_array.reshape(-1,decompiled_array.shape[-2],decompiled_array.shape[-1]))
decompiled_pauli_POVM = POVM(pauli_6_array)
qst_adaptive = QST(povm, true_states, n_shots, n_qubits, False,{}, n_cores=n_cores)



noise_levels = [0,0.05,0.15,0.25]
n_steps = len(noise_levels)
infidelity_container_nonadaptive = []
infidelity_container_unmitigated = []
noise_strengths = []
setting_array = []

start_time = datetime.now()
print(f'Total shots for the whole run: {n_shots_total * n_averages*n_steps}')
print(f'Starting adaptive QST with {n_shots_total} shots, {n_qubits} qubits, \n{n_steps} of noise runs, with {n_averages} averages each, and {n_shots} shots per measurement setting for adaptive QST.')
print(f'Start time: {now_string}.')


for i in range(n_steps):
    # Adaptive QST
    depolarizing_strength = noise_levels[i]
    noise_strengths.append(depolarizing_strength)
    print(f'Starting nonadaptive QST run {i+1}/{n_steps}')
    start_time = datetime.now()
    depolarized_pauli_6_array = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in pauli_6_array])
    noisy_povm = POVM(depolarized_pauli_6_array)
    # # Note that we use n_shots*3**n_qubits here, since nonadaptive QST does not split the shots between different measurement settings. 
    # # We also multiply rather than use n_shots total since there could be rounding difference between the adaptive and non-daptive QST. 
    qst_nonadaptive = QST([noisy_povm], true_states, n_shots*3**n_qubits, n_qubits, False,{}, n_cores=n_cores) # They are initalized the same
    qst_nonadaptive.generate_data()
    qst_nonadaptive.perform_BME()
    non_adaptive_mitigated_state = qst_nonadaptive.get_rho_estm()
    infidelity_container_nonadaptive.append(qst_nonadaptive.get_infidelity())
    
    qst_nonadaptive.perform_BME(override_POVM_list=[decompiled_pauli_POVM])
    infidelity_container_unmitigated.append(qst_nonadaptive.get_infidelity())
    print(f'Nonadaptive QST run {i+1}/{n_steps} took {datetime.now()-start_time}')
    
    settings = {
        'n_shots': n_shots,
        'n_qubits': n_qubits,
        'n_averages': n_averages,
        'noise_strengths': noise_strengths,
        'true_states': true_states,
        'reconstructed_states_unmitigated': qst_nonadaptive.get_rho_estm(),
        'reconstructed_states_nonadaptive': non_adaptive_mitigated_state,
    }
    setting_array.append(settings)

container_dict = {
    'infidelity_container_unmitigated': infidelity_container_unmitigated,
    'nonadaptive_infidelity_container': infidelity_container_nonadaptive
}



with open(f'{data_path}/infidelity_container.npy', 'wb') as f:
    np.save(f, container_dict)
with open(f'{data_path}/settings.npy', 'wb') as f:    
    np.save(f, setting_array)
print(f'Saved to {data_path}')