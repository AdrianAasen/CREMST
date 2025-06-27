import numpy as np
import scipy as sp
from EMQST_lib import support_functions as sf
from EMQST_lib import overlapping_tomography as ot
from EMQST_lib.povm import POVM
from functools import reduce


def measurement(n_shots, povm, rho, bool_exp_measurements = False, exp_dictionary = None, state_angle_representation = None, custom_measurement_function = None, return_frequencies = False):
    """
    Measurment settings and selects either experimental or simulated measurements. 
    For experimental measurements some settings are converted to angle arrays. 
    """
    if exp_dictionary is None:
        exp_dictionary = {}
        exp_dictionary["standard_measurement_function"] = None
        
        
    if bool_exp_measurements:
        if custom_measurement_function is None:
            if state_angle_representation is None:
                print("Experimental measurement: No angle representation has been given! Returning None.")
                return np.array([None]*n_shots) 
            outcome_index = exp_dictionary["standard_measurement_function"](n_shots,povm.get_angles(),state_angle_representation,exp_dictionary)
        else:
            outcome_index = custom_measurement_function(n_shots,povm.get_angles(),exp_dictionary)
    else:
        outcome_index = simulated_measurement(n_shots,povm,rho,return_frequencies)
        
    return outcome_index


def simulated_measurement(n_shots,povm,rho, return_frequencies = False):
    """
    Takes in number of shots required from a single POVM on a single quantum states.
    Returns and outcome_index vector where the index corresponds the the POVM that occured.
    """

    # Find probabilites for different outcomes
    histogram = povm.get_histogram(rho)

    # Create cumulative sum
    cumulative_sum = np.cumsum(histogram)

    # Sample outcomes 
    r = np.random.random(n_shots)

    # Return index list of outcomes 
    outcome_list = np.searchsorted(cumulative_sum, r)
    if return_frequencies:
            
        min_unique_outcomes = len(histogram)
        frequencies = outcomes_to_frequencies(outcome_list,min_unique_outcomes)
        return frequencies
    else:    
        return outcome_list

def outcomes_to_frequencies(outcomes,min_lenght):
    # Count the occurrences of each outcome
    unique_outcomes, frequencies = np.unique(outcomes, return_counts=True)


    if len(unique_outcomes) < min_lenght:
        # Pad frequencies where unique_elements_to_contain does not appear in unique_found
        missing_outcomes = np.setdiff1d(np.arange(min_lenght), unique_outcomes)
        # Insert where we have the missing outcome in the list (loop to keep )
        for outcome in missing_outcomes:
            frequencies = np.insert(frequencies, outcome, 0)
    return frequencies


def measure_separable_state(n_shots, povm_array, rho_array):
    """
    Takes in a list of single qubit states, a list of single qubit POVMs, and the number of shots.
    Returns a list of outcomes for each qubit. Each measurement is sampled independently for each qubit.
    This function is designed to be used for large system sizes.

    Parameters:
    - n_shots (int): The number of shots for each measurement.
    - povm_list (ndarray): A numpy array of single qubit POVMs.
    - rho_list (ndarray): A numpy array of single qubit states.

    Returns:
    - outcomes (ndarray): A numpy array of shape (n_shots, n_qubits) containing the outcomes for each qubit.
    """
    
    n_qubits = len(rho_array)
    outcomes_temp = np.array([simulated_measurement(n_shots, povm_array[i], rho_array[i]) for i in range(n_qubits)])
    # Change axis such that order matches what is expected from experiments.
    outcomes = np.moveaxis(outcomes_temp, 0, -1)
    return outcomes




def measure_hashed_calibration_states(n_shots, povm_array, one_qubit_calibration_states, hashed_QDT_instructions, experimental_dictionary = {"Experimental_run": False}):
    """
    Function takes in a hash function and a creates set of calibration states,
    and calls apropriate measurement function. Returns the list of measurements.
    
    Args:
        n_shots (int): The number of shots for each measurement.
        povm_array (numpy.ndarray): The array of POVMs (Positive Operator-Valued Measures).
        one_qubit_calibration_states (numpy.ndarray): The array of one-qubit calibration states.
        hashed_QDT_instructions (numpy.ndarray): The array of hashed QDT (Quantum Decision Tree) instructions.
        experimental_dictionary (dict, optional): A dictionary containing experimental settings. Defaults to {"Experimental_run": False}.
    
    Returns:
        numpy.ndarray: The array of measurement outcomes.
        
    Notes:
        - The outcomes array has shape [n_hashed_instructions, n_shots, n_qubits].
        - If experimental_dictionary["experimental_run"] is True, the measurements are performed using experimental settings.
        - If experimental_dictionary["experimental_run"] is False, the measurements are simulated.
    """
        
    if experimental_dictionary["experimental_run"]:
        possible_instruction_array = np.array([0, 1, 2, 3])
        one_qubit_calibration_angles = experimental_dictionary["one_qubit_calibration_angles"]
        comp_measurement_angles = experimental_dictionary["comp_measurement_angles"]
        hashed_state_angles = np.array([ot.instruction_equivalence(instruction, possible_instruction_array, one_qubit_calibration_angles) for instruction in hashed_QDT_instructions])
        outcomes = np.array([experimental_dictionary["standard_measurement_function"](n_shots, comp_measurement_angles, state_angles, experimental_dictionary) for state_angles in hashed_state_angles])
    else:
        # Create hashed calibration states
        hashed_calib_states = np.array([ot.calibration_states_from_instruction(instruction, one_qubit_calibration_states) for instruction in hashed_QDT_instructions])
        # Simulate measurements
        outcomes = np.array([measure_separable_state(n_shots,povm_array, rho_array) for rho_array in hashed_calib_states])
    return outcomes


def measure_hashed_POVM(n_shots, rho_array, single_qubit_pauli_6, hashed_QST_instructions, experimental_dictionary={"Experimental_run": False}):
    """
    Creates hashed POVMs and measures the array of quantum state.

    Parameters:
    - n_shots (int): The number of measurement shots to perform.
    - rho_array (numpy.ndarray): An array of quantum states to measure.
    - single_qubit_pauli_6 (numpy.ndarray): An array of single-qubit Pauli matrices.
    - hashed_QST_instructions (numpy.ndarray): An array of hashed QST instructions.
    - experimental_dictionary (dict): A dictionary containing experimental settings (default: {"Experimental_run": False}).

    Returns:
    - outcomes (numpy.ndarray): An array of measurement outcomes.
    
    Notes:
        - The outcomes array has shape [n_hashed_instructions, n_shots, n_qubits].
        - If experimental_dictionary["experimental_run"] is True, the measurements are performed using experimental settings.
        - If experimental_dictionary["experimental_run"] is False, the measurements are simulated.
    """
    possible_instruction_array = np.array(["X", "Y", "Z"])
    if experimental_dictionary["experimental_run"]:
        true_state_angles = experimental_dictionary["true_state_angles"]
        single_qubit_measurement_angles = experimental_dictionary["single_qubit_measurement_angles"]
        hashed_POVM_angles = np.array([ot.instruction_equivalence(instruction, possible_instruction_array, single_qubit_measurement_angles) for instruction in hashed_QST_instructions])
        outcomes = np.array([experimental_dictionary["standard_measurement_function"](n_shots, POVM_angles, true_state_angles, experimental_dictionary) for POVM_angles in hashed_POVM_angles])
    else:
        hashed_POVM = np.array([ot.instruction_equivalence(instruction, possible_instruction_array, single_qubit_pauli_6) for instruction in hashed_QST_instructions])
        # Measure with the hashed POVMs
        outcomes = np.array([measure_separable_state(n_shots, povm, rho_array) for povm in hashed_POVM])

    return outcomes



def measure_clusters(n_shots, povm_array, factorized_rho, cluster_size):
    """
    This function takes in a factorized density matrix and measures it using the cluster noise povm_list.
    """

    n_qubits = np.sum(cluster_size)
    n_clusters = len(cluster_size)
    full_outcomes = np.zeros((n_shots, n_qubits),dtype = int)
    for i in range(n_clusters):

        sub_rho = factorized_rho[sum(cluster_size[:i]):sum(cluster_size[:i+1])]
        
        # tensor together rho
        rho = reduce(np.kron, sub_rho)
        outcome = simulated_measurement(n_shots, povm_array[i], rho)

        # Add outcomes to the full_outcomes array in binary form
        full_outcomes[:,sum(cluster_size[:i]):sum(cluster_size[:i+1])] = sf.decimal_to_binary_array(outcome, cluster_size[i])

        # Concatinate all outcomes into a single array

    return full_outcomes 



def measure_cluster_QST(n_QST_shots, povm_array, rho_true_array, hashed_QST_instructions,cluster_size):
    """
    Because we have genuine clusted POVMs, we need to apply the rotations to the qubits rather than the POVMs for the meaurements.
    Luckily the instructions are are single qubit rotations, so we can simply create a copy of the factorized rhos and apply the appropriete unitary in accordance with the hashed_QST_instructions.
    """
    
    possible_instructions = np.array(["X", "Y", "Z"])
    sigma_x = np.array([[0,1], [1,0]])
    sigma_y = np.array([[0,-1j], [1j,0]])
    # NOTE: measuring in the x-basis is eqivalent to rotate the x eigenstte to become a z state, which requires a rotation of -pi/2 around the y-axis.
    # NOTE: that this is the inverse rotation as we used in for the rotations applied to the POVMs in def generate_Pauli_from_comp in the povm class.
    rot_x_to_z = sp.linalg.expm(-1j * (-np.pi/4) * sigma_y)
    rot_y_to_z = sp.linalg.expm(-1j * (np.pi/4) * sigma_x)
    
    # Create list of single qubit rotations from comp to Pauli
    rotation_matrices = np.array([rot_x_to_z, rot_y_to_z, np.eye(2)]) 

    #conjugate_rotation_matrices = np.array([rot_x_to_z.conj().T, rot_y_to_z.conj().T, np.eye(2)]) 
    hashed_unitaries = np.array([ot.instruction_equivalence(hashed_QST_instruction, possible_instructions , rotation_matrices) for hashed_QST_instruction in hashed_QST_instructions])
    #hashed_conjugate_unitaries = ot.instruction_equivalence(hashed_QST_instructions, possible_instructions , conjugate_rotation_matrices)
    hashed_factorized_rhos = np.einsum('nmij,mjk,nmlk->nmil', hashed_unitaries, rho_true_array,hashed_unitaries.conj()) # Note that the second hashed_unitaries are swapped to perform a transpose. 
    outcomes = np.array([measure_clusters(n_QST_shots, povm_array, rho_array, cluster_size) for rho_array in hashed_factorized_rhos])
    return outcomes 



def measure_hashed_chunk_QST(n_shots, chunk_size, povm_array, povm_size_array, state_array, state_size_array, hashed_QST_instructions):
    """
    Measures sets of qubits as genuine chunk-size states and POVMs.
    Assumes that the states and POVMs are already split into chunks of spesificed size. This function is only for QST, as calibration states should always be factorized.
    n_shots: number of shots used for each possible computational basis measurement, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ etc.
    """
    n_qubits = np.sum(state_size_array)
    n_hashes = len(hashed_QST_instructions)
    full_outcomes = np.zeros((n_hashes,n_shots, n_qubits) ,dtype = int)
    # Each chunk constitutes chunck_size qubits. We assume that the states and POVMs are already split into chunks of spesificed size.
    # The stratergy is to create chunks of size chunk_size, and then create a geneuine state and POVM on that chunk, and measure it. 
    # Find index to partition the POVM array and state array
    povm_index_array = ot.create_chunk_index_array(povm_size_array,chunk_size)
    state_index_array = ot.create_chunk_index_array(state_size_array,chunk_size)
    unitary_index_array = ot.create_chunk_index_array(np.ones((n_qubits)),chunk_size)
 
    # Create hash instructions unitaries. The unitaries will be applied to the states. 
    possible_instructions = np.array(["X", "Y", "Z"])
    sigma_x = np.array([[0,1], [1,0]])
    sigma_y = np.array([[0,-1j], [1j,0]])
    # NOTE: measuring in the x-basis is eqivalent to rotate the x eigenstte to become a z state, which requires a rotation of -pi/2 around the y-axis.
    # NOTE: that this is the inverse rotation as we used in for the rotations applied to the POVMs in def generate_Pauli_from_comp in the povm class.
    rot_x_to_z = sp.linalg.expm(-1j * (-np.pi/4) * sigma_y)
    rot_y_to_z = sp.linalg.expm(-1j * (np.pi/4) * sigma_x)
    
    # Create list of single qubit rotations from comp to Pauli
    rotation_matrices = np.array([rot_x_to_z, rot_y_to_z, np.eye(2)]) 

    #conjugate_rotation_matrices = np.array([rot_x_to_z.conj().T, rot_y_to_z.conj().T, np.eye(2)]) 
    
    hashed_unitaries = np.array([ot.instruction_equivalence(hashed_QST_instruction, possible_instructions , rotation_matrices) for hashed_QST_instruction in hashed_QST_instructions])
    # These unitaries will have the shape: # (unique measurements x n_qubits)
    # print(f'Hashed unitaries: {hashed_unitaries.shape}')
    # print(f'State index array: {state_index_array}' )
    for i in range(len(povm_index_array)-1): # Loop over chunks
        # Tensor together objects for current chunk
        tensored_unitaries = np.array([reduce(np.kron,hashed_unitaries[unitary_index_array[i]:unitary_index_array[i+1]]) for hashed_unitaries in hashed_unitaries])
        # for unitary in tensored_unitaries:
        #     print(unitary.shape)
        if state_index_array[i+1]-state_index_array[i] == 1: # If only single entry in chunk no reduction call is needed.
            tensored_chunk_rho = state_array[state_index_array[i]]
        else:
            tensored_chunk_rho = reduce(np.kron,state_array[state_index_array[i]:state_index_array[i+1]])
        if povm_index_array[i+1]-povm_index_array[i] == 1: # If only single entry in chunk no reduction call is needed.
            sub_povm = povm_array[povm_index_array[i]]
        else:
            sub_povm = reduce(POVM.tensor_POVM, povm_array[povm_index_array[i]:povm_index_array[i+1]])[0]
        
        # Einsum is split into two operations as it is too slow for larger chunk sizes. 
        # Note that the second hashed_unitaries are swapped to perform a transpose.
        rotated_rhos = np.einsum('nij,jk,nlk->nil', tensored_unitaries, tensored_chunk_rho,tensored_unitaries.conj(), optimize=True) 
        

        # Chunk measurements
        outcomes = np.array([simulated_measurement(n_shots, sub_povm, rho) for rho in rotated_rhos])
     
        # Add outcomes to the full_outcomes array in binary form
        full_outcomes[:,:,chunk_size*i:chunk_size*(i+1)] = np.array([sf.decimal_to_binary_array(hashed_outcome, chunk_size) for hashed_outcome in outcomes])
    return full_outcomes


def measure_and_QST_target_qubit_only(two_point_array,noise_cluster_labels,n_QST_shots, n_qubits, chunk_size, povm_array, cluster_size, rho_true_array, state_size_array,clustered_QDOT):
    """
    QST method that both measures and reconstructs the state for the spesific correlators given in two_point_array.
    This function does not check if it has previosly measured correlatros that span the same noise structure/qubits. 
    """
    selected_cluster_index =  ot.get_cluster_index_from_correlator_labels(noise_cluster_labels, two_point_array) 
    target_qubits = []
    for index in selected_cluster_index:
        target_qubits = np.append(target_qubits, noise_cluster_labels[index])
    target_qubits = np.array(target_qubits.astype(int), dtype=int)
    target_qubits = np.sort(target_qubits)[::-1]
    QST_only_instructions = ot.create_QST_instructions(n_qubits, target_qubits)

    QST_outcomes_reduced_instructions = measure_hashed_chunk_QST(n_QST_shots, chunk_size, povm_array, cluster_size, rho_true_array, state_size_array, QST_only_instructions)

    print(f'Target Qubits: {target_qubits}')
    rho_recon_1 = ot.QST_from_instructions(QST_outcomes_reduced_instructions, QST_only_instructions,two_point_array, target_qubits, clustered_QDOT, noise_cluster_labels)
    return rho_recon_1, target_qubits


def correlator_spesific_QST_measurements(noise_cluster_labels, two_point_corr_label,
                                       n_averages, n_qubits, method,
                                       n_total_QST_shots,  chunk_size, povm_array, cluster_size, rho_true_array, state_size_array):
    """
    Function performs comparativ QST measurements where each method recieves the same measurements outcomes as correlated QREM. 
    
    The protocol is as follows:
    - Performs the nessecary QST measurements to reconstruct the connected noise clusters.
    - Computes the readout error mitigated states for the connected cluster, the two-point correlators and the one-qubit POVMs.
    - Assumes it takes in a single two-point correlator label, to make function parallelizable.
    Args
    method: list of integers that selects which methods to compare to correlated QREM.
    0: no QREM
    1: factorized QREM
    2: two RDM QREM
    3: Classical correlated QREM
    """
    if method is None:
        method = [0]
        print(f'No method selected. Comparsion defaults to no QREM.')
        
    # Find the relevant cluster qubits.
    selected_cluster_index = ot.get_cluster_index_from_correlator_labels(noise_cluster_labels, two_point_corr_label) 
    target_qubits = []
    for index in selected_cluster_index:
        target_qubits = np.append(target_qubits, noise_cluster_labels[index])
    target_qubits = np.array(target_qubits.astype(int), dtype=int)
    target_qubits = np.sort(target_qubits)[::-1]
    
    # The spesific number of QST shots needs to be modified such that it adhers to the total number of shots.
    # To do this we devide the total number of shots by the number of basis that will be measured
    n_shots_pr_basis = int(n_total_QST_shots/3**len(target_qubits) +1) # +1 to make sure that each group gets at least total number of shots. Some groups will have slightly more. 
    #print(f'Qubit labels to be reconstructed: {target_qubits}.')
    QST_only_instructions = ot.create_QST_instructions(n_qubits, target_qubits)
    #print(f'QST instructions: {QST_only_instructions}.')
    QST_outcomes_reduced_instructions = [measure_hashed_chunk_QST(n_shots_pr_basis, chunk_size, povm_array, cluster_size, rho_true_array[i], state_size_array, QST_only_instructions) for i in range(n_averages)]
    #print(len(QST_outcomes_reduced_instructions))
    return QST_outcomes_reduced_instructions, target_qubits, QST_only_instructions