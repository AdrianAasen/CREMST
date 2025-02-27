import numpy as np
from joblib import Parallel, delayed, parallel_config, wrap_non_picklable_objects
from joblib import Parallel, delayed
from functools import reduce
from itertools import product, chain, combinations
from EMQST_lib import support_functions as sf
from EMQST_lib.povm import POVM, generate_pauli_6_rotation_matrice
from EMQST_lib import dt


def trace_out(qubit_to_keep_labels, qubit_array):
    """
    Trace out qubits from an array.
    
    Args:
        qubit_to_keep_labels (ndarray): List of labels of qubits to keep. The order does not matter.
        qubit_array (ndarray): Array that contains qubit information. The array is provided in a shape where the last axis iterates over the qubits. 
        
    Returns:
        ndarray: The traced out array.
    """
    # Sort labels such that input order does not matter.
    # Note the last conversion as we order our qubits in reverse order, [3,2,1,0]
    qubit_to_keep_labels = np.sort(qubit_to_keep_labels)[::-1]
    qubit_to_keep_index = qubit_label_to_list_index(qubit_to_keep_labels, qubit_array.shape[-1])
    traced_down_outcomes = qubit_array[..., qubit_to_keep_index]
    return traced_down_outcomes


def create_hashed_instructions(hash_family, possible_instruction_array, n_hash_symbols):
    """
    Creates all instructions according to hash in a minimal way. 
    """
    n_qubits_total = len(hash_family[0])
    unique_instructions = np.array([hash_to_instruction(function, possible_instruction_array, n_hash_symbols) for function in hash_family]).reshape(-1, n_qubits_total)
    base_instructions = np.array([[instruction]*n_qubits_total for instruction in possible_instruction_array])
    combined_instructions = np.vstack((unique_instructions, base_instructions))
    
    return combined_instructions #unique_instructions, base_instructions


def calibration_states_from_instruction(instruction, one_qubit_calibration_states, return_tesored=False):
    """
    Creates calibration states for a set of instructions.
    
    This function takes a set of instructions and generates calibration states based on those instructions.
    
    Args:
        instructions (numpy.ndarray): A list of instructions. Needs to be on form [0, 1, 2, 3]. 
        one_qubit_calibration_states (numpy.ndarray): A list of one-qubit calibration states.
        return_tesored (bool, optional): Whether to return the calibration states tensored together. 
                                          Defaults to False.
                                          Note:
                                          This function scales exponentially. For n qubits, the number of calibration states is 4^n.
    
    Returns:
        numpy.ndarray: An array of calibration states.
    """
    
    possible_instructions = np.array([0, 1, 2, 3])
    calib_states = instruction_equivalence(instruction, possible_instructions, one_qubit_calibration_states)
    if return_tesored:  # Tensor together for convenience
        calib_states = reduce(np.kron, calib_states)
    
    return calib_states


def qubit_label_to_list_index(qubit_label, n_total_qubits):
    """
    Function that takes in a hash function and returns the corresponding index of the qubit in a list.

    Parameters:
    - qubit_label: A list of qubit labels (e.g. a hash function)
    - n_total_qubits: The total number of qubits (e.g. the number of hash symbols)

    Returns:
    - The corresponding index of the qubit in a list.

    Example:
    >>> qubit_label_to_list_index(3, 5)
    1
    """
    return (n_total_qubits - 1) - qubit_label


def hash_to_instruction(hash_function, instruction_list, n_hash_symbols):
    """
    Function that takes in a hash function and returns the corresponding instruction list.

    Parameters:
    - hash_function: The hash function used to determine the instruction list.
    - instruction_list: The list of instructions.
    - n_hash_symbols: The number of hash symbols.

    Returns:
    - The corresponding instruction list based on the hash function.

    The instruction list contains operations that are supposed to be distributed such that they are all present for each qubit combination according to the hash.
    This function does also remove the cases where all outputs are the same, to avoid creating duplicate instructions in each hash function.
    """
    # Translate hash qubit labels to list index
    qubit_index = qubit_label_to_list_index(hash_function, n_hash_symbols)
    # Remove the cases where all outputs are the same, to avoid creating duplicate instructions in each hash function.
    pruned_comb_list = create_unique_combinations(instruction_list, n_hash_symbols)
    
    return pruned_comb_list[:, qubit_index] 



def get_traced_out_indicies(index_list,n_qubits_total):
    """
    Takes in a list of qubits of intrest, and returns the list of all qubit indices to trace over.
    """
    # Creates a full list of all qubits
    full_list = np.arange(n_qubits_total)
    # Checks which qubits qubits are not present in original list
    traced_indices = np.setdiff1d(full_list,index_list)
    #print(traced_indices)
    return np.sort(traced_indices)



def instruction_equivalence(instruction, possible_instructions, instruction_equivalence):
    """
    Translate instructions to the corresponding instruction in the ordered_instruction_equivalence list. 
    Primary use it to create Hilbert Space objects from simple instructions that can be stored. 

    Parameters:
    - instruction: The instruction to be translated.
    - instruction_equivalence (list): A list of instructions in the desired order.
    - possible_instructions (list): A list of all possible accepted instructions in desired order. .

    Returns:
    - list: The translated instruction.
    
    NOTE: This function is not optimized for speed, if keys and value are the same, e.g. {1:2, 2:3}, you get a bug where both 1 and 2 is set to 3. 

    """
    # Create instruction dictionary
    instruction_dict = dict(zip(possible_instructions, instruction_equivalence))

    new_instruction = np.array([instruction_dict[element] for element in instruction])
    #print(new_instruction)
    return new_instruction



def create_unique_combinations(elements, n_repeat, remove_duplicates=True):
    """
    Generate unique combinations of elements with repetition. Where entries with all equal elements are removed.

    Args:
        elements (iterable): The elements to be combined.
        n_repeat (int): The number of times each element can be repeated in a combination.

    Returns:
        numpy.ndarray: An array of unique combinations.
    """
    comb_list = np.array(list(product(elements, repeat=n_repeat)))
    if remove_duplicates:
        # Remove the duplicate elements
        n_unique_instructions = len(elements)
        indicies = np.linspace(0, len(comb_list)-1, n_unique_instructions, dtype=int)
        mask_array = np.ones(len(comb_list), dtype=bool)
        mask_array[indicies] = False
        comb_list = comb_list[mask_array]
    return comb_list


def get_index_counts_from_decimal_outcomes(decimal_outcomes,n_subsysten_qubits):
    return np.array([np.bincount(outcomes,minlength=2**n_subsysten_qubits) for outcomes in decimal_outcomes])



def create_traced_out_calibration_states(subsystem_labels, hash_family, one_qubit_calibration_states, n_hash_symbols, n_qubits_total):
    """
    Takes in index of the qubits of intrest and the full set of measurements on the whole system and
    traces it down to the outcomes on the system of interest, and provies the set of calibraton states on the relevant system. 
    """
  
    # Sort labels in descending order  
    subsystem_labels = np.sort(subsystem_labels)[::-1]
    n_subsystem_qubits = len(subsystem_labels)

    # Instructions here indicate the number of calibration states, by default 4.
    instructin_list = np.array([0, 1, 2, 3])
    # Get qubit index of the subsystem in the total qubit system
    subsystem_qubit_index = qubit_label_to_list_index(subsystem_labels,n_qubits_total)
    # Create the instructions for the hashed subsystem (NOTE we slice out only the subsystem qubits from the hash family)
    hashed_subsystem_instructions = np.array([hash_to_instruction(function, instructin_list, n_hash_symbols) for function in hash_family[:,subsystem_qubit_index]]).reshape(-1, n_subsystem_qubits)
    #print(hashed_subsystem_instructions.shape)
    # Base instructions are the same for all subsystems.
    base_subsystem_instructions = np.array([[0]*n_subsystem_qubits,[1]*n_subsystem_qubits,[2]*n_subsystem_qubits,[3]*n_subsystem_qubits])
    combined_hashed_subsystem_instructions = np.vstack((hashed_subsystem_instructions,base_subsystem_instructions))
    # The calibration states are tensored together, to be at most 4 qubit operators, according to the traced down hash. 
    traced_out_calib_states = np.array([calibration_states_from_instruction(instruction,one_qubit_calibration_states,True) for instruction in combined_hashed_subsystem_instructions])
    #traced_out_base_states = np.array([ot.calibration_states_from_instruction(instruction,one_qubit_calibration_states,True) for instruction in base_subsystem_instructions])
    #print(traced_out_calib_states.shape)
    # Reshape and combine the calibration sates with the base states into one large set of measurements. 
    #traced_out_calib_states = traced_out_calib_states.reshape(-1,*traced_out_calib_states.shape[-2:])
    
    # Combine the outcomes and calibration states into one large array.
    #combined_calib_states = np.concatenate((traced_out_calib_states.reshape(-1,*traced_out_calib_states.shape[-2:]),traced_out_base_states), axis = 0)

    return traced_out_calib_states 




def create_traced_out_reconstructed_POVM(subsystem_labels, reconstructed_comp_POVM, hash_family, n_hash_symbols, n_qubits_total):
    # Create a fully Pauli POVM from reconstructed computational basis POVM
    subsystem_labels = np.sort(subsystem_labels)[::-1]
    reconstructed_Pauli_POVM = POVM.generate_Pauli_from_comp(reconstructed_comp_POVM)

    n_subsystem_qubits = len(subsystem_labels)
    # ii) Create the POVM list that assosicated to each row in the downconverted frequency list
    # Get qubit index of the subsystem in the total qubit system
    subsystem_qubit_index = qubit_label_to_list_index(subsystem_labels,n_qubits_total)
    possible_instructions = np.array([0, 1, 2])
    #options_check = np.array(["X","Y", "Z"])
    # Create the instructions for the hashed subsystem (NOTE we slice out only the subsystem qubits from the hash family)
    hashed_subsystem_instructions = np.array([hash_to_instruction(function, possible_instructions, n_hash_symbols) for function in hash_family[:,subsystem_qubit_index]]).reshape(-1, n_subsystem_qubits)
    base_instructions = np.array([[0]*n_subsystem_qubits,[1]*n_subsystem_qubits,[2]*n_subsystem_qubits])
    #print(hashed_subsystem_instructions.shape)
    combined_hash_instructions = np.vstack((hashed_subsystem_instructions, base_instructions))
    combined_povm_array = subsystem_instructions_to_POVM(combined_hash_instructions, reconstructed_Pauli_POVM, n_subsystem_qubits)
    return combined_povm_array


def create_traced_out_POVM_rotation(subsystem_labels, hash_family, n_hash_symbols, n_qubits_total):
    # Create a fully Pauli POVM from reconstructed computational basis POVM
    subsystem_labels = np.sort(subsystem_labels)[::-1]
    n_subsystem_qubits = len(subsystem_labels)
    povm_rotators = generate_pauli_6_rotation_matrice(n_subsystem_qubits)

    
    # ii) Create the POVM list that assosicated to each row in the downconverted frequency list
    # Get qubit index of the subsystem in the total qubit system
    subsystem_qubit_index = qubit_label_to_list_index(subsystem_labels,n_qubits_total)
    possible_instructions = np.array([0, 1, 2])
    #options_check = np.array(["X","Y", "Z"])
    # Create the instructions for the hashed subsystem (NOTE we slice out only the subsystem qubits from the hash family)
    hashed_subsystem_instructions = np.array([hash_to_instruction(function, possible_instructions, n_hash_symbols) for function in hash_family[:,subsystem_qubit_index]]).reshape(-1, n_subsystem_qubits)
    base_instructions = np.array([[0]*n_subsystem_qubits,[1]*n_subsystem_qubits,[2]*n_subsystem_qubits])
    #print(hashed_subsystem_instructions.shape)
    combined_hash_instructions = np.vstack((hashed_subsystem_instructions, base_instructions))
    combined_rotation_array = subsystem_instructions_to_POVM(combined_hash_instructions, povm_rotators, n_subsystem_qubits)
    return combined_rotation_array


def subsystem_instructions_to_POVM(instructions, reconstructed_Pauli_POVM, n_subsystem_qubits ):
    """
    Takes in an instruction and turns it into a Pauli POVM element.
    E.g. [0,1] -> [X,Y] reconstructed POVM element
    """

    base_3 = np.array([3**i for i in range(n_subsystem_qubits)])[::-1]
    
    povm_array = np.array([reconstructed_Pauli_POVM[np.dot(instruction,base_3)] for instruction in instructions])
    return povm_array


@wrap_non_picklable_objects
def get_traced_out_index_counts(outcomes, subsystem_label):
    """
    Takes in outcomes and subsystem labels and returns the index counts for the subsystem.
    The order of the input subsystem labels does not matter.
    The index counts are returned in the decending order of the subsystem labels. E.g subsystem label 0 is always the last entry in the returned array.
    """
    n_subsystem_qubits = len(subsystem_label)
    traced_out_outcomes = trace_out(subsystem_label,outcomes)
    decimal_outcomes = sf.binary_to_decimal_array(traced_out_outcomes)
    index_counts = get_index_counts_from_decimal_outcomes(decimal_outcomes,n_subsystem_qubits)
    return index_counts


def OT_MLE(hashed_subsystem_reconstructed_Pauli_6, index_counts):
    """
    Performs Overlapping Tomography Maximum Likelihood Estimation (OT-MLE) on a given set of hashed subsystems.

    Parameters:
    hashed_subsystem_reconstructed_Pauli_6 (ndarray): A list of measurementd from a traced down subsystem. 
    index_counts (ndarray): An array containing the index counts.

    Returns:
    ndarray: The estimated density matrix of the system.

    """

    full_operator_list = np.array([a.get_POVM() for a in hashed_subsystem_reconstructed_Pauli_6])
    dim = full_operator_list.shape[-1]
    OP_list = full_operator_list.reshape(-1,dim,dim)
    index_counts = index_counts.reshape(-1)

    iter_max = 1000
    dist     = float(1)

    rho_1 = np.eye(dim)/dim
    rho_2 = np.eye(dim)/dim
    j = 0
    optimize_path_p = np.einsum_path('ik,nki->n', rho_1, OP_list)[0]
    optimize_path_R = np.einsum_path('n,n,nij->ij', index_counts, index_counts, OP_list)[0]
    
    while j<iter_max and dist>1e-10:
        p      = np.einsum('ik,nki->n', rho_1, OP_list, optimize=optimize_path_p)
        R      = np.einsum('n,n,nij->ij', index_counts, 1/p, OP_list, optimize=optimize_path_R)
        # p      = np.einsum('ik,nki->n', rho_1, OP_list)
        # R      = np.einsum('n,n,nij->ij', index_counts, 1/p, OP_list)
        update = R@rho_1@R
        rho_1  = update/np.trace(update)

        if j>=40 and j%100==0:
            dist  = sf.qubit_infidelity(rho_1, rho_2)
        rho_2 = rho_1
        j += 1
        
        
    return rho_1


def OT_MLE_efficient(comp_basis_POVM, hashed_subsystem_Pauli_6_rotators, index_counts):
    """
    Performs Overlapping Tomography Maximum Likelihood Estimation (OT-MLE) on a given set of hashed subsystems.

    Parameters:
    hashed_subsystem_reconstructed_Pauli_6 (ndarray): A list of measurementd from a traced down subsystem. 
    index_counts (ndarray): An array containing the index counts.

    Returns:
    ndarray: The estimated density matrix of the system.

    """

    print(hashed_subsystem_Pauli_6_rotators.shape)
    comp_basis_operator_list = comp_basis_POVM.get_POVM()
    print(comp_basis_operator_list.shape)
    print(index_counts.shape)

    #full_operator_list = np.array([a.get_POVM() for a in hashed_subsystem_reconstructed_Pauli_6])
    dim = comp_basis_operator_list.shape[-1]
    #OP_list = full_operator_list.reshape(-1,dsim,dim)
    # rotated operatros should be rotators comp rotators
    #index_counts = index_counts.reshape(-1)

    iter_max = 1000
    dist     = float(1)

    rho_1 = np.eye(dim)/dim
    rho_2 = np.eye(dim)/dim
    j = 0
    hashed_subsystem_Pauli_6_rotators_conj = np.transpose(hashed_subsystem_Pauli_6_rotators, axes=[0,2,1]).conj()
    optimize_path_p = np.einsum_path('ik,nkl,mlo,noi->nm', rho_1, hashed_subsystem_Pauli_6_rotators, comp_basis_operator_list,hashed_subsystem_Pauli_6_rotators_conj, optimize="optimal")[0]
    optimize_path_R = np.einsum_path('nm,nm,nkl,mlo,noi->ki', index_counts, index_counts, hashed_subsystem_Pauli_6_rotators, comp_basis_operator_list,hashed_subsystem_Pauli_6_rotators_conj, optimize="optimal")[0]
    
    while j<iter_max and dist>1e-10:
        # new_mesh = np.einsum('nij, mjk, nkl->nmil', tensored_rot, comp_list, np.transpose(tensored_rot, axes=[0,2,1]).conj()) 
        p = np.einsum('ik,nkl,mlo,noi->nm', rho_1, hashed_subsystem_Pauli_6_rotators, comp_basis_operator_list,hashed_subsystem_Pauli_6_rotators_conj, optimize=optimize_path_p)
        R = np.einsum('nm,nm,nkl,mlo,noi->ki', index_counts, 1/p, hashed_subsystem_Pauli_6_rotators, comp_basis_operator_list,hashed_subsystem_Pauli_6_rotators_conj, optimize=optimize_path_R)
        #p = np.einsum('ik,nkl,mlo,noi->nm', rho_1, hashed_subsystem_Pauli_6_rotators, comp_basis_operator_list,hashed_subsystem_Pauli_6_rotators_conj, optimize=True)
        #R = np.einsum('nm,nm,nkl,mlo,noi->ki', index_counts, 1/p, hashed_subsystem_Pauli_6_rotators, comp_basis_operator_list,hashed_subsystem_Pauli_6_rotators_conj, optimize=True)
        
        update = R@rho_1@R
        rho_1  = update/np.trace(update)

        if j>=40 and j%100==0:
            dist  = sf.qubit_infidelity(rho_1, rho_2)
        rho_2 = rho_1
        j += 1
    return rho_1

def QST(subsystem_label, QST_index_counts, hash_family, n_hash_symbols, n_qubits_total, reconstructed_comp_POVM):
    """
    Performs Quantum overlapping State Tomography (OT) on a subsystem.

    Args:
        subsystem_label (ndarray): The label of the subsystem.
        QST_index_counts (ndarray): A array containing the counts of measurement outcomes for each measurement index.
        hash_family (ndarray): The hash family used for creating the traced-out reconstructed POVM.
        n_hash_symbols (int): The number of hash symbols used for creating the traced-out reconstructed POVM.
        n_qubits (int): The number of qubits in the system.
        reconstructed_comp_POVM (ndarray): A array of reconstructed computational basis POVMs.

    Returns:
        rho_recon (numpy.ndarray): The reconstructed density matrix of the subsystem.
    """
    n_local_qubits = len(subsystem_label)
    
    if n_local_qubits<6: # Run more efficient version if the number of qubits is less than 6. 
        # Create a new system that does not require the the full Pauli-6 to be reconstructed, but rather we track only the hashed rotation matrices. 
        hashed_subsystem_reconstructed_Pauli_6 = create_traced_out_reconstructed_POVM(subsystem_label, reconstructed_comp_POVM, hash_family, n_hash_symbols, n_qubits_total)
        rho_recon = OT_MLE(hashed_subsystem_reconstructed_Pauli_6, QST_index_counts)
    else: # Runs a memory efficient version if qubit number is larger than 6.
        # The efficiency comes from more efficient memory usage, as the full Pauli-6 is not reconstructed. Only rotation matrices are stored.
        print(f'Running memory efficient version of QST.')
        rho_recon = QST_memory_efficient(subsystem_label, QST_index_counts, hash_family, n_hash_symbols, n_qubits_total, reconstructed_comp_POVM)
    return rho_recon

def QST_memory_efficient(subsystem_label, QST_index_counts, hash_family, n_hash_symbols, n_qubits, reconstructed_comp_POVM):
    """
    Runs a memory efficient version of QST, suitable for qubit systems larger than 6, slightly slower for qubit numbers between 2 and 5.

    Args:
        subsystem_label (ndarray): The label of the subsystem.
        QST_index_counts (ndarray): A array containing the counts of measurement outcomes for each measurement index.
        hash_family (ndarray): The hash family used for creating the traced-out reconstructed POVM.
        n_hash_symbols (int): The number of hash symbols used for creating the traced-out reconstructed POVM.
        n_qubits (int): The number of qubits in the system.
        reconstructed_comp_POVM (ndarray): A array of reconstructed computational basis POVMs.

    Returns:
        rho_recon (numpy.ndarray): The reconstructed density matrix of the subsystem.
    """
    
    hashed_subsystem_pauli_6_rotators = create_traced_out_POVM_rotation(subsystem_label, hash_family, n_hash_symbols, n_qubits)

    rho_recon = OT_MLE_efficient(reconstructed_comp_POVM,hashed_subsystem_pauli_6_rotators, QST_index_counts)
    return rho_recon

    
    

def QDT(subsystem_label, QDT_index_counts, hash_family, n_hash_symbols, n_qubits, one_qubit_calibration_states):
    """
    Performs Quantum Overlapping Detector Tomography (QDT) on a subsystem.

    Args:
        subsystem_label (ndarray): The label of the subsystem
        QDT_index_counts (ndarray): A array containing the counts of measurement outcomes for each measurement index.
        hash_family (ndarray): The hash family used for creating traced out calibration states.
        n_hash_symbols (int): The number of hash symbols.
        n_qubits (int): The number of qubits in the subsystem.
        one_qubit_calibration_states (list): A list of one-qubit calibration states.

    Returns:
        reconstructed_comp_POVM (POVM): The reconstructed computational POVM.

    """
    n_subsystem_qubits = len(subsystem_label)
    hashed_subsystem_calibration_states = create_traced_out_calibration_states(subsystem_label, hash_family, one_qubit_calibration_states, n_hash_symbols, n_qubits)
    guess_POVM = POVM.generate_computational_POVM(n_subsystem_qubits)[0]
    reconstructed_comp_POVM = dt.POVM_MLE(n_subsystem_qubits, QDT_index_counts, hashed_subsystem_calibration_states, guess_POVM)
    return reconstructed_comp_POVM


def create_2RDM_hash(n_total_qubits):
    """
    Create a hash family for the 2-RDM (two-particle reduced density matrix) using the Wilczek hashing function.
    
    Parameters:
        n_total_qubits (int): The total number of qubits.
    
    Returns:
        hash_family (ndarray): The hash family for the 2-RDM.
    
    References:
        - Cotler, J.,  Wilczek, F. (2020). Quantum Overlapping Tomography. Physical Review Letters, 124(10), 100401.
          https://link.aps.org/doi/10.1103/PhysRevLett.124.100401
    """
    # Create array of qubit labels
    qubit_array = np.arange(n_total_qubits)
    # Find the max lenght of binary string
    binary_length = int(np.ceil(np.log2(n_total_qubits)))
    # Create binary array for each interger
    binary_array = (((qubit_array[:, None] & (1 << np.arange(binary_length)))) > 0).astype(int)
    # Transpose binary array to create the hash family. 
    hash_family = np.transpose(binary_array)
    return hash_family
    


def check_qubit_pairs(subsystem_labels,n_total_qubits):
    """
    Check and adjust qubit pairs if they are the same.
    If they are equal the second qubit label is adjusted to be the next qubit label.

    Args:
        subsystem_labels (ndarray): A list of qubit pairs.

    Returns:
        ndarray: The number of qubits in the subsystem.

    """
    subsystem_labels = subsystem_labels% n_total_qubits # Convert them to be in the range of the total number of qubits
    for pair in subsystem_labels:
        if pair[0] == pair[1]:
            print(f'Adjusted pair {pair}')
            pair[1] = (pair[1] + 1) % n_total_qubits
            
    
    return subsystem_labels


def find_2PC_cluster(two_point_qubit_labels, quantum_correlation_array, subsystem_labels, max_clusters, cluster_limit = 0.1, update_list = False):
    """
    Finds the qubit labels that should be clustered together based on the quantum correlation coefficients.

    Parameters:
    two_point_qubit_labels (numpy.ndarray): List of qubit labels for two-point correlations n_correlators x 2.
    quantum_correlation_array (numpy.ndarray): Array of quantum correlation coefficients.
    subsystem_labels (numpy.ndarray): Array of subsystem labels. The correlation coefficients tells us
                                     how much the first index qubit is affected by the second index qubit.
    max_clusters (int): Maximum number of qubits in a cluster.
    cluster_limit(float): maximal value of the correlation coefficient to be clustered.

    Returns:
    numpy.ndarray: Array of qubit labels that should be clustered together.
    """
    if max_clusters <2:
        print(f"Max cluster size must be 2 or larger.")
        return two_point_qubit_labels
    
    cluster_qubits = []#np.empty((len(two_point_qubit_labels),max_clusters),dtype = int)
    
    for i in range(len(two_point_qubit_labels)):
        # Create a mask that removed all correlators not connected to any of the target qubits

        # Searches among two-way correlations
        mask = np.any(np.isin(subsystem_labels,two_point_qubit_labels[i] ),axis=1)
        #print(mask)
        # Apply mask to both the correlators and the labels. 
        cluster_correlators = quantum_correlation_array[mask]
        masked_subsystem_labels = subsystem_labels[mask]
        # Remove all correlators which are below cluster limit
        correlator_limit_mask = np.abs(cluster_correlators) > cluster_limit
        cluster_correlators = cluster_correlators[correlator_limit_mask]
        masked_subsystem_labels = masked_subsystem_labels[correlator_limit_mask]
        # Compute the highest correlators, returns the indecis of these correlators in the masked subsystem_labels

        highest_label_array = np.array(two_point_qubit_labels[i]) # Defined to just be part of the while argument. 
        it = 1
        if len(np.unique( np.append(masked_subsystem_labels,two_point_qubit_labels[i])))<max_clusters: # Checks if there are high enough correlators to warrent a search. 
            highest_label_array = np.unique(np.append(masked_subsystem_labels,two_point_qubit_labels[i]))
        else: # If there are enough candidates, select the ones with highest correlation coefficients. 
            while len(np.unique(highest_label_array))<max_clusters and len(np.unique(masked_subsystem_labels)): # It keeps adding one more qubit correlator untill the cluster is at max size. 
                indecis = np.argpartition(cluster_correlators, -(it))[-(it):]
                highest_label_array = np.append(masked_subsystem_labels[indecis],two_point_qubit_labels[i]).reshape(-1,2)
                it +=1
                if update_list: # The list of cluster qubits are updated with each added qubit. 
                    temp_subsystem_labels = np.unique(highest_label_array)

                    mask = np.any(np.isin(subsystem_labels,temp_subsystem_labels ),axis=1)
                    # Update masked arrays
                    cluster_correlators = quantum_correlation_array[mask]
                    masked_subsystem_labels = subsystem_labels[mask]
                    # Cut off the non-relevant correlators
                    correlator_limit_mask = np.abs(cluster_correlators) > cluster_limit
                    cluster_correlators = cluster_correlators[correlator_limit_mask]
                    masked_subsystem_labels = masked_subsystem_labels[correlator_limit_mask]
            
            if len(np.unique(highest_label_array))>max_clusters:
                print(f"Cluster {i} is larger than max cluster size {max_clusters}, removing lowest cluster")
                indecis = np.argpartition(cluster_correlators, -(it-2))[-(it-2):]
                highest_label_array = np.append(masked_subsystem_labels[indecis],two_point_qubit_labels[i]).reshape(-1,2)
        cluster_qubits.append(np.unique(highest_label_array))

        
    return cluster_qubits



def generate_random_hash(n_qubits,k_hash_symbols):
    hash = (np.arange(n_qubits))%k_hash_symbols
    np.random.shuffle(hash)
    return hash


def generate_kRDm_hash_brute(n_qubits,k_hash_symbols):
    """
    Brute force algorithm to create kRDM perfect hash family
    """
    def _is_hash_perfect(hash_list,k_hash_symbols):
        """
        Checks if hash list is perfect.
        """
        number_array = np.arange(len(hash_list[0]))
        k_array = np.arange(k_hash_symbols) 
        # Create all possible k-RDM labels
        check_array_index = np.array(list(combinations(number_array, k_hash_symbols)))
        masked_list = np.all(np.array([[np.isin(k_array,hash_list[:,line].T) for line in check] for check in check_array_index]),axis=(1,2))
        return  np.all(masked_list) # Return if all checks pass. 
    
    
    hash_list = np.array([(np.arange(n_qubits) )%k_hash_symbols])
    #hash_list = np.append(hash_list,[generate_random_hash(n_qubits, k_hash_symbols)],axis = 0)
    #print(hash_list)
    perfect_hash = False
    while perfect_hash == False:
    
        hash_list = np.append(hash_list,[generate_random_hash(n_qubits, k_hash_symbols)],axis = 0)
        #print(hash_list)
        perfect_hash = _is_hash_perfect(hash_list,k_hash_symbols)
        #print(f'Added hash \n{hash_list[-1]}')
    return hash_list
    
def generate_random_pairs_of_qubits(n_qubits,n_pairs):
    """
    Function generates n_pairs of random qubit pairs.
    """
    pairs = []
    for _ in range(n_pairs):
        q1 = np.random.randint(0,n_qubits)
        q2 = np.random.randint(0,n_qubits)
        while q1 == q2:
            q2 = np.random.randint(0,n_qubits)
        pairs.append([q1,q2])
    return np.array(pairs)

    

def conditioned_trace_out_POVM(povm, qubit_label_to_trace_out):
    """
    Traces out given qubit indices from the POVM elements. The POVM elements are not summed over.
    The new POVM elements are conditioned on the outcomes of the environment being traced out.  
    
    Inputs:
    - povm: POVM object to be traced out
    - qubit_label_to_trace_out: list of qubit indices to be traced out
    
    Returns:
    - traced_down_POVM: the traced down POVM with the specified qubit indices traced out
    """
    
    povm_list = povm.get_POVM()
    n_qubits = int(np.log2(len(povm_list[0])))
    # Convert qubit label to index to trace down. Sort to trace down the smallest list index first. 
    index_to_trace_out = np.sort(qubit_label_to_list_index(np.sort(qubit_label_to_trace_out),n_qubits))
    #print(f"Index to trace out{index_to_trace_out}")
    # Reshape povm elements to be easy to trace down. 
    dim_tuple = tuple([2]*n_qubits*2)
    
    traced_down_POVM = povm_list.reshape(tuple([-1]) + dim_tuple)

    for i in range(len(index_to_trace_out)): # trace out each dimension. The +1 comes from the povm list dimension. The -1 is to make sure to skip the correct amount of dimensions.
        traced_down_POVM = np.trace(traced_down_POVM, axis1 = index_to_trace_out[i]+1-i, axis2 = index_to_trace_out[i] + n_qubits-2*i + 1)

    # Reshape the povm to be in the correct shape. 
    return traced_down_POVM.reshape((-1,2**(n_qubits - len(index_to_trace_out)),2**(n_qubits - len(index_to_trace_out))))
     
def get_cluster_index_from_correlator_labels(cluster_labels, correlator_labels):
    """
    Takes in cluster structure and correlator labels and returns the cluster index for the clusters that contains the correlated labels.

    Parameters:
    cluster_labels (list): A list of cluster labels.
    correlator_labels (list): A list of correlator labels.

    Returns:
    list: A list of cluster indices.
    """
    cluster_index = []
    for correlator_label in correlator_labels:
        for j, cluster_label in enumerate(cluster_labels):
            if np.any(np.isin(correlator_label, cluster_label)) and (j not in cluster_index):
                cluster_index.append(j)
                
    return cluster_index
     
def reduce_cluster_POVMs(povm_list, cluster_label_list, correlator_labels):
    """ 
    Takes in a list of POVMs and their cluster labels, max lenght 2, and correlator labels lenght 2. 
    Returns a list of redced POVM elements. The order of the elements follows the decending order of qubit labels of the first cluster, then the second cluster.
    """
    

    def get_index_to_keep(cluster_label_list, correlator_labels):
        """
        Takes in a list of sorted cluster labels and a list of correlators, and returns the index of the qubit to keep from the correlator list. 

        Parameters:
        - cluster_label_list (list): A list of sorted cluster labels.
        - correlator_labels (list): A list of correlators.

        Returns:
        - index_to_keep (int): The index of the qubit to keep from the correlator list.

        Recursive solution for more qubits.
        """
        
        index_to_keep_1 = np.nonzero(np.equal(cluster_label_list, correlator_labels[0]))[0]
        index_to_keep_2 = np.nonzero(np.equal(cluster_label_list, correlator_labels[1]))[0]
        return np.concatenate((index_to_keep_1,index_to_keep_2),axis=0)
    
    # Check if there are two or one povm_list
    if len(povm_list)==1: # One POVM, either one or two correlator labels in same POVM. 
        cluster_label_list = np.sort(cluster_label_list[0])[::-1]
        povm = povm_list[0]
        # Need to reduce cluster labels to 4 qubit labels, and remove the qubit label of the correlator. 
        # Find the index of the entry that should be kept.
        index_to_keep = get_index_to_keep(cluster_label_list,correlator_labels)
        if len(index_to_keep)==povm.get_n_qubits():
           return [povm.get_POVM()]
        elif len(index_to_keep)>povm.get_n_qubits():
            print("Number of qubits to trace out is larger than qubits to the POVM dimension.")
            return None
        index_to_remove = np.setdiff1d(np.arange(len(cluster_label_list)),index_to_keep)
        label_to_remove = np.arange(len(cluster_label_list))[::-1][index_to_remove] 
        # Trace out the qubits from the POVMs 
        reduced_povm = conditioned_trace_out_POVM(povm, label_to_remove)
        return [reduced_povm]
        
    elif len(povm_list)==2: # Two separate cluster case. Recursive call. 
        reduced_povm_A = reduce_cluster_POVMs([povm_list[0]], [cluster_label_list[0]],correlator_labels)[0]
        reduced_povm_B = reduce_cluster_POVMs([povm_list[1]], [cluster_label_list[1]],correlator_labels)[0]
        return [reduced_povm_A,reduced_povm_B]
        
    else: # No case was met
        print("Number of povms was not met for POVM reduction")
        return None


def trace_down_qubit_state(state, state_labels, labels_to_trace_out):
    """
    Reduces the dimensionality of a quantum state by tracing out specified qubits.
    Parameters:
    state (np.ndarray): The input quantum state represented as a density matrix.
    state_labels (list): A list of labels corresponding to the qubits in the state.
    labels_to_trace_out (list): A list of labels corresponding to the qubits to be traced out.
    Returns:
    np.ndarray: The reduced quantum state after tracing out the specified qubits.
    Notes:
    - The function first checks if the labels to be traced out exist in the state labels.
    - If the number of qubits to trace out equals the number of qubits in the state, the original state is returned.
    - The function reshapes the state and performs partial trace operations to reduce the state.
    - The final reduced state is reshaped to the appropriate dimensions before being returned.
    """
    # Check if trace_out_label exist in state_labels:
    trace_out_mask = np.isin(labels_to_trace_out,state_labels)
    labels_to_trace_out = np.array(labels_to_trace_out)[trace_out_mask]
    n_state_qubits = len(state_labels)
    n_trace_out_qubits = len(labels_to_trace_out)
    state_labels = np.sort(state_labels)[::-1]
    labels_to_trace_out = np.sort(labels_to_trace_out)[::-1]
    if n_state_qubits == n_trace_out_qubits:
        print("No qubits to trace out. Returning original state.")
        return state
    n_return_qubits = n_state_qubits - n_trace_out_qubits

    # Get the qubit index to trace out 
    trace_out_qubit_index = np.sort(np.array([np.where(state_labels == label)[0][0] for label in labels_to_trace_out]))[::-1]
    
    # Reshape the state
    size = (2,)*(2*n_state_qubits)
    state=state.reshape((size))
    # Trace out each 2 dimnsion
    for i in range(len(trace_out_qubit_index)):
        state = np.trace(state, axis1 = trace_out_qubit_index[i], axis2 = trace_out_qubit_index[i] + n_state_qubits-i) 
    #print(f'final_cluster_state_shape: {cluster_state.shape}')
    return state.reshape((2**n_return_qubits,2**n_return_qubits))


def POVM_reduction_premade_cluster_QST(two_point, noise_cluster_labels, QST_outcomes, clustered_QDOT, hash_family, n_hash_symbols, n_qubits):
    '''
    Takes in the qubit label of two qubits, labels of the found noise clusters,
    the outcomes of the hashed QST measurements, and the reconstructed POVMs for those outcomes. 
    
    Returns the traced down reconstructed state for the two-point correlator.
    
    NOTE: This method is outclassed by newer cluster QST method, but is kept for reference and comparison.
    NOTE: Use method state_reduction_premade_cluster_QST instead.
    
    This method performs the following steps:
    - Finds the relevant clusters based on the two-point correlator.
    - Does a partial trace of the POVM, where we keep the knowledge of the environment state for each POVM element. 
    - The POVM elements used for QST are dictated by the full outcomes on all the clusters, 
      while the dimension of the elements is dictated by the two-point correlators 
      (generally these elements will be for two qubits or a tensor-product of single POVM elements).
    
    Parameters:
        two_point (list): The qubit label of two qubits.
        noise_cluster_labels (list): The labels of the found noise clusters.
        QST_outcomes (ndarray): The outcomes of the hashed QST measurements.
        clustered_QDOT (list): The reconstructed POVMs for those outcomes.
        hash_family (str): The hash family used for QST.
        n_hash_symbols (int): The number of hash symbols.
        n_qubits (int): The number of qubits.
    
    Returns:
        ndarray: The traced down reconstructed state for the two-point correlator.
    '''
    two_point = np.sort(two_point)[::-1]
    # Finds relevant clusters based on the two-point correlator. 
    relevant_cluste_index = get_cluster_index_from_correlator_labels(noise_cluster_labels, two_point)
    relevant_cluster_labels = [np.sort(noise_cluster_labels[index])[::-1] for index in relevant_cluste_index]
    relevant_cluster_POVMs = [clustered_QDOT[index] for index in relevant_cluste_index]
    # Trace down to the relevant cluster qubits.
    traced_out_cluster_outcomes = [trace_out(cluster,QST_outcomes) for cluster in relevant_cluster_labels]
    
    # Join the outcomes if there are two clusters.
    if len(relevant_cluster_labels) == 1:
        joined_outcomes = traced_out_cluster_outcomes[0]
    else:
        joined_outcomes = np.array([np.concatenate(tuple(traced_out_cluster_outcomes),axis = 2)])[0]

    # Find the total number of qubits.
    n_cluster_qubits = sum([len(cluster) for cluster in relevant_cluster_labels])
    # Reduce down the POVMs to the relevant correlator qubits while keeping full cluster outcome structure. 
    reduced_POVM = reduce_cluster_POVMs(relevant_cluster_POVMs,relevant_cluster_labels,two_point)

    if len(relevant_cluster_labels) == 1:
        tensored_reduced_POVM = POVM(reduced_POVM[0])
    else:
        tensored_reduced_POVM = POVM(np.array([np.kron(reduced_POVM_A,reduced_POVM_B) for reduced_POVM_A in reduced_POVM[0]  for reduced_POVM_B in reduced_POVM[1]]))
    # Collect outcomes to index-counts for the new tensor-product POVM structure. 
    decimal_outcomes = sf.binary_to_decimal_array(joined_outcomes)
    index_counts = np.array([np.bincount(outcomes,minlength =2**n_cluster_qubits) for outcomes in decimal_outcomes])
    # Reconstruct the RDM for the two-point correlator.
    rho_recon = QST(two_point, index_counts, hash_family, n_hash_symbols, n_qubits,  tensored_reduced_POVM)
    return rho_recon


def state_reduction_premade_cluster_QST(two_point_correlator_list, cluster_labels, QST_outcomes, clustered_QDOT, hash_family, n_hash_symbols, n_qubits):
    """
    Prefered method for creating correlator states from a premade cluster. 
    It is a memory inefficient wrapper for the cluster_QST + create_2RDMs_from_cluster_states methods. 
    
    """
    cluster_rho_recon = cluster_QST(QST_outcomes, cluster_labels, clustered_QDOT, hash_family, n_hash_symbols, n_qubits)
    correlator_rho_recon_list = create_2RDMs_from_cluster_states(cluster_rho_recon, cluster_labels, two_point_correlator_list)
    return correlator_rho_recon_list

def cluster_QST(QST_outcomes, cluster_labels, clustered_QDOT, hash_family, n_hash_symbols, n_qubits):
    """
    Perform QST on the cluster outcomes
    """
    cluster_QST_index_counts = [get_traced_out_index_counts(QST_outcomes, cluster_label) for cluster_label in cluster_labels]
    cluster_rho_recon = [QST(cluster_labels[i], cluster_QST_index_counts[i], hash_family, n_hash_symbols, n_qubits, clustered_QDOT[i]) for i in range(len(cluster_labels))]
    return cluster_rho_recon



def create_2RDMs_from_cluster_states(cluster_state_list, cluster_labels, correlator_labels_list):
    """
    Takes in a list of cluster states, cluster labels, and a list of correlators and returns a list of the two-qubit states for the correlators.
    
    Args:
        cluster_state_list (list): A list of cluster states.
        cluster_labels (list): A list of cluster labels.
        correlator_labels_list (list): A list of correlator labels.
        
    Returns:
        list: A list of 2RDMs (two-qubit reduced density matrices) for each correlator.
    """
    # Sort the cluster state labels such that the order is the same as the correlator labels.
    two_point_list = [np.sort(correlator_labels)[::-1] for correlator_labels in correlator_labels_list]
    
    # Find the relevant clusters for each correlator.
    relevant_cluster_index_list = [get_cluster_index_from_correlator_labels(cluster_labels, two_point) for two_point in two_point_list]
    relevant_cluster_labels_list = [[np.sort(cluster_labels[index])[::-1] for index in relevant_cluster_index] for relevant_cluster_index in relevant_cluster_index_list]
    relevant_cluster_states_list = [[cluster_state_list[index] for index in relevant_cluster_index] for relevant_cluster_index in relevant_cluster_index_list]
    label_to_trace_out = [[np.setdiff1d(relevant_cluster_labels_list[j][i], two_point_list[j]) for i in range(len(relevant_cluster_labels_list[j]))] for j in range(len(two_point_list))]

    # The traced down states have the form [[cluster1, cluster2], [cluster1, cluster2], ...] where the first index indicates the correlator label.
    traced_down_cluster_states_list = [[trace_down_qubit_state(relevant_cluster_states_list[j][i], relevant_cluster_labels_list[j][i], label_to_trace_out[j][i]) for i in range(len(relevant_cluster_labels_list[j]))] for j in range(len(two_point_list))]
    tensored_together_cluster = [reduce(np.kron, states) for states in traced_down_cluster_states_list]
    return tensored_together_cluster



def create_chunk_index_array(size_array, chunk_size):
    """Creates a list of indices for chunking an array based on the provided chunk size.

    This function assumes that the size array is distributed such that the chunks fit perfectly.
    The first index is 0 so slices can be written simply as size_array[index_array[i]:index_array[i+1]].

    Parameters:
    size_array (list or array-like): An array of sizes to be chunked.
    chunk_size (int): The desired size of each chunk.

    Returns:
    list: A list of indices indicating where each chunk ends.

    Example:
    >>> size_array = [2, 3, 4, 1, 5]
    >>> chunk_size = 5
    >>> create_chunk_index_array(size_array, chunk_size)
    [0, 2, 4, 5]
    """
    old_index = 0
    index = 1
    index_array = [0] # Include the 0 so slices can be written simply as size_array[index_array[i]:index_array[i+1]].
    while index<=len(size_array):
        while np.sum(size_array[old_index:index]) < chunk_size:
            index += 1
        index_array.append(index)
        old_index = index
        index += 1
    return index_array


def generate_chunk_sizes(chunk_size, n_chunks, cluster_cap):
    """
    Generates a list of chunk sizes for a given number of chunks, ensuring that each chunk
    is appropriately sized and does not exceed the specified cluster capacity.

    Parameters:
    chunk_size (int): The target size for each chunk.
    n_chunks (int): The number of chunks to generate.
    cluster_cap (int): The maximum size of any individual cluster within a chunk.

    Returns:
    list: A list of integers representing the sizes of the generated chunks.
    """
    cluster_size = []
    for i in range(n_chunks):
        local_chunk = []
        while sum(local_chunk) < chunk_size: # Make sure we apropriatly sized clusters.
            integer = np.random.randint(1, np.minimum(cluster_cap + 1, chunk_size - sum(local_chunk) + 1))
            local_chunk.append(integer)
        cluster_size.extend(local_chunk)
    return cluster_size
    
    
    
    
def entangled_state_reduction_premade_clusters_QST(two_point_correlator_list, cluster_labels, QST_outcomes, cluster_QDOT, hash_family, n_hash_symbols, n_qubits):
    #two_point_correlator_list, cluster_labels, QST_outcomes, clustered_QDOT, hash_family, n_hash_symbols, n_qubits):
    """
    The main difference between this function and the state_reduction_premade_cluster_QST is 
    that this function assumes that the sampled states can be entangled, which requires the full
    density matrix across both clusters to be reconstructed, instead of the density matrices for each cluster.
    # IMPORTANT: In tensoring together POVMs the order of the qubits matter, as the tensored together POVM will have the order:
    [[Cluster 1], [Cluster 2]], where they are ordered within each cluster. e.g. [[8,1,0], [6,5]]-> [8,1,0,6,5] (these are qubit labels)
    The stratergy is to swap the order of the qubits in the POVM to follow the normal qubit order, in the example [8,6,5,1,0].
    
    returns the reduced states and the qubit labels in decending order.
    """
    # Get relevant clusters for each two_point_correlator
    relevant_cluster_index_list = [get_cluster_index_from_correlator_labels(cluster_labels, two_point) for two_point in two_point_correlator_list]
    #print(relevant_cluster_index_list)
    # Target qubit label order.
    relevant_cluster_labels = [[np.sort(cluster_labels[index])[::-1] for index in relevant_cluster_index] for relevant_cluster_index in relevant_cluster_index_list]
    #print(relevant_cluster_labels)
    relevant_qubit_labels_unsorted = [list(chain.from_iterable(cluster)) for cluster in relevant_cluster_labels] 
    
    relevant_qubit_labels_sorted = [np.sort(cluster)[::-1] for cluster in relevant_qubit_labels_unsorted]
    #print(relevant_qubit_labels_sorted)
    
    traced_down_outcomes = [get_traced_out_index_counts(QST_outcomes, relevant_qubit_label) for relevant_qubit_label in relevant_qubit_labels_sorted]
    
    
    # Generate the the sorting order to sort the POVM qubits. 
    # index list that generates the sorted qubit order from the unsorted order (decending order)
    sorting_index_list = [np.argsort(label)[::-1] for label in relevant_qubit_labels_unsorted] 
    
    # # Generate inverting swap order by applying argsort twice
    # unsorting_index = [np.argsort(label) for label in sorting_index]  

    # unsorted_outcomes = [outcomes[unsorting_index] for outcomes, unsorting_index in zip(traced_down_outcomes, unsorting_index)]
    
    # At most two clusters are present.
    # The tensored together POVM will have the strict order of the qubits in the
    tensored_cluster_POVM_list = [cluster_QDOT[relevant_cluster_index[0]] if len(relevant_cluster_index) == 1 else  POVM.tensor_POVM(cluster_QDOT[relevant_cluster_index[0]], cluster_QDOT[relevant_cluster_index[1]])[0]  for relevant_cluster_index in relevant_cluster_index_list]
    # We will sort the POVM to occur in decending qubit order. 
    sorted_POVM_list = [POVM_sort(tensored_cluster_POVM, sorting_index)[0] for tensored_cluster_POVM, sorting_index in zip(tensored_cluster_POVM_list, sorting_index_list)]
    print(f'POVM shapes to be reconstructed:')
    for POVM in sorted_POVM_list:
        print(f'({len(POVM.get_POVM())}, {len(POVM.get_POVM())})')    
    # Reconstruct the states in decenting qubit label order.
    states = [QST(relevant_qubit_label, traced_down_outcome, hash_family,
                  n_hash_symbols, n_qubits, sorted_POVM) 
              for relevant_qubit_label, traced_down_outcome, sorted_POVM in 
              zip(relevant_qubit_labels_sorted,traced_down_outcomes,sorted_POVM_list)]

    # # swap order of the qubits such that they are in decending label oreder. 
    # relevant_qubit_labels_sorted
    # qubit_labels = np.copy(relevant_qubit_labels_unsorted)
    # for state in states:
    #     for i, expected_label in enumerate(relevant_qubit_labels_sorted):
    #         if qubit_labels[i]!= expected_label:
    #             swap_index = np.where(relevant_qubit_labels_unsorted == expected_label)[0]
    #             print(f'Swap index {swap_index}')
    #             state = swap_qubits(state,qubit_labels,np.array([swap_index,i]))
    #             # Swap label in qubit label order
    #             qubit_labels[[swap_index,i]] = qubit_labels[[i, swap_index]]
            
    return states, relevant_qubit_labels_sorted  

def POVM_sort(povm, sorting_index):
    """ 
    Sorts POVM accordng to sorting index order. Will swap the qubits and the outcome order. 
    NOTE: Function takes in sorting index and not sorting label!!!
    
    povm: POVM object.
    sorting_index: index array that if inserted as an argument for an array would sort it. Comes from np.argsort().
    """
    # Check if POVM already sorted:
    if np.all(sorting_index == np.arange(len(sorting_index),dtype=int)):
        #print("POVM already sorted.")
        return np.array([povm])
    
    n_qubits = len(sorting_index)
    wanted_order = np.arange(0,n_qubits,1,dtype=int)[::-1]
    # MAKE DEEP COPT IMPORTANT
    povm_array = np.copy(povm.get_POVM())
    #print(f'POVM shape {povm_array.shape}')
    # Find scrambling order
    scralbing_order = np.argsort(sorting_index)
    current_order = wanted_order[scralbing_order]
    # To swap a POVM qubit order we need to swap each individual matrix, and also their outcome order.
    for i in range(n_qubits):
        if current_order[i] != wanted_order[i]:
            # Find index in the current order to swap with i
            swap_label_index = np.where(current_order == wanted_order[i])[0][0]
            # Swap the qubit order in the effects
            povm_array = np.array([swap_qubits(matrix, current_order,np.array([current_order[i],current_order[swap_label_index]])) for matrix in povm_array ])

            # Swap outcome order
            # Outcomes are sorted in terms of 000, 001, 010. Rewrite index to binary array and swap array dimension, covert indecies back to decimal 
            index_array = np.arange(2**n_qubits, dtype = int)
            # Transcribe all otcomes to binary array
            binary_label_array = sf.decimal_to_binary_array(index_array, max_length = n_qubits)
            # Swap the binary entries
            binary_label_array[:,[i,swap_label_index]] = binary_label_array[:,[swap_label_index,i]]
            # Convert back to decimal
            swapped_decimal = sf.binary_to_decimal_array(binary_label_array)
            povm_array = povm_array[swapped_decimal]   
            # Swap qubits in the qubit label array
            current_order[[i,swap_label_index]] = current_order[[swap_label_index,i]]
            
    return np.array([POVM(povm_array)])        
    
def swap_qubits(rho, qubit_labels, labels_to_swap):
    """
    Swaps the dimensions of a density matrix based on the provided qubit labels.
    The qubit labels are assumed to be in a spesific order.
     
    Example:
    A qubit label of [5,1,2,4,8] and swap labels [1,4] will swap qubit 1 and qubit 3 of of the 
    qubit labels following the convention [..., 3, 2, 1, 0]. 
    
    Parameters:
    rho: single density matrix np.array
    qubit_label: the labels of qubits in the order they appear in rho
    labels_to_swap: list of two labels to be swapped. Order for these labels does not matter.
    """

    # Find the index of the label be swapped. 
    index_to_swap = np.array([np.where(qubit_labels == label)[0][0] for label in labels_to_swap])
    # Defien the size of the qubit state
    n_state_qubits = len(qubit_labels)
    size = (2,)*(2*n_state_qubits)
    rho = rho.reshape((size))
    # Swap the requiested axis
    rho  = np.swapaxes(rho,index_to_swap[0], index_to_swap[1])
    rho = np.swapaxes(rho,n_state_qubits + index_to_swap[0], n_state_qubits + index_to_swap[1])
    return rho.reshape((2**n_state_qubits,2**n_state_qubits))
    
    
def tensor_chunk_states(rho_list, state_label_array, povm_label_array, correlator_labels):
    """
    Creates the state that has has overlap between both POVM clusters of two-point correlators.
    
    Parameters:
    rho_true_list (list): A list of true states.
    state_size_array (list): A list of sizes for the states.
    povm_size_array (list): A list of sizes for the POVMs.
    correlator_label (list): A list of correlator labels.
    
    Returns:
    list: A list of tensor products of the states and POVMs in chunks.
    """
    # Create the chunk index arrays for the states and POVMs.
    relevant_povm_index_list = [get_cluster_index_from_correlator_labels(povm_label_array, two_point) for two_point in correlator_labels]

    # Target qubit label order.
    relevant_povm_label_list = [[povm_label_array[index] for index in relevant_povm_index] for relevant_povm_index in relevant_povm_index_list]

    relevant_povm_qubit_labels_sorted =  [np.sort(list(chain.from_iterable(cluster)))[::-1] for cluster in relevant_povm_label_list]

    # Find which state_labels are relevant for the correlator.
    relevant_state_index_list = [np.sort(get_cluster_index_from_correlator_labels(state_label_array, povm_label))for povm_label in relevant_povm_qubit_labels_sorted]
    
    # Find the relevant state labels
    relevant_labels = [[state_label_array[index] for index in relevant_state_index] for relevant_state_index in relevant_state_index_list]
    concatinated_labels = [np.concatenate(label, axis = 0) for label in relevant_labels]
    return [reduce(np.kron, [rho_list[index] for index in relevant_state_index]) for relevant_state_index in relevant_state_index_list], concatinated_labels



def create_QST_instructions(n_total_qubits,target_qubit_labels):
    """
    Creates instructions for Quantum State Tomography (QST) where only the target qubits are measured in specific bases (X, Y, Z),
    and all other qubits are measured in the computational basis (Z).
    Parameters:
    n_total_qubits (int): The total number of qubits in the system.
    target_qubit_labels (list of int): The labels of the qubits that are to be measured in the X, Y, or Z basis.
    Returns:
    numpy.ndarray: A 2D array where each row represents a unique combination of measurement instructions for the qubits.
    The target qubits will have instructions from the set {X, Y, Z}, and all other qubits will have 'Z'.              
    """  

    # Check if al qubit labels are smaller than the total number of qubits
    if np.max(target_qubit_labels) >= n_total_qubits:
        print("The target qubit labels are larger than the total number of qubits.")
        print("Ignoring qubit labels that falls outside range.")	
    taget_qubit_index = qubit_label_to_list_index(np.sort(target_qubit_labels)[::-1],n_total_qubits) 
    n_target_qubits = len(target_qubit_labels)
    # Need to create all possible instructions for the qubits we are measureing. 
    #The idea will be to create a full set QST instructions for each qubit
    elements = ['X','Y','Z']
    instructions = create_unique_combinations(elements, n_target_qubits, remove_duplicates=False)
    base_array = np.array([["Z"]*n_total_qubits]*len(instructions))
    for i in range(len(instructions)):
        base_array[i,taget_qubit_index] = instructions[i]
        
    return base_array


def QST_from_instructions(QST_outcomes, QST_instructions, two_point_correlators, relevant_qubit_labels, cluster_QDOT, cluster_labels):
    cluster_QST_index_counts = get_traced_out_index_counts(QST_outcomes, relevant_qubit_labels)
    # Trace down instructions to the relevant qubit labels
    #print(QST_instructions)
    traced_down_instructions = trace_out(relevant_qubit_labels, QST_instructions)

    
    
    # Sorting POVMs to the correct order
    relevant_cluster_index = get_cluster_index_from_correlator_labels(cluster_labels, two_point_correlators) 
    #print(relevant_cluster_index_list)
    # Target qubit label order.
    relevant_cluster_label = [np.sort(cluster_labels[index])[::-1] for index in relevant_cluster_index]
    relevant_qubit_label_unsorted = list(chain.from_iterable(relevant_cluster_label))
    
    relevant_qubit_labels_sorted = np.sort(relevant_qubit_label_unsorted)[::-1]
    sorting_index = np.argsort(relevant_qubit_label_unsorted)[::-1]

    tensored_cluster_POVM = cluster_QDOT[relevant_cluster_index[0]] if len(relevant_cluster_index) == 1 else  POVM.tensor_POVM(cluster_QDOT[relevant_cluster_index[0]], cluster_QDOT[relevant_cluster_index[1]])[0]  
    # We will sort the POVM to occur in decending qubit order. 
    sorted_POVM_list = POVM_sort(tensored_cluster_POVM, sorting_index)[0]
    instruction_eq= np.array([0, 1, 2])
    possible_instructions = np.array(['X', 'Y', 'Z'])
    translated_instruction = [instruction_equivalence(instruction, possible_instructions, instruction_eq) for instruction in traced_down_instructions]
        
    n_local_qubits = len(relevant_qubit_labels_sorted)
    if n_local_qubits < 6: # Run faster MLE
        reconstructed_Pauli_POVM = POVM.generate_Pauli_from_comp(sorted_POVM_list)
        combined_povm_array = subsystem_instructions_to_POVM(translated_instruction, reconstructed_Pauli_POVM, n_local_qubits) 
        rho_recon = OT_MLE(combined_povm_array, cluster_QST_index_counts)
    
    else: # Runs memory efficient MLE
        # Create rotation array
        povm_rotators = generate_pauli_6_rotation_matrice(n_local_qubits)
        #povm_rotators = subsystem_instructions_to_POVM(translated_instruction, povm_rotators, n_local_qubits) 
        rho_recon = OT_MLE_efficient(sorted_POVM_list, povm_rotators, cluster_QST_index_counts)
    return rho_recon



def parallel_QDOT(QDT_subsystem_labels, QDT_index_counts, hash_family, n_hash_symbols, n_qubits, one_qubit_calibration_states):
    """
    Function defined to be used in parallel for the POVM reconstruction.
    """
    # Reconstruct the POVMs in parallel
    reconstructed_comp_POVM = QDT(QDT_subsystem_labels,QDT_index_counts, hash_family, n_hash_symbols, n_qubits, one_qubit_calibration_states)
    return reconstructed_comp_POVM



def get_all_subsystem_labels(n_qubits):
    """
    Function creates all possible combinations of two qubit correlation labels, and corresponding qubit label they  labels.
    """
    QDT_subsystem_labels = np.array([[1,0]])
    # Since the quantum correlation coefficients are not symmetric, we need to have all combinations with swapped orders too.
    # The 0 index qubit is the one that we want to know the effects from the 1 index qubit on.
    # The POVM is reconstructed with the largest qubit number first. The correlator traces out the 0 qubit first, so we have the ordere [1,0] then [0,1].
    corr_subsystem_labels = np.array([[1,0]])
    for i in range(n_qubits):
        for j in range(i):
            QDT_subsystem_labels = np.append(QDT_subsystem_labels, [[i,j]], axis = 0)
            corr_subsystem_labels = np.append(corr_subsystem_labels, [[i,j],[j,i]], axis = 0)
    QDT_subsystem_labels = QDT_subsystem_labels[1:]
    corr_subsystem_labels = corr_subsystem_labels[1:]
    return QDT_subsystem_labels, corr_subsystem_labels


def reconstruct_all_two_qubit_POVMs(QDT_outcomes, n_qubits, hash_family, n_hash_symbols, one_qubit_calibration_states, n_cores):
    QDT_subsystem_labels, corr_subsystem_labels = get_all_subsystem_labels(n_qubits)
    print(f'Number of 2 qubit POVMs to reconstruct: {len(QDT_subsystem_labels)}')
    two_point_POVM = reconstruct_spesific_two_qubit_POVMs(QDT_outcomes, QDT_subsystem_labels , n_qubits, hash_family, n_hash_symbols, one_qubit_calibration_states, n_cores)
    return two_point_POVM, corr_subsystem_labels


def reconstruct_spesific_two_qubit_POVMs(QDT_outcomes, QDT_subsystem_labels , n_qubits, hash_family, n_hash_symbols, one_qubit_calibration_states, n_cores):
    QDT_index_counts =  Parallel(n_jobs = 1, verbose = 1)(delayed(get_traced_out_index_counts)(QDT_outcomes, subsystem_label) for subsystem_label in QDT_subsystem_labels)
    QDT_index_counts = np.asarray(QDT_index_counts)
    two_point_POVM = Parallel(n_jobs = n_cores, verbose = 1)(delayed(parallel_QDOT)(QDT_subsystem_labels[i], QDT_index_counts[i], hash_family, n_hash_symbols, n_qubits, one_qubit_calibration_states) for i in range(len(QDT_subsystem_labels)))
    two_point_POVM = np.asarray(two_point_POVM)
    return two_point_POVM


def reconstruct_all_one_qubit_POVMs(QDT_outcomes, n_qubits, hash_family, n_hash_symbols, one_qubit_calibration_states, n_cores):
    # Create all 1 qubit POVMS for comparison
    one_qubit_subsystem_labels = np.array([[i] for i in range(n_qubits)])[::-1] # This creates qubit label order [..., 3,2,1,0]
    one_qubit_QDT_index_counts = [get_traced_out_index_counts(QDT_outcomes, subsystem_label) for subsystem_label in one_qubit_subsystem_labels]
    one_qubit_POVMs = Parallel(n_jobs = n_cores,verbose = 1)(delayed(parallel_QDOT)(one_qubit_subsystem_labels[i], one_qubit_QDT_index_counts[i], hash_family, n_hash_symbols, n_qubits, one_qubit_calibration_states) for i in range(len(one_qubit_subsystem_labels)))
    return one_qubit_POVMs


def reconstruct_POVMs_from_noise_labels(QDT_outcomes,noise_cluster_labels, n_qubits, hash_family, n_hash_symbols, one_qubit_calibration_states, n_cores ):
    # Create a all POVMS for the noise clusters
    QDT_index_counts = [get_traced_out_index_counts(QDT_outcomes, subsystem_label) for subsystem_label in noise_cluster_labels]
    clustered_QDOT = Parallel(n_jobs = n_cores,verbose = 10)(delayed(parallel_QDOT)(noise_cluster_labels[i], QDT_index_counts[i], hash_family, n_hash_symbols, n_qubits, one_qubit_calibration_states) for i in range(len(noise_cluster_labels)))
    return clustered_QDOT



def compute_quantum_correlation_coefficients(two_point_POVM, corr_subsystem_labels, mode="WC", wc_distance_ord = None):	
    """
    Compute the quantum correlation coefficients with selected mode, either worse case or average case.
    """
    quantum_corr_array = [povm.get_quantum_correlation_coefficient(mode, wc_distance_ord = wc_distance_ord).flatten() for povm in two_point_POVM]
    summed_array_V2 = np.array([quantum_corr_array[i][0] + quantum_corr_array[i][1] for i in range(len(quantum_corr_array))])/2
    summed_quantum_corr_array = summed_array_V2.flatten()
    
    unique_corr_labels = corr_subsystem_labels[::2] # Takes out every other label, since the neighbouring label is the swapped qubit labels. 
    
    # quantum_corr_array = np.asarray(quantum_corr_array)
    # quantum_corr_array = quantum_corr_array.flatten()
    # summed_quantum_corr_array = np.array([quantum_corr_array[2*i] + quantum_corr_array[2*i+1] for i in range(len(quantum_corr_array)//2)])

    # if np.all(compareative_sum == summed_quantum_corr_array):
    #     print("Summed quantum correlation array is equal to the sum of the two quantum correlation coefficients.")
    return summed_quantum_corr_array, unique_corr_labels



def factorized_state_list_to_correlator_states(two_point_corr_labels, factorized_state_list, n_qubits):
    """
    Function converts a list of factorized states to a list of correlator states.
    """
    correlator_states = []
    for j in range(len(factorized_state_list)):
        correlator_states.append([])
        for i in range(len(two_point_corr_labels)):
            subsystem_index = qubit_label_to_list_index(np.sort(two_point_corr_labels[i])[::-1], n_qubits) 
            qubit_sublist = factorized_state_list[j][subsystem_index]
            correlator_states[j].append(reduce(np.kron, qubit_sublist))
    return correlator_states


def generate_random_pauli_string(n_samples,n_qubits):
    """
    Function generates a random Pauli string of length n_elements.
    """
    pauli_1 = np.eye(2)
    pauli_x = np.array([[0,1],[1,0]])
    pauli_y = np.array([[0,-1j],[1j,0]])
    pauli_z = np.array([[1,0],[0,-1]])
    pauli_operators = np.array([pauli_1,pauli_x, pauli_y, pauli_z])
    pauli_string = np.random.randint(0,4,n_samples*n_qubits).reshape(n_samples,n_qubits)
    op_list = pauli_operators.take(pauli_string,axis = 0)  
    return np.array([reduce(np.kron, op_list[i]) for i in range(n_samples)]) 


def compute_op_and_n_averages_mean_MSE(exp_value_array, true_exp_value):
    """
    exp values comes in the shape of (n_method,n_averages, n_correlators, n_ops)
    true_exp_value comes in the shape of (n_averages, n_correlators, n_ops)
    Returns the mean MSE of the exp values on the form of n_method x n_correlators
    """
    
    return np.array([np.mean((true_exp_value - method)**2,axis = (0,2)) for method in exp_value_array])

def compute_state_array_exp_values(state_array,op_array):
    """
    Computes the expectation value of the states with shape (n_modes, n_averages, len(two_point_corr_labels), 2**n_qubits, 2**n_qubits)
    with a list of operators with shape (n_ops, 2**n_qubits, 2**n_qubits)
    returns the expectation value with shape (n_modes, n_averages,  len(two_point_corr_labels),n_ops)
    """
    return np.einsum('mncij,oji->mnco', state_array, op_array).real



def compute_double_list_of_infidelities(rho_array,rho_true_array):
    """
    Computes compuesa a lit of infidelities of the states that has the shape n_average, len(two_point_corr_labels), 2**n_qubits, 2**n_qubits
    """
    return np.array([[np.real(sf.qubit_infidelity(rho,rho_ture)) for rho, rho_ture in zip(rho_array[n],rho_true_array[n])] for n in range(len(rho_array))])


def compute_k_mean_expectation_values(state_matrix,op_string_array):
    """
    Computes expectation values for a given state matrix.
    state_matrix shape has [n_k_mean, n_modes, n_averages, len(two_point_corr_labels) , 2**n_qubits, 2**n_qubits]
    returns expectation values of shape [n_k_mean, n_modes, n_averages, len(two_point_corr_labels), n_op]
    """
    return np.array([compute_state_array_exp_values(matrix,op_string_array) for matrix in state_matrix])

def compute_k_mean_mean_MSE(exp_value_array, true_exp_value):
    """
    Computes the mean MSE for a given expectation value array
    true_exp_value is of shape [n_averages, len(two_point_corr_labels), n_op]
    exp_value_array is of shape [n_k_mean, n_modes, n_averages, len(two_point_corr_labels), n_op]
    returns an array of shape [n_k_mean, n_modes]
    """
    return np.array([[np.mean((true_exp_value - mode)**2) for mode in k_mean] for k_mean in exp_value_array])

def compute_mode_mean_infidelitites(rho_array, rho_true_array):
    """
    Computes the mean of n_average infidelities for each recon_mode. See mean_double_list_infidelities for more info.
    Retursn list of infidelities of shape [n_recon_modes, n_averages]
    """
    full_inf_array = np.array([compute_double_list_of_infidelities(recon_mode, rho_true_array) for recon_mode in rho_array])
    # Full inf array has the shape [n_recon_modes, n_averages, len(two_point_corr_labels)]
    return np.mean(full_inf_array, axis = 1) # Average over the n_averages


def k_mean_infidelity_computation(state_matrix, rho_true_array):
    """
    Computes the inifdelities to be plotted for the k-mean plot
    state_matrix comes in shape [n_k_mean, n_modes, n_averages, len(two_point_corr_labels) , 2**n_qubits, 2**n_qubits]
    rho_true_array comes in shape [n_averages, n_two_point_corr, 2**n_qubits, 2**n_qubits]

    will return infidelities averaged over n_averages and n_two_point_corr, final shape will be [n_k_mean, n_modes]
    """
    mode_mean_inf = [compute_mode_mean_infidelitites(k_mean,rho_true_array) for k_mean in state_matrix]
    # Returns a list of arrays of shape [n_k_mean, n_modes, len(two_point_corr_labels)]
    return np.mean(mode_mean_inf, axis = 2)


def is_state_array_physical(state_array):
    '''
    Checks the QST comparison arrays are physical.
    '''
    physical_array = [[[sf.is_state_physical(state) for state in corr] for corr in average] for average in state_array]
    if np.all(physical_array):
        print("All states are physical.")
        return True, physical_array
    else:
        print("Not all states are physical.")
        print("Returning is_physical array.")
        return False, physical_array




        
def perform_comparative_QST(noise_cluster_labels,  two_point_corr_label, QST_outcomes,
                                 clustered_QDOT, one_qubit_POVMs, two_point_POVM, n_averages, 
                                 n_qubits,comparison_methods, target_qubits, QST_instructions,):
    """
    
    comparison_methods: list of integers that selects which methods to compare to correlated QREM.
    0: no QREM
    1: factorized QREM
    2: two RDM QREM
    3: Classical correlated QREM
    """
    result_array = []
    # Need to create index counts for the compared methods
    traced_index_counts = np.array([get_traced_out_index_counts(QST_outcomes[i], two_point_corr_label) for i in range(n_averages)])
    # Trace down instructions to just the two-point qubits and translate to integers.
    two_point_traced_instructions = trace_out(two_point_corr_label, QST_instructions)
    two_point_POVM_instuctions = [instruction_equivalence(instruction, ['X','Y','Z'], [0,1,2]) for instruction in two_point_traced_instructions]

    
    if 0 in comparison_methods: # No QREM
        # To create naiv instruction we supply a standard Pauli-POVM
        naive_POVM =  POVM.generate_Pauli_POVM(len(two_point_corr_label))
        naive_POVM_instructions = subsystem_instructions_to_POVM(two_point_POVM_instuctions, naive_POVM, len(two_point_corr_label))
        no_QREM_two_RDM_recon = [OT_MLE(naive_POVM_instructions, index_counts)for index_counts in traced_index_counts] # Each index count is for one of the n_averages states
        result_array.append(no_QREM_two_RDM_recon)
        
    if 1 in comparison_methods: # Factorized QREM
        
        two_index = qubit_label_to_list_index(np.sort(two_point_corr_label)[::-1], n_qubits)
        factorized_POVMs = POVM.tensor_POVM(one_qubit_POVMs[two_index[0]],one_qubit_POVMs[two_index[1]])[0]
        factorized_pauli_POVM = POVM.generate_Pauli_from_comp(factorized_POVMs)
        factorized_POVM_instructions = subsystem_instructions_to_POVM(two_point_POVM_instuctions, factorized_pauli_POVM, len(two_point_corr_label))
        factorized_rho_recon =  [OT_MLE(factorized_POVM_instructions, index_counts) for index_counts in traced_index_counts]
        result_array.append(factorized_rho_recon)
        
    if 2 in comparison_methods: # Two-point REMST method
        two_point_Pauli_POVM = POVM.generate_Pauli_from_comp(two_point_POVM)
        two_point_POVM_instructions = subsystem_instructions_to_POVM(two_point_POVM_instuctions, two_point_Pauli_POVM, len(two_point_corr_label))
        two_point_rho_recon = [OT_MLE(two_point_POVM_instructions, index_counts) for index_counts in traced_index_counts]
        result_array.append(two_point_rho_recon)
        
    if 3 in comparison_methods: # Classical correlated QREM
        # Create classical POVM from the reconstructed one
        classical_povm = [povm.get_classical_POVM() for povm in clustered_QDOT]
        classical_QREM_recon = [QST_from_instructions(outcome, QST_instructions, np.array([two_point_corr_label]), target_qubits, classical_povm, noise_cluster_labels)for outcome in QST_outcomes]
        traced_down_classical_recon = [trace_down_qubit_state(recon, target_qubits, np.setdiff1d(target_qubits, two_point_corr_label)) for recon in classical_QREM_recon]
        result_array.append(traced_down_classical_recon)
        
    # Correlator QREM, always computed
    correlator_QREM_recon = [QST_from_instructions(outcome, QST_instructions, np.array([two_point_corr_label]), target_qubits, clustered_QDOT, noise_cluster_labels) for outcome in QST_outcomes]
    traced_down_correlator_recon = [trace_down_qubit_state(recon, target_qubits, np.setdiff1d(target_qubits, two_point_corr_label)) for recon in correlator_QREM_recon]
    #print(traced_down_correlator_recon)
    result_array.append(traced_down_correlator_recon)
    return result_array
            

# def perform_full_comparative_QST(noise_cluster_labels, QST_outcomes_array, two_point_corr_labels,
#                                  clustered_QDOT, one_qubit_POVMs, two_point_POVM, n_averages, hash_family, n_hash_symbols, n_qubits, n_cores, method=None):
#     """
#     This method is outdated and should not be used. It is kept for reference only.
#     Function that can perform all the different QST methods. 
#     The method argument can be used to select which methods to perform.
#     The methods are labeld as follows:
#     0: no QREM
#     1: factorized QREM
#     2: two RDM QREM
#     3: Cluster-concious two-point QREM
#     4: Classical correlated QREM
#     5: Entanglement safe QREM
#     There are in addition two more methods that are not included which can be accessed with the following labels:
#     6: Classical state reduction QREM
#     7: State reduction QREM
#     """
    
#     state_reduction_rho_average_array = []
#     two_RDM_QREM_rho_average_array = []
#     no_QREM_rho_average_array = []
#     povm_reduction_rho_average_array = []
#     factorized_QREM_rho_average_array = []
#     classical_cluster_QREM_rho_average_array = []
#     entanglement_safe_QREM_rho_average_array = []
#     classical_entangelment_safe_QREM_rho_average_array = []

#     if method is None: # Defaults to no QREM. 
#         method = [0]
#         print(f'No method selected. Defaults to no QREM.')
    
    
#     for k in range(n_averages):
#         QST_outcomes = QST_outcomes_array[k]      
#         # Simplifed methods
#         naive_QST_index_counts = [get_traced_out_index_counts(QST_outcomes, two_point) for two_point in two_point_corr_labels]
#         classical_povm = [povm.get_classical_POVM() for povm in clustered_QDOT]

#         # Reconstruction step 1)
#         if 0 in method:
#             no_QREM_two_RDM_recon = [QST(two_point_corr_labels[i], naive_QST_index_counts[i], hash_family, n_hash_symbols, n_qubits, POVM.generate_computational_POVM(len(two_point_corr_labels[i]))[0]) for i in range(len(two_point_corr_labels))]
#             no_QREM_rho_average_array.append(no_QREM_two_RDM_recon)
        
#         # Reconstruction step 2) 
#         # We have facorized POVMs, just need to tensor them together.
#         if 1 in method:
#             factorized_POVMs = np.empty(len(two_point_corr_labels),dtype = object)
#             for i,two_point in enumerate(two_point_corr_labels):
#                 two_index = qubit_label_to_list_index(np.sort(two_point)[::-1], n_qubits)
#                 factorized_POVMs[i] = POVM.tensor_POVM(one_qubit_POVMs[two_index[0]],one_qubit_POVMs[two_index[1]] )[0]
    
#             factorized_rho_recon = np.array([QST(two_point_corr_labels[i], naive_QST_index_counts[i], hash_family, n_hash_symbols, n_qubits, factorized_POVMs[i]) for i in range(len(two_point_corr_labels))])
#             factorized_QREM_rho_average_array.append(factorized_rho_recon)   
        
    
#         # Reconstruction step 3)
#         if 2 in method:
#             two_RDM_state_recon = np.array([QST(two_point_corr_labels[i], naive_QST_index_counts[i], hash_family, n_hash_symbols, n_qubits, two_point_POVM[i]) for i in range(len(two_point_corr_labels))])
#             two_RDM_QREM_rho_average_array.append(two_RDM_state_recon)
    
    
#         # Reconstruction step 4)
#         if 3 in method:
#             povm_reduction_rho_list = Parallel(n_jobs = n_cores, verbose = 10)(delayed(POVM_reduction_premade_cluster_QST)(two_point,noise_cluster_labels, 
#                                                                                                                         QST_outcomes, clustered_QDOT, hash_family,
#                                                                                                                         n_hash_symbols, n_qubits) for two_point in two_point_corr_labels)
#             povm_reduction_rho_average_array.append(povm_reduction_rho_list)
    
#         # Reconstruction step 5) Classical entanglement safe QREM
#         if 4 in method:
#             classical_entangled_recon_states, classical_entangled_qubit_order = entangled_state_reduction_premade_clusters_QST(two_point_corr_labels,
#                 noise_cluster_labels, QST_outcomes, classical_povm, hash_family, n_hash_symbols, n_qubits)
#             traced_down_classical_entangled_recon= [trace_down_qubit_state(classical_entangled_recon_states[i], classical_entangled_qubit_order[i], np.setdiff1d(classical_entangled_qubit_order[i], two_point_corr_labels[i])) for i in range(len(classical_entangled_recon_states))]
#             classical_entangelment_safe_QREM_rho_average_array.append(traced_down_classical_entangled_recon)
        
        
#         # Reconstruction step 6) Entanglement safe QREM
#         if 5 in method:
#             entangled_recon_states, entangled_qubit_order = entangled_state_reduction_premade_clusters_QST(two_point_corr_labels,
#                 noise_cluster_labels, QST_outcomes, clustered_QDOT, hash_family, n_hash_symbols, n_qubits)

#             traced_down_entangled_recon = [trace_down_qubit_state(entangled_recon_states[i], entangled_qubit_order[i], np.setdiff1d(entangled_qubit_order[i], two_point_corr_labels[i])) for i in range(len(entangled_recon_states))]
#             entanglement_safe_QREM_rho_average_array.append(traced_down_entangled_recon)
            
            
#         # Outdated legacy methods classical state reduction method 
#         if 6 in method:
#             classical_rho_recon = state_reduction_premade_cluster_QST(two_point_corr_labels, noise_cluster_labels, QST_outcomes, 
#                                                                     classical_povm, hash_family, n_hash_symbols, n_qubits)
#             classical_cluster_QREM_rho_average_array.append(classical_rho_recon)
            
#         # Outdated legacy methods state reduction method
#         if 7 in method:
#             state_reduction_rho_list = state_reduction_premade_cluster_QST(two_point_corr_labels, noise_cluster_labels, QST_outcomes, 
#                                                                         clustered_QDOT, hash_family, n_hash_symbols, n_qubits)
#             state_reduction_rho_average_array.append(state_reduction_rho_list)


#     result_QST_dict = {
#     # The states has the shape (n_averages, len(two_point_corr_labels), 2**n_qubits, 2**n_qubits)
#     "method": method, # Tells us which methods were used.
#     "state_reduction_rho_average_array": state_reduction_rho_average_array,
#     "two_RDM_QREM_rho_average_array": two_RDM_QREM_rho_average_array,
#     "no_QREM_rho_average_array": no_QREM_rho_average_array,
#     "povm_reduction_rho_average_array": povm_reduction_rho_average_array,
#     "factorized_QREM_rho_average_array": factorized_QREM_rho_average_array,
#     "classical_cluster_QREM_rho_average_array": classical_cluster_QREM_rho_average_array,
#     "entanglement_safe_QREM_rho_average_array": entanglement_safe_QREM_rho_average_array,
#     "classical_entangelment_safe_QREM_rho_average_array": classical_entangelment_safe_QREM_rho_average_array,
#     "two_point_corr_labels": two_point_corr_labels,          
#     "n_average": n_averages
# }
#     return result_QST_dict

def load_state_array_from_result_dict(result_dict):
    """
    Loads data of state averaging. Will only load non-empty arrays.
    """
    method = np.array(result_dict['comparison_modes'],dtype = int)
    method = np.append(method, 4) # To always include the correlated QREM
    no_QREM_rho_array = result_dict['no_QREM']
    factorized_QREM_rho_array = result_dict['factorized_QREM']
    two_RDM_QREM_rho_array = result_dict['two_RDM_QREM']
    classical_correlated_QREM_rho_array = result_dict['classical_correlated_QREM']
    correlated_QREM_rho_array = result_dict['correlated_QREM']
    
    # state_reduction_rho_average_array = result_dict['state_reduction_rho_average_array']
    # two_RDM_QREM_rho_average_array = result_dict['two_RDM_QREM_rho_average_array']
    # no_QREM_rho_average_array = result_dict['no_QREM_rho_average_array']
    # povm_reduction_rho_average_array = result_dict['povm_reduction_rho_average_array']
    # factorized_QREM_rho_average_array = result_dict['factorized_QREM_rho_average_array']
    # classical_cluster_QREM_rho_average_array = result_dict['classical_cluster_QREM_rho_average_array']
    # entanglement_safe_QREM_rho_average_array = result_dict['entanglement_safe_QREM_rho_average_array']
    # classical_entangelment_safe_QREM_rho_average_array = result_dict['classical_entangelment_safe_QREM_rho_average_array']

    # state_array = [no_QREM_rho_average_array,
    #             factorized_QREM_rho_average_array,
    #             two_RDM_QREM_rho_average_array,
    #             povm_reduction_rho_average_array,
    #             classical_entangelment_safe_QREM_rho_average_array,
    #             entanglement_safe_QREM_rho_average_array,
    #             classical_cluster_QREM_rho_average_array,
    #             state_reduction_rho_average_array
    # ]
    
    state_array = [no_QREM_rho_array,
                factorized_QREM_rho_array,
                two_RDM_QREM_rho_array,
                classical_correlated_QREM_rho_array,
                correlated_QREM_rho_array
    ]
    label_array = ['No QREM ',
                'Factorized QREM',
                'Two-point QREM',
                'Classical correlated QREM',
                'Correlated QREM',
    ]
    return [state_array[it] for it in method], [label_array[it] for it in method]





def find_all_args_of_label(corr_labels, label_to_find):
    """
    Finds all indices of a specified label in a list of correlation labels.

    Args:
        corr_labels (ndarray): An array of correlation labels.
        label_to_find (int): The label to search for within the list.
    Returns:
        ndarray: An array of indices where the specified label is found in the input list.
    """
    return np.array([i for i, label in enumerate(corr_labels) if np.any(label == label_to_find)])