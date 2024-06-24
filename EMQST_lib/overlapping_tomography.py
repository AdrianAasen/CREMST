import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import os
import uuid
from functools import reduce
from itertools import product, chain, repeat, combinations
from EMQST_lib import support_functions as sf
from EMQST_lib.povm import POVM
from EMQST_lib import dt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm



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
    - possible_instructions (list): A list of all possible accepted.

    Returns:
    - list: The translated instruction.
    
    NOTE: This function is not optimized for speed, if keys and value are the same, e.g. {1:2, 2:3}, you get a bug where both 1 and 2 is set

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
    combined_povm_array = subsystem_instructions_to_POVM(combined_hash_instructions, reconstructed_Pauli_POVM, n_subsystem_qubits = n_subsystem_qubits)
    return combined_povm_array



def subsystem_instructions_to_POVM(instructions, reconstructed_Pauli_POVM, n_subsystem_qubits ):
    """
    Takes in an instruction and turns it into a Pauli POVM element.
    E.g. [0,1] -> [X,Y] reconstructed POVM element
    """

    base_3 = np.array([3**i for i in range(n_subsystem_qubits)])[::-1]
    
    povm_array = np.array([reconstructed_Pauli_POVM[np.dot(instruction,base_3)] for instruction in instructions])
    return povm_array



def get_traced_out_index_counts(outcomes, subsystem_label):
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

    while j<iter_max and dist>1e-14:
        p      = np.einsum('ik,nki->n', rho_1, OP_list)
        R      = np.einsum('n,n,nij->ij', index_counts, 1/p, OP_list)
        update = R@rho_1@R
        rho_1  = update/np.trace(update)

        if j>=40 and j%100==0:
            dist  = sf.qubit_infidelity(rho_1, rho_2)
        rho_2 = rho_1
        j += 1
    return rho_1

def QST(subsystem_label, QST_index_counts, hash_family, n_hash_symbols, n_qubits, reconstructed_comp_POVM):
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
    hashed_subsystem_reconstructed_Pauli_6 = create_traced_out_reconstructed_POVM(subsystem_label, reconstructed_comp_POVM, hash_family, n_hash_symbols, n_qubits)
    rho_recon = OT_MLE(hashed_subsystem_reconstructed_Pauli_6, QST_index_counts)
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

    
    #test_list = np.array(list(itertools.combinations(number_list, 4)))
    
    def _is_hash_perfect(hash_list,k_hash_symbols):
        """
        Checks if hash list is perfect.
        """
        number_array = np.arange(len(hash_list[0]))
        k_array = np.arange(k_hash_symbols) 
        # Create all possible k-RDM labels
        check_array_index = np.array(list(combinations(number_array, k_hash_symbols)))

        #print(hash_list[:,check_array_index[0]].T)
        #print(k_array)
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
    

# def trace_out_outcomes(qubit_to_keep_labels, outcomes):
#     """
#     Trace out qubits from a measurement outcome.
#     Input: 
#     qubit_labels: list of labels of qubits to keep, order does not matter. [n] 
#     outcomes: Measurement outcomes. Could be in any shape. 
#     return: the downconverted outcomes.
#     """
#     qubit_to_keep_labels = np.sort(qubit_to_keep_labels) # Sort labels such that input order does not matter
#     #qubit_index = qubit_label_to_list_index(qubit_labels, len(outcomes.shape))
    
#     # Check the index for it's binary representation by whole number division and then modulo 2
#     floor_div = np.array([np.floor_divide(outcomes,2**index)%2 for index in qubit_to_keep_labels ])
    
#     # Create binary representationf or the current string
#     binary = 2**np.arange(len(qubit_to_keep_labels))
    
#     # Multiply the binary representation with their the index of the new system. 
#     donconverted_outcomes = np.einsum('i,i...->...',binary,floor_div)
#     return donconverted_outcomes

# def downconvert_frequencies(subsystem_index,outcome_frequencies):
#     """
#     Takes in the outcome frequency measurement of the whole system and a set of qubit subsystem indices,
#     and return the downconverted frequencies.
#     Here the row structure matters, so outcome frequencies should be flattend to be m x n_outcomes. One can reshape back to original structure afterwards.
#     Input:
#         - system_index ndarray [n_subsystem_qubits]
#         - outcome_frequencies ndarray n x 2**n_qubits_total
#     Return:
#         - The downconverted frequencies ndarray n x 2**len(subsystem_index)
#     """
#     # Check how many qubits are in the total system
#     n_qubits_total = int(np.log2(len(outcome_frequencies[0])))
    
#     # Define the axis that need to be traced out
#     # Find the indices that are not present int he above, and invert the order (such that qubit 0 is axis n_qubit_total). 
#     traced_indices = tuple(n_qubits_total-1 - get_traced_out_indicies(subsystem_index,n_qubits_total))
    
#     # Define the subsystem shape
#     reshape_tuple = (2,)*n_qubits_total

#     # Sum over all cases where the subsystem indecies are the same, then reshape to be in the same shape as original list. 
#     downconverted_frequencies = np.array([np.sum(measurement.reshape(reshape_tuple), axis = traced_indices).reshape(2**len(subsystem_index)) for measurement in outcome_frequencies])
#     return downconverted_frequencies
