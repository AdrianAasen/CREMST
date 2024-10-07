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
import copy



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



def is_pair_in_more_than_one_cluster(pair_label, clusters):
    """
    Check if a pair is in more than one cluster.
    Takes in a pair of qubits, and check the list of clusters if the pair is present in more than one cluster.
    """
    counter = 0
    for cluster in clusters:
        if np.any(np.isin(pair_label,cluster)):
            counter +=1
    if counter > 1:
        return True
    else:    
        return False
    
def find_clusters_from_correlator_labels(correlator_labels, clusters):
    """
    From the correlation labels finds the clusters that contain any of the labels.

    Parameters:
    - correlator_labels (list): A list of correlation labels.
    - clusters (list): A list of clusters.

    Returns:
    - return_cluster (list): A list of clusters that contain any of the correlator labels.
    """
    return_cluster = []
    for label in correlator_labels:
        temp_cluster = []
        for cluster in clusters:
            if np.any(np.isin(cluster, label)): # label is in cluster
                temp_cluster.append(cluster) # Adds cluster to label
            
        return_cluster.append(temp_cluster)
    return return_cluster

def assign_init_cluster(cluster_correlator_array,corr_labels,n_qubits,corr_limit):
    """
    Creates initial cluster by sorting for highest correlation coefficients, 
    and grouping those qubit pairs together.
    
    Corr_limit: sets limit for how low clusters should be considered. 
    """
    partitions = []
    it = 1
    while len(partitions) < n_qubits/2:
        index = np.argpartition(cluster_correlator_array, -(it))[-(it):][0]
        if not np.any(np.isin(corr_labels[index],partitions )):
            partitions.append(corr_labels[index].tolist())
            # print(f'Correlation strenght: {cluster_correlator_array[index]}')
            # print(f'Current partition: {partitions}')
        it+=1
    return partitions


def obj_func(partitions,corr_array,corr_labels, max_cluster_size,corr_limit=0, alpha=0.1,):
    """
    Objective for the cluster opitmization problem.
    """
    #print(partitions,corr_array)
    cost = 0
    # Calculate the current cluster strenght
    S = np.zeros(len(partitions))
    for i in range(len(partitions)):
        mask = np.all(np.isin(corr_labels,partitions[i]),axis=1)
        #print(mask,partitions[i])
        S[i] = np.sum(corr_array[mask])
    S_sum = np.sum(S)
    
    partition_size = np.array([len(partition) for partition in partitions])
    
    # Finding c_avg
    correlator_limit_mask = np.abs(corr_array) > corr_limit
    masked_correlators = corr_array[correlator_limit_mask]
    c_avg = np.sum(masked_correlators)/len(masked_correlators)
    
    for i in range(len(partitions)):
        if partition_size[i] > max_cluster_size:
            cost -= 1e10
        else:
            cost -=c_avg*alpha*partition_size[i]**2
    return cost + S_sum

def optimize_cluster(n_runs,init_partition,corr_array,corr_labels,max_cluster_size, corr_limit, alpha = 0 ):
    """
    Cluster optimization loop.
    """
    print('Starting optimization of premade cluster structure.')
    rng = np.random.default_rng()
    partition = copy.deepcopy(init_partition)
    for i in range(n_runs):
        print('Run:',i)
        S_pairs = copy.deepcopy(corr_labels)
        S_pairs = rng.permutation(S_pairs)
        cost_0 = obj_func(partition, corr_array, corr_labels,max_cluster_size, corr_limit, alpha)
        for pair in S_pairs:
            #print(pair,parition,np.isin(partitions,pair))
            if is_pair_in_more_than_one_cluster(pair,partition): # If pairs exist in more than one cluster:
                
                masked_partition = []  # Retrieve the partitions that include the pair
                new_partition_1 = copy.deepcopy(partition)
                temp_count = 0
                for i in range(len(partition)): # Create a partition with the pair removed and one with just the pair
                    if np.any(np.isin(partition[i],pair)):
                        masked_partition.append(partition[i])
                        new_partition_1.pop(i-temp_count)
                        temp_count+=1
                #print(f'Partion removed:{new_partition_1}')
                #print(f'Parition added {masked_partition}')
                #partion_mask = np.any(np.isin(parition,pair),axis=1) # Create mask for where what clusters include any of element in the pair

                if len(masked_partition)>2:
                    print("Pair is assigned to more than 2 clusters.")
                    print(masked_partition)
                    return 0
                # We now check 3 instances of these partitions:
                # 1) Swap 1st qubit to second
                # 2) Swat 2nd qubit to first
                # 3) Exchange qubits between the two partitions
                
                new_partition_2 = copy.deepcopy(new_partition_1)
                new_partition_3 = copy.deepcopy(new_partition_1)
                #print(pair)
                # 1) Swap 1st qubit to second
                masked_partition_1 = copy.deepcopy(masked_partition)
                #print(f'Original: {new_partition_1}, {masked_partition}, {pair}')
                if pair[0] in masked_partition_1[0]:
                    masked_partition_1[0].remove(pair[0])
                    masked_partition_1[1].append(pair[0])
                else:
                    masked_partition_1[1].remove(pair[0])
                    masked_partition_1[0].append(pair[0])
                new_partition_1.append(masked_partition_1[0])
                new_partition_1.append(masked_partition_1[1])

                # 2) Swat 2nd qubit to first
                masked_partition_2 = copy.deepcopy(masked_partition)
                if pair[1] in masked_partition_2[0]:
                    masked_partition_2[0].remove(pair[1])
                    masked_partition_2[1].append(pair[1])
                else:
                    masked_partition_2[1].remove(pair[1])
                    masked_partition_2[0].append(pair[1])
                new_partition_2.append(masked_partition_2[0])
                new_partition_2.append(masked_partition_2[1])
            
                # 3) Exchange qubits between the two partitions
                masked_partition_3 = copy.deepcopy(masked_partition)
                if pair[0] in masked_partition_3[0]:
                    masked_partition_3[0].remove(pair[0])
                    masked_partition_3[1].append(pair[0])
                    masked_partition_3[1].remove(pair[1])
                    masked_partition_3[0].append(pair[1])
                else:
                    masked_partition_3[1].remove(pair[0])
                    masked_partition_3[0].append(pair[0])
                    masked_partition_3[0].remove(pair[1])
                    masked_partition_3[1].append(pair[1])
                new_partition_3.append(masked_partition_3[0])
                new_partition_3.append(masked_partition_3[1])
                #print(new_partition_1)
                for new_partition in [new_partition_1,new_partition_2,new_partition_3]:
                    cost = obj_func(new_partition,corr_array,corr_labels,max_cluster_size,corr_limit,alpha)
                    #print(cost)
                    if cost > cost_0:
                        print(f'New partition {new_partition}')
                        print('Cost:',cost)
                        partition = copy.deepcopy(new_partition)
                        cost_0 = cost
    while [] in partition: # Remove empty paritions before sending back
        partition.remove([])           
    return partition



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
    Function takes in a state list with associated state labels, then a list of labels to trace out. Returns the traced_down state
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
    cluster_size = []
    for i in range(n_chunks):
        local_chunk = []
        while sum(local_chunk) < chunk_size: # Make sure we apropriatly sized clusters.
            integer = np.random.randint(1, np.minimum(cluster_cap + 1, chunk_size - sum(local_chunk) + 1))
            local_chunk.append(integer)
        cluster_size.extend(local_chunk)
    return cluster_size
    
# def outcomes_to_reduced_POVM(outcomes, povm_list, cluster_label_list, correlator_labels):
#     """
#     Takes in outcomes and cluster labels and returns the reduced POVM elements.
#     The reduced POVM elements are conditioned on the outcomes of the environment being traced out.

#     Parameters:
#     - outcomes (ndarray): The outcomes of the system.
#     - cluster_labels (list): A list of cluster labels.
#     - correlator_labels (list): A list of correlator labels.

#     Returns:
#     - reduced_POVM (ndarray): The reduced POVM elements.
#     """
    
    
#     # Find the relevant cluster of the correlators.
#     relevant_cluste_index = get_cluster_index_from_correlator_labels(cluster_label_list, correlator_labels)
#     relevant_clusters = cluster_label_list[relevant_cluste_index]
#     # Trace down the outcomes to the relevant qubits in each cluster separatly.
#     # Each row in the outcomes correspond the binary index of the POVM to use of the QST reconstruction. 
#     traced_out_cluster_outcomes = [trace_out(cluster,outcomes) for cluster in relevant_clusters]
#     traced_out_cluster_outcome_index = [sf.binary_to_decimal_array(traced_out_cluster_outcome) for traced_out_cluster_outcome in traced_out_cluster_outcomes]


#     # Reduce the POVM elements
#     reduced_POVM = reduce_cluster_POVMs(outcomes, cluster_labels, correlator_labels)
#     return reduced_POVM, cluster_index

# def _traced_down_outcome_to_reduced_POVM_index(traced_down_outcome, relevant_cluster):
#     """
#     Takes in a list of outcomes and the relevant cluster, and returns the reduced POVM index for that outcome in terms of the one/two reduced POVMs. . 
#     The outcomes follow the qubit order 4, 3, 2 ,1, 0 etc. Cluster structurer is not the same {5,3,1}, {4,2,0} etc.
#     The reduced POVMs will be tensored together from two labels. their internal counts are 00,01,10,11. 
    
#     Return the index of the reduced POVM element for the two differnet reduced POVM lists. 
#     """
#     # Sort reduced POVM from tensor structure. Outcome counts should go like 00,01,10,11 from the reduced POVM, first for the first qubit, then for the second qubit. 
#     if len(relevant_cluster) == 1: # Both correlators are in the same cluster. 
        
#     elif len(relevant_cluster) == 2: # Both correlators are in different clusters. 
        
#     else:
#         print(" Relevant cluster structure not met.")
#         return None
    