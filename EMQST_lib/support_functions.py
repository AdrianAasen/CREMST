import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import os
import uuid
from functools import reduce
from itertools import product, chain, repeat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
#from povm import *



def main():

    return 1



def get_projector_from_angles(angles): # Overloading function name
    return get_density_matrix_from_angles(angles)


def get_density_matrix_from_angles(angles):
    """
    Creates a density matrix as a tensor product of single density matrices described by the angles.
    Takes in angles on a ndarrya nQubits x 2 and returns the correspoinding density matrix
    """
    Bloch_vectors = get_Bloch_vector_from_angles(angles)
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]])
    Pauli_vector = np.array([X,Y,Z])
    rho=1
    for i in range(len(angles)):
        rho=np.kron(rho,1/2*(np.eye(2) + np.einsum("i,ijk->jk",Bloch_vectors[i],Pauli_vector)))
    return rho


def get_Bloch_vector_from_angles(angles):
    """
    Takes in a set of angles nSets x 2 and returns nSets x 3 Bloch vectors
    """
    return np.array([[np.sin(angles[i,0])*np.cos(angles[i,1]),np.sin(angles[i,0])*np.sin(angles[i,1]),np.cos(angles[i,0])] for i in range(len(angles))])

def get_angles_from_density_matrix_single_qubit(rho):
    """
    Takes in a single qubit 2x2 matrix and return the Bloch angles as 1x2 matrix.
    If state is not pure, it returns the angles to the closest pure state. 
    Does not work for the thermal state.
    """
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]])
    Bloch_vector=np.real(np.array([np.trace(X@rho),np.trace(Y@rho),np.trace(Z@rho)]))
    return np.array([[np.arccos(Bloch_vector[2]),np.arctan2(Bloch_vector[1], Bloch_vector[0])]])


def get_opposing_angles(angles):
    """
    Takes in a set of angles and returns the angles
    anti-parallel to the vector created by input angles.

    Parameters:
    angles (numpy.ndarray): Array of shape (N, 2) containing the input angles in radians.

    Returns:
    numpy.ndarray: Array of shape (N, 2) containing the anti-parallel angles in radians.
    """

    # anti_angles = np.zeros_like(angles, dtype=float)
    # for i in range(len(angles)):
    #     x = np.sin(angles[i, 0]) * np.cos(angles[i, 1])
    #     y = np.sin(angles[i, 0]) * np.sin(angles[i, 1])
    #     z = np.cos(angles[i, 0])
    #     Bloch_vector = np.array([-x, -y, -z])
        
    #     anti_angles[i] = np.array([[np.arccos(Bloch_vector[2]), np.arctan2(Bloch_vector[1], Bloch_vector[0])]])
    anti_angles = np.array([[np.pi - angle[0], (np.pi + angle[1]) % (2 * np.pi)] for angle in angles])
    return anti_angles



#print 'random positive semi-define matrix for today is', B
def generate_random_Hilbert_Schmidt_mixed_state(nQubit):
    """ 
    Generates a random mixed state from the Hilbert-Schmidt metric.

    Parameters:
    nQubit (int): The number of qubits.

    Returns:
    randomRho (ndarray): The randomly generated mixed state.

    """
    # Generate a random complex square matrix with Gaussian random numbers.
    A = np.random.normal(size=(4**nQubit)) + np.random.normal(size=(4**nQubit))*1j
    A = np.reshape(A, (2**nQubit, 2**nQubit))

    # Project the random matrix onto the positive semi-definite space of density matrices. 
    randomRho = A @ A.conj().T / (np.trace(A @ A.conj().T))
    return randomRho




def generate_random_Bures_mixed_state(nQubit):
    """
    Generates a Bures random state. See ...
    """
    Id=np.eye(2**nQubit)
    A=np.random.normal(size=(4**nQubit)) + np.random.normal(size=(4**nQubit))*1j
    A=np.reshape(A,(2**nQubit,2**nQubit))
    U=unitary_group.rvs(2**nQubit)
    rho=(Id+U)@A@A.conj().T@(Id + U.conj().T)
    return rho/np.trace(rho)

def generate_random_pure_state(nQubit):
    """
    Generates Haar random pure state.
    To generate a random pure state, take any basis state, e.g. |00...00>
    and apply a random unitary matrix. For consistency each basis state should be the same. 
    """
    baseRho=np.zeros((2**nQubit,2**nQubit),dtype=complex)
    baseRho[0,0]=1
    U=unitary_group.rvs(2**nQubit)
    return U@baseRho@U.conj().T


def generate_random_factorized_states(n_qubits,n_averages):
    """
    Creates n_averages n_qubit factorized state of Haar random single qubit states.

    Args:
        n_qubits (int): Number of qubits.
        n_averages (int): Number of states. 
    """
    state_list = np.zeros((n_averages, 2**n_qubits, 2**n_qubits),dtype=complex)
    angle_list = np.zeros((n_averages, n_qubits, 2))
    for i in range(n_averages):
        temp_state_list = np.array([generate_random_pure_state(1) for _ in range(n_qubits)])
        temp_angle_list = np.array([get_angles_from_density_matrix_single_qubit(state)[0] for state in temp_state_list])  
        temp_state=temp_state_list[0]
        for j in range(n_qubits-1):
            temp_state=np.kron(temp_state,temp_state_list[j+1])   
        angle_list[i] = temp_angle_list
        state_list[i] = temp_state
        
    return state_list, angle_list
        
        
            



def POVM_distance(M,N):
    """
    Computes the operational distance for two POVM sets.
    It is based on maximizing over all possible quantum states the "Total-Variation" distance.
    Currently only works for single qubit
    """
    d=0
    n=1000
    n_qubits=int(np.log2(len(M[0])))
    for _ in range(n):
        rho=generate_random_Hilbert_Schmidt_mixed_state(n_qubits)
        p=np.real(np.einsum('nij,ji->n',M,rho))
        q=np.real(np.einsum('nij,ji->n',N,rho))
        dTemp=1/2*np.sum(np.abs(p-q))
        if dTemp>d:
            d=dTemp
            #worst=p-q
    #print(f'Worst: {worst}')
    return d

def Pauli_expectation_value(rho):
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]])
    return np.real(np.einsum('ij,kji->k',rho,np.array([X,Y,Z])))


def power_law(x, a, b):
    """
    Calculates the power law function.

    Parameters:
    x (float): The input value.
    a (float): The coefficient.
    b (float): The exponent.

    Returns:
    float: The result of the power law function.

    """
    # Calculate the power law function
    return a * x ** b



def get_cailibration_states(n_qubits, calib = None):
    """
    Generates a complete set of calibration states. Default is the eigenstates of the pauli matrices.
    Other options are the SIC-states. 
    """
    if calib is None: # Defaults to Pauli matrices 
        calib = "Pauli"
        one_qubit_calibration_angles=np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]],
                            [[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]],
                            [[0,0]],[[np.pi,0]]])
    elif calib == "SIC":
        one_qubit_calibration_angles = np.array([[[0,0]],[[2*np.arccos(1/np.sqrt(3)),0]],
                                                 [[2*np.arccos(1/np.sqrt(3)),2*np.pi/3]],
                                                 [[2*np.arccos(1/np.sqrt(3)),4*np.pi/3]]])
    calibration_angles=np.copy(one_qubit_calibration_angles)
    one_qubit_calibration_states=np.array([get_density_matrix_from_angles(angle) for angle in calibration_angles])
    calibration_states=np.copy(one_qubit_calibration_states)

    recursion=n_qubits
    while recursion>1:
        calibration_states=np.array([np.kron(a,b) for a in calibration_states for b in one_qubit_calibration_states])
        calibration_angles=np.array([np.concatenate((angle_a,angle_b)) for angle_a in calibration_angles for angle_b in one_qubit_calibration_angles] )
        recursion-=1
    return calibration_states, calibration_angles

def one_qubit_infidelity(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the infidelity of two one qubit states according to Wikipedia.
    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :retur: infidelity
    '''
    if np.any([is_pure(rho_1), is_pure(rho_2)]):
        return 1-np.real(np.trace(rho_1@rho_2))
    elif rho_1.shape[-1]==2:
        return 1-np.real(np.trace(rho_1@rho_2) + 2*np.sqrt(np.linalg.det(rho_1)*np.linalg.det(rho_2)))
    else:
        return 1-np.real(np.trace(sqrtm(rho_1@rho_2))**2)

def is_pure(rhos: np.array, prec=1e-15):
    '''
    Checks the purity of multiple density matrices.
    :param rhos: Nxdxd array of density matrices
    :param prec: precision of the purity comparison
    :return: boolean
    '''
    # compute purity
    purity = np.trace(rhos@rhos, axis1=-2, axis2=-1, dtype=complex)

    # exclude inaccuracies caused by finte number representation of a computer
    if np.all(np.abs(np.imag(purity)) < prec) and np.all(np.abs(purity-1) < prec):
        return True
    else:
        return False
    
def purity(rhos):
    """
    Computes purity of a quantum state
    :param rhos: Nxdxd array of density matrices 
    :return: N array
    """
    purity = np.trace(rhos@rhos, axis1=-2, axis2=-1)
    return np.real(purity)


def initialize_estimation(exp_dictionary):
    # Check if restuls exist:
    check_path='results'
    path_exists=os.path.exists(check_path)
    if not path_exists:
        print("Created results dictionary.")
        os.makedirs('results')


    # Generate new dictionary for current run
    now=datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    dir_name= now_string+str(uuid.uuid4())


    data_path=f'results/{dir_name}'
    os.mkdir(data_path)

    with open(f'{data_path}/experimental_settings.npy','wb') as f:
        np.save(f,exp_dictionary)    
    return data_path



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
    # Note the last onversion as we order our qubits in reverse order, [3,2,1,0]
    qubit_to_keep_labels = np.sort(qubit_to_keep_labels)[::-1]
    qubit_to_keep_index = qubit_label_to_list_index(qubit_to_keep_labels, qubit_array.shape[-1])
    traced_down_outcomes = qubit_array[..., qubit_to_keep_index]
    return traced_down_outcomes


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


def downconvert_frequencies(subsystem_index,outcome_frequencies):
    """
    Takes in the outcome frequency measurement of the whole system and a set of qubit subsystem indices,
    and return the downconverted frequencies.
    Here the row structure matters, so outcome frequencies should be flattend to be m x n_outcomes. One can reshape back to original structure afterwards.
    Input:
        - system_index ndarray [n_subsystem_qubits]
        - outcome_frequencies ndarray n x 2**n_qubits_total
    Return:
        - The downconverted frequencies ndarray n x 2**len(subsystem_index)
    """
    # Check how many qubits are in the total system
    n_qubits_total = int(np.log2(len(outcome_frequencies[0])))
    
    # Define the axis that need to be traced out
    # Find the indices that are not present int he above, and invert the order (such that qubit 0 is axis n_qubit_total). 
    traced_indices = tuple(n_qubits_total-1 - get_traced_out_indicies(subsystem_index,n_qubits_total))
    
    # Define the subsystem shape
    reshape_tuple = (2,)*n_qubits_total

    # Sum over all cases where the subsystem indecies are the same, then reshape to be in the same shape as original list. 
    downconverted_frequencies = np.array([np.sum(measurement.reshape(reshape_tuple), axis = traced_indices).reshape(2**len(subsystem_index)) for measurement in outcome_frequencies])
    return downconverted_frequencies


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
    #print(instruction_dict)
    new_instruction = np.array([instruction_dict[element] for element in instruction])
    #print(new_instruction)
    return new_instruction


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


def create_unique_combinations(elements, n_repeat):
    """
    Generate unique combinations of elements with repetition. Where entries with all equal elements are removed.

    Args:
        elements (iterable): The elements to be combined.
        n_repeat (int): The number of times each element can be repeated in a combination.

    Returns:
        numpy.ndarray: An array of unique combinations.
    """
    comb_list = np.array(list(product(elements, repeat=n_repeat)))
   
    # Remove the duplicate elements
    n_unique_instructions = len(elements)
    indicies = np.linspace(0, len(comb_list)-1, n_unique_instructions, dtype=int)
    mask_array = np.ones(len(comb_list), dtype=bool)
    mask_array[indicies] = False
    prune_comb_list = comb_list[mask_array]
    
    return prune_comb_list


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

def binary_to_decimal(a):
    """
    Converts an arbitrary sized binary array to its decimal integer representation.

    Parameters:
    a (ndarray): The binary array to be converted.

    Returns:
    ndarrya : The decimal integer representation of the binary array. This array has one dimension less the inital array.
    """
    return a.dot(1 << np.arange(a.shape[-1] - 1, -1, -1)).copy()


if __name__=="__main__":
    main()
