import numpy as np
from scipy.stats import unitary_group
from datetime import datetime
from functools import reduce
import scipy as sp
import os
import uuid
from scipy.linalg import sqrtm


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
    Inputs must be numpy arrays and not POVM objects.
    """
    # Check if the instance i POVM. If not, convert it to POVM.
    #if isinstance(M, POVM):
    #    M = M.get_POVM()
    #if isinstance(N, POVM):
    #    N = N.get_POVM()
    d=0
    n=1000
    n_qubits=int(np.log2(len(M[0])))
    d = np.zeros(n)
    for i in range(n):
        rho=generate_random_Hilbert_Schmidt_mixed_state(n_qubits)
        p=np.real(np.einsum('nij,ji->n',M,rho))
        q=np.real(np.einsum('nij,ji->n',N,rho))
        d[i]=1/2*np.sum(np.abs(p-q))
        
    return np.max(d)

    

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



def get_calibration_states(n_qubits, calib = None):
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
    elif calib == "Comp":
        one_qubit_calibration_angles = np.array([[[0,0]],[[np.pi,0]]])
        
    calibration_angles=np.copy(one_qubit_calibration_angles)
    one_qubit_calibration_states=np.array([get_density_matrix_from_angles(angle) for angle in calibration_angles])
    calibration_states=np.copy(one_qubit_calibration_states)
        
    recursion=n_qubits
    while recursion>1:
        calibration_states=np.array([np.kron(a,b) for a in calibration_states for b in one_qubit_calibration_states])
        calibration_angles=np.array([np.concatenate((angle_a,angle_b)) for angle_a in calibration_angles for angle_b in one_qubit_calibration_angles] )
        recursion-=1
    return calibration_states, calibration_angles

def qubit_infidelity(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the infidelity of two qubit states according to Wikipedia.
    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :return: infidelity
    '''
    if np.any([is_pure(rho_1), is_pure(rho_2)]): # Pure states
        return 1-np.real(np.trace(rho_1@rho_2))
    elif rho_1.shape[-1]==2: # One qubit states
        return 1-np.real(np.trace(rho_1@rho_2) + 2*np.sqrt(np.linalg.det(rho_1)*np.linalg.det(rho_2)))
    else: # General infidelity for higher dimensions and non-pure states
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



def binary_to_decimal_array(a):
    """
    Converts an arbitrary sized binary array to its decimal integer representation.

    Parameters:
    a (ndarray): The binary array to be converted.

    Returns:
    ndarray : The decimal integer representation of the binary array. This array has one dimension less the initial array.
    """
    return a.dot(1 << np.arange(a.shape[-1] - 1, -1, -1)).copy()


def decimal_to_binary_array(decimal_array, max_length=None):
    """
    Takes in an array of decimal numbers and converts it into an array of binary numbers.

    Parameters:
    - decimal_array (array-like): An array of decimal numbers.
    - max_length (int, optional): The maximum length of the binary representation. If not provided, it is calculated based on the maximum decimal value in the array.

    Returns:
    - binary_array (ndarray): An array of binary numbers.
    """
    
    if max_length is None:
        max_len = np.max(decimal_array)
        if np.max(max_len) == 0:
            max_length = 1
        else: 
            max_length = int(np.ceil(np.log2(max_len)))
    
    # Create binary array for each integer
    binary_array = (((decimal_array[:, None] & (1 << np.arange(max_length)[::-1]))) > 0).astype(int)
    
    return binary_array
def partial_trace(rho, qubit = 0):
    """
    Takes the partial trace of a list of 2 qubit density matrices. 
    """

    rho = np.reshape(rho,(2,2,2,2))
    if qubit == 0:
        traced_down_rho = np.einsum('jiki->jk',rho)
    else:
        traced_down_rho = np.einsum('ijik->jk',rho)

    return traced_down_rho

def ac_POVM_distance(M,N):
    """
    Computes the average case distance for two POVM sets. Follows the definition in https://doi.org/10.22331/q-2023-09-11-1106 (eq. 3)
    """
    dim = len(M[0])
    op = M-N
    return 1/(2 * dim) * np.sum(np.array([np.sqrt(np.linalg.norm(element)**2 + np.abs(np.trace(element))**2) for element in op] ))


def generate_GHZ(n_qubits):
    """
    Generates the GHZ state for n_qubits.
    """
    GHZ = np.zeros((2**n_qubits, 2**n_qubits))
    GHZ[0,0] = 1/2
    GHZ[-1,-1] = 1/2
    return GHZ
    


def is_state_physical(rho):
    """
    Checks if the density matrix is physical.
    """
    if np.any(np.linalg.eigvals(rho) < 0):
        print(f'State has negative egienvalues: {np.linalg.eigvals(rho)}')
        return False
    if not np.isclose(np.trace(rho), 1):
        print(f'State is not normalized: {np.real(np.trace(rho))}')
        return False
    
    return True    
    

def check_positive_eigenvalues(rho):
    """
    Checks if the density matrix has positive eigenvalues.
    """
    return np.all(np.linalg.eigvals(rho) >= 0)


def generate_data_folder(base_path, subfolder_names = None):
    """
    Creates a random folder name within base_path and returns the path to the new folder for storage of results.
    """
    # Check path
    path_exists=os.path.exists(base_path)
    if not path_exists:
        print(f"Created {base_path} dictionary.")
        os.makedirs(base_path)

    now=datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    dir_name= now_string+str(uuid.uuid4())
    data_path = f'{base_path}/{dir_name}'
    os.mkdir(data_path)
    
    # Creates subfolders if specified
    if subfolder_names is not None:
        for subfolder in subfolder_names:
            os.mkdir(f'{data_path}/{subfolder}')
            
    return data_path


def rot_about_collective_X(angle,n_qubits):
    X = np.array([[0,1],[1,0]],dtype = complex)
    collective_axis = reduce(np.kron,[X]*(n_qubits))
    return sp.linalg.expm(-1j/2 * angle * collective_axis)


def generate_two_qubit_Pauli_string(use_identity: bool = False):
    """
    Creates all possible two-qubit Pauli strings madefrom the Pauli operators X, Y and Z. 
    If use_identity is set to True, the identity operator is included in the list of operators.
    """
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    id = np.eye(2, dtype = complex)
    if use_identity:
        op_array = [X,Y,Z,id]
    else:
        op_array = [X,Y,Z]
    op_string_array = [np.kron(a, b) for a in op_array for b in op_array]
    return op_string_array


def depolarizing_channel(rho, strength):
    """
    Applies a depolarizing channel to the density matrix rho with given strength.
    """
    if strength==0: # If no strenght then return the input state
        return rho
    # Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    # Kraus operators
    K0 = np.sqrt(1 - strength) * I
    K1 = np.sqrt(strength / 3) * X
    K2 = np.sqrt(strength / 3) * Y
    K3 = np.sqrt(strength / 3) * Z
    kraus_ops = [K0, K1, K2, K3]

    # Apply channel
    rho_out = sum(K @ rho @ K.conj().T for K in kraus_ops)
    return rho_out


if __name__=="__main__":
    main()


