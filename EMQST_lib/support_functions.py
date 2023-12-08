import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
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
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,-1j],[1j,0]])
    Z=np.array([[1,0],[0,-1]])
    Bloch_vector=np.real(np.array([np.trace(X@rho),np.trace(Y@rho),np.trace(Z@rho)]))
    return np.array([[np.arccos(Bloch_vector[2]),np.arctan2(Bloch_vector[1], Bloch_vector[0])]])

def get_opposing_angles(angles):
    """"
    Takes in a set of angles and returns the angles
    anti-parallel to the vector created by input angles.
    """

    anti_angles=np.zeros_like(angles,dtype=float)
    for i in range (len(angles)):
        x = np.sin(angles[i,0]) * np.cos(angles[i,1])
        y = np.sin(angles[i,0]) * np.sin(angles[i,1])
        z = np.cos(angles[i,0])
        Bloch_vector=np.array([-x,-y,-z])
        
        anti_angles[i]=np.array([[np.arccos(Bloch_vector[2]),np.arctan2(Bloch_vector[1], Bloch_vector[0])]])
    return anti_angles




#print 'random positive semi-define matrix for today is', B
def generate_random_Hilbert_Schmidt_mixed_state(nQubit):
    """ 
    Generates random mixed state from the Hilbert-Schmidt metric.
    """
    # Generate a random complex square matrix with gaussian random numbers.
    A=np.random.normal(size=(4**nQubit)) + np.random.normal(size=(4**nQubit))*1j
    A=np.reshape(A,(2**nQubit,2**nQubit))

    # Project the random matrix onto positive semi-definite space of density matrices. 
    randomRho=A@A.conj().T/(np.trace(A@A.conj().T))
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


def power_law(x,a,b):
    return a*x**(b)


def get_cailibration_states(n_qubits):
    """
    Generates a complete set of eigenstates of the pauli matrices.
    These states are the default calibration set used to calibrate the detector tomography.
    """
    
    one_qubit_calibration_angles=np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]],
                        [[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]],
                        [[0,0]],[[np.pi,0]]])
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

if __name__=="__main__":
    main()
