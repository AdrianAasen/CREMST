
import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
import scipy as sp
import sys
#sys.path.append("../")
#from support_functions import *

from EMQST_lib import support_functions as sf
from EMQST_lib import measurement_functions as mf
from EMQST_lib.qst import QST
from EMQST_lib.povm import POVM


def main():
    print("test")
    return 1


def device_tomography(n_qubits,n_shots_each,POVM,calibration_states,bool_exp_meaurements=False,exp_dictionary={},initial_guess_POVM=None,calibration_angles=None):
    """
    Takes in a single POVM object, a set of calibration states and experimental dictionary
    and performs device tomography or POVM set tomography

    returns corrected POVM object. 
    """
    # If no experimental angles are provided
    if calibration_angles is None:
        calibration_angles=np.zeros((len(calibration_states),1))
    # Perform measurement over all calibration states
    outcome_index_matrix=np.zeros((len(calibration_states),n_shots_each))
    for i in range(len(calibration_states)):
        outcome_index_matrix[i]=mf.measurement(n_shots_each,POVM,calibration_states[i],bool_exp_meaurements,exp_dictionary,state_angle_representation=calibration_angles[i])
    if bool_exp_meaurements or initial_guess_POVM is None:
        initial_guess_POVM=POVM
    corrected_POVM=POVM_MLE(n_qubits,outcome_index_matrix,calibration_states,initial_guess_POVM)
    return corrected_POVM 




def POVM_MLE(n_qubits,outcome_index_matrix, calibration_states,initial_guess_POVM):
    """
    Performs POVM reconstruction from measurements performed on calibration states.
    Follows prescription give by https://link.aps.org/doi/10.1103/PhysRevA.64.024102
    """
    optm='optimal'
    # Initialize POVM
    # Make a list over all possible index for spin measurement
    index_list=np.arange(2**n_qubits)

    # Create a count function that stores the data on the form (POMV index x calib.state index)
    index_counts=np.zeros((2**n_qubits,len(calibration_states)))
    for i in range(len(index_list)): # Runs over the POVM index    
        for j in range (len(calibration_states)): # Runs over the calibration state index
            index_counts[i,j] = np.count_nonzero(outcome_index_matrix[j] == index_list[i])


    POVM_reconstruction=initial_guess_POVM.get_POVM()
    # Apply small depolarizing noise such that channel does not yield zero-values
    perturb_param=0.01
    POVM_reconstruction=np.array([perturb_param/2**n_qubits*np.eye(2**n_qubits) + (1-perturb_param)*POVM_elem for POVM_elem in POVM_reconstruction])
    iter_max = 2*10**3
    j=0
    dist=1

    #tol=10**-15
    
    while j<iter_max and dist>1e-9:    

        p=np.abs(np.real(np.einsum('qij,nji->qn',POVM_reconstruction,calibration_states,optimize=optm)))
        fp=index_counts/p # Whenever p=0 it will be cancelled by the elemetns in G also being zero
    
       
        G=np.einsum('qn,qm,nij,qjk,mkl->il',fp,fp,calibration_states,POVM_reconstruction,calibration_states,optimize=optm)
        
        eigV,U=sp.linalg.eig(G)
        D=np.diag(1/np.sqrt(eigV))
        L=U@D@U.conj().T

        R=np.einsum('qn,ij,njk->qik',fp,L,calibration_states,optimize=optm)
        POVM_reconstruction_old=POVM_reconstruction
        POVM_reconstruction=np.einsum('qij,qjk,qlk->qil',R,POVM_reconstruction,R.conj(),optimize=optm)
        j+=1
        if j%50==0:
            dist=POVM_convergence(POVM_reconstruction,POVM_reconstruction_old)

    print(f'\tNumber of MLE iterations: {j}, final distance {sf.POVM_distance(POVM_reconstruction,POVM_reconstruction_old)}')
    return POVM(POVM_reconstruction)

def POVM_convergence(POVM_reconstruction,POVM_reconstruction_old):
    """
    Computes the matrix norm of the difference of each element in the POVM.
    """
    return np.sum([np.linalg.norm(POVM_reconstruction-POVM_reconstruction_old)])




if __name__=="__main__":
    main()
