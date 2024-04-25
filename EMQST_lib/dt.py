
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
import time
#sys.path.append("../")
#from support_functions import *

from EMQST_lib import support_functions as sf
from EMQST_lib import measurement_functions as mf
from EMQST_lib.qst import QST
from EMQST_lib.povm import POVM


def main():
    print("test")
    return 1


def device_tomography(n_qubits,n_shots_each,povm,calibration_states,n_cores=1,bool_exp_meaurements=False,exp_dictionary={},initial_guess_POVM=None,calibration_angles=None):
    """
    Takes in a list of  POVM objects, a set of calibration states and experimental dictionary
    and performs device tomography or POVM set tomography
    Standard format for superconducting qubit is each POVM object is a set of spin measurement on each qubit. 
    
    returns an array corrected POVM object. 
    """
    
    # If no experimental angles are provided
    if calibration_angles is None:
        calibration_angles=np.zeros((len(calibration_states),1))
    # Perform measurement over all calibration states and all POVMs
    outcome_index_matrix=np.zeros((n_shots_each))
    print(f'Collecting and sorting QDT data.')
    mesh_start=time.time()
    
    #index_list=np.arange(2**n_qubits)
    
    # Create a count function that stores the data on the form (POMV index x calib.state index)
    index_counts=np.zeros((len(povm),len(calibration_states),2**n_qubits))
    #index_count_efficient=np.zeros((len(POVM),2**n_qubits,len(calibration_states)))
    for i in range(len(povm)):
        for j in range(len(calibration_states)):
            outcome_index_matrix=mf.measurement(n_shots_each,povm[i],calibration_states[j],bool_exp_meaurements,exp_dictionary,state_angle_representation=calibration_angles[j])
            index_counts[i,j]=np.bincount(outcome_index_matrix,minlength=2**n_qubits)
            #for k in range(len(index_list)):
                #print(outcome_index_matrix)
            #    index_counts[i,k,j] = np.count_nonzero(outcome_index_matrix == index_list[k])
      
    mesh_end = time.time()
    print(f'Done collecting and sorting QDT data, total runtime {mesh_end - mesh_start}.')
    print(f'Starting POVM reconstruction.')
    if bool_exp_meaurements or initial_guess_POVM is None:
        initial_guess_POVM=povm
        
        
    #dt_start=time.time()
    #corrected_POVM=np.array([POVM_MLE(n_qubits,index_count_efficient[i],calibration_states,initial_guess_POVM[i]) for i in range(len(POVM))])
    #dt_end = time.time()
    #print(f'Runtime of POVM reconstruction {dt_end - dt_start}')

    parallel_dt_start=time.time()
    corrected_POVM = Parallel(n_jobs=n_cores)(delayed(POVM_MLE)(n_qubits,index_counts[i],calibration_states,initial_guess_POVM[i]) for i in range(len(povm)))
    parallel_dt_end = time.time()
    print(f'Runtime of parallel POVM reconstruction {parallel_dt_end - parallel_dt_start}')
    #print(f'Relative runtime impovement: {(dt_end - dt_start)/(paralleldt_end - paralleldt_start)} ')
    return corrected_POVM 


def donwconverted_device_tomography(n_qubits,downconvert_to_qubits,n_shots_each,noisy_POVM,n_cores=1,bool_exp_meaurements=False,exp_dictionary={},initial_guess_POVM=None,calibration_angles=None):
    """
    Does the same as device_tomography, but returns the two sets of POVMs, 
    one sampled from the original size, and one downconverted to a certain qubit size.
    This comparison only works with the full set of pauli-calibration states and Paili-6 POVM.
    
    returns both POVM and donwconverted POVM. 
    """
    
    calibration_states, _ = sf.get_calibration_states(n_qubits)
    povm_initial = POVM.generate_Pauli_POVM(n_qubits)
    # If no experimental angles are provided
    if calibration_angles is None:
        calibration_angles=np.zeros((len(calibration_states),1))
    # Perform measurement over all calibration states and all POVMs
    outcome_index_matrix=np.zeros((n_shots_each))
    print(f'Collecting and sorting QDT data.')
    mesh_start=time.time()
    
    
    # Create a count function that stores the data on the form (POMV index x calib.state index x outcome index)
    index_counts=np.zeros((len(noisy_POVM),len(calibration_states),2**n_qubits))
    
    for i in range(len(noisy_POVM)):
        for j in range(len(calibration_states)):
            outcome_index_matrix=mf.measurement(n_shots_each,noisy_POVM[i],calibration_states[j],bool_exp_meaurements,exp_dictionary,state_angle_representation=calibration_angles[j])
            index_counts[i,j]=np.bincount(outcome_index_matrix,minlength=2**n_qubits)

      
    mesh_end = time.time()
    print(f'Done collecting and sorting QDT data, total runtime {mesh_end - mesh_start}.')
    print(f'Starting POVM reconstruction.')
    if bool_exp_meaurements or initial_guess_POVM is None:
        initial_guess_POVM=povm_initial
    
    parallel_dt_start=time.time()
    corrected_POVM = Parallel(n_jobs=n_cores)(delayed(POVM_MLE)(n_qubits,index_counts[i],calibration_states,initial_guess_POVM[i]) for i in range(len(noisy_POVM)))
    parallel_dt_end = time.time()
    print(f'Runtime of parallel POVM reconstruction {parallel_dt_end - parallel_dt_start}')
    print(f'Starting downconverted POVM reconstruction.')
    downcoverted_index_counts = downconvert_QDT_counts(index_counts,downconvert_to_qubits)
    
    # For the moment we assume that the downconverted POVM would be the Pauli states and pauli POVM. 
    downconverted_calibration_states, _ = sf.get_calibration_states(downconvert_to_qubits)
    downconverted_inital_POVM = POVM.generate_Pauli_POVM(downconvert_to_qubits)
    
    parallel_dt_start=time.time()
    downconverted_POVM = Parallel(n_jobs=n_cores)(delayed(POVM_MLE)(downconvert_to_qubits,downcoverted_index_counts[i],downconverted_calibration_states,downconverted_inital_POVM[i]) for i in range(len(downconverted_inital_POVM)))
    parallel_dt_end = time.time()
    print(f'Runtime of parallel POVM reconstruction {parallel_dt_end - parallel_dt_start}')
    
    return corrected_POVM, downconverted_POVM
    





def POVM_MLE(n_qubits,index_counts, calibration_states,initial_guess_POVM):
    """
    Performs POVM reconstruction from measurements performed on calibration states.
    Follows prescription give by https://link.aps.org/doi/10.1103/PhysRevA.64.024102
    """
    optm='optimal'
    # Initialize POVM


    POVM_reconstruction=initial_guess_POVM.get_POVM()
    # Apply small depolarizing noise such that channel does not yield zero-values
    perturb_param=0.01
    POVM_reconstruction=np.array([perturb_param/2**n_qubits*np.eye(2**n_qubits) + (1-perturb_param)*POVM_elem for POVM_elem in POVM_reconstruction])
    iter_max = 2*10**3
    j=0
    dist=1
    #tol=10**-15
    
    while j<iter_max and dist>1e-9:    
        
        p=np.abs(np.real(np.einsum('qij,nji->nq',POVM_reconstruction,calibration_states,optimize=optm)))
        fp=index_counts/p # Whenever p=0 it will be cancelled by the elemetns in G also being zero
    
       
        G=np.einsum('nq,mq,nij,qjk,mkl->il',fp,fp,calibration_states,POVM_reconstruction,calibration_states,optimize=optm)
        
        eigV,U=sp.linalg.eig(G)
        D=np.diag(1/np.sqrt(eigV))
        L=U@D@U.conj().T

        R=np.einsum('nq,ij,njk->qik',fp,L,calibration_states,optimize=optm)
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



def experimental_QDT(n_qubits,n_calibration_shots_each,exp_dictionary,n_cores = 1,POVM_list = None,calibration_states = None,calibration_angles = None):
    
    if POVM_list == None:
        POVM_list = POVM.generate_Pauli_POVM(n_qubits)
    
    if calibration_states is None:
        calibration_states,calibration_angles=sf.get_calibration_states(n_qubits)
        
    reconstructed_POVM_list = device_tomography(n_qubits,n_calibration_shots_each,POVM_list,calibration_states,n_cores = n_cores, bool_exp_meaurements = True,exp_dictionary=exp_dictionary,initial_guess_POVM=POVM_list,calibration_angles=calibration_angles)
    return reconstructed_POVM_list


def downconvert_QDT_counts(index_counts,to_qubits,):
    """
    Converts any size index counts down to the qubit size desired. It removed qubits from the left in binary counting. 
    
    For a step by step explenation, paste the follwing code into a notebook or otherwise. 
    
    
    print("Downconversion happens in 3 steps;") 
    print(" 1) Outcomes")
    print(" 2) Calibration states")
    print(" 3) POVMs")
    # n_qubits is the target qubit size. 

    print("Step by step explaination of what happenes. Initial list created simplified 2 qubit structure." )
    arr=np.reshape(np.arange(9*12*4),(9,12,4))
    print( "Array has shape 9*12*4,  9 POVMs (xx, xy, ..) 12 calibration states, 4 outcomes (for simplicity)" )

    print(f'Select outcome of first POVM (XX) \n{arr[i]}')
    print("Sum over calibration states (They go x+x+, x+x- ,...) so every 6.")
    print(np.array([np.sum(arr[0,j::6**n_qubits],axis=0) for j in range(6**n_qubits) ]))
    print("Perform for each of these rows sum over every second outcome")
    print(np.array([np.sum(np.array([np.sum(arr[0,j::6**n_qubits],axis=0)[k::2**n_qubits] for k in range(2**n_qubits)]),axis=1) for j in range(6**n_qubits) ]))
        
    print("Now do this for each of the POVMS separatly")    
    arr_2 = np.array([np.array([np.sum(np.array([np.sum(arr[k,i::6**n_qubits],axis=0)[j::2**n_qubits] for j in range(2**n_qubits)]),axis=1) for i in range(6**n_qubits)]) for k in range(len(arr))])
    print(arr_2)

    # Downconvert POVM
    print("In the final step we need to downconver the POVMs, XX, XY, XZ, YX, so every 3rd POVM should be summed")
    arr_3 = np.array([np.sum(arr_2[i::3**n_qubits],axis=0) for i in range(3**n_qubits)])
    print(arr_3)
    """
    
    arr_1 = np.array([np.array([np.sum(np.array([np.sum(index_counts[k,i::6**to_qubits],axis=0)[j::2**to_qubits] for j in range(2**to_qubits)]),axis=1) for i in range(6**to_qubits)]) for k in range(len(index_counts))])
    arr_2 = np.array([np.sum(arr_1[i::3**to_qubits],axis=0) for i in range(3**to_qubits)])
    return arr_2

if __name__=="__main__":
    main()
