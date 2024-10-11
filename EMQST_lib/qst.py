import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
import scipy
import sys
#sys.path.append("../")

import EMQST_lib.support_functions as sf
from EMQST_lib import measurement_functions as mf
from EMQST_lib.povm import POVM
#from EMQST_lib import povm

class QST():
    """
    This class will handle the QST estimator and maintain all relevant parameters.
    Currently it only performs BME, but could easily be extended to adaptiv BME and MLE
    """
    __MH_steps=50


    def __init__(self,POVM_list,true_state_list,n_shots_each_POVM,n_qubits,
                 bool_exp_measurements,exp_dictionary,n_cores=4,
                 noise_corrected_POVM_list=np.array([]),
                 true_state_angles_list=None):
        """
        Initalization of estimator.
        POVM_list:                  The measurement set to be performed (or set of measurements) array of POVM class
        true_state_list:            A list of all states that are to be measured. 
        n_shots_each_POVM:          Number of measurments to be performed. If there are multiple POVMs each POVM will have n_shots_each shots perfomed.
        n_qubtis:                   Number of qubits in the state (1 or 2 for now)
        experimental_dictionary:    Contains all relevant paramters for experimenal runs
        n_cores:                    Tells us how many cores to use during resampling
        """
        self.POVM_list=POVM_list
        self.noise_corrected_POVM_list=noise_corrected_POVM_list
        # If no true states are given, set empty list. 
        if true_state_list is None: # Insert a un-normalized thermal state. 
            self.true_state_list = np.array([np.eye(2**n_qubits)])
        else:
            self.true_state_list = true_state_list
            
        if true_state_list is None:
            self.n_averages = 1
        else:
            self.n_averages=len(true_state_list)
        
        # Checks if angle representation has been given
        if true_state_angles_list is None:
            self.true_state_angles_list=np.array([None]*self.n_averages)
        else:
            self.true_state_angles_list=true_state_angles_list
            
        self.n_shots_each_POVM=n_shots_each_POVM
        self.n_shots_total=n_shots_each_POVM*len(POVM_list)
        self.exp_dictionary=exp_dictionary
        self.n_qubits=n_qubits
        
        self.n_cores=n_cores
        self.bool_exp_measurement=bool_exp_measurements
        full_operator_list=np.array([a.get_POVM() for a in self.POVM_list])
        full_operator_list=np.reshape(full_operator_list,(-1,2**self.n_qubits,2**self.n_qubits))
        self.full_operator_list=full_operator_list

        
        # BME parameters
        if n_qubits==1:
            self.n_bank=100
        elif n_qubits==2:
            self.n_bank=500
            self.__MH_steps=75
        elif n_qubits>2:
            self.n_bank=0
        
        # Initalize empty containser that will carry mesaruement results.
        self.outcome_index=np.zeros((self.n_averages,self.n_shots_total))
        self.infidelity=np.zeros((self.n_averages,self.n_shots_total))
        self.uncertainty=np.zeros((self.n_averages,self.n_shots_total))
        self.rho_estimate=np.zeros((self.n_averages,2**self.n_qubits,2**self.n_qubits),dtype=complex)

    def save_QST_settings(self,path,noise_mode=0):
        QST_settings={
        "n_QST_shots_each": self.n_shots_each_POVM,
        "list_of_true_states": self.true_state_list,
        "n_qubits": self.n_qubits,
        "MH_steps": self.__MH_steps,
        "bank_size": self.n_bank,
        "n_cores": self.n_cores,
        "bool_exp_measurements": self.bool_exp_measurement,
        "POVM_list": self.POVM_list,
        "noise_corrected_POVM_list":self.noise_corrected_POVM_list,
        "n_averages": self.n_averages,
        "noise_mode": noise_mode,
        "outcome_index": self.outcome_index,  
        "uncertainty": self.uncertainty      
    }
        
        with open(f'{path}/QST_settings.npy','wb') as f:
            np.save(f,QST_settings)


    @classmethod
    def load_data(cls,base_path):
        """
        Loads data from a QST_settings file.
        Base_path should indicate folder the QST_settings.npy file lies in.
        Returns an QST object initialized with data and setting from previous run. 
        Results such as infidelity and rho_estm will be found in QST_results.npy.
        """
        
        with open(f'{base_path}QST_settings.npy','rb') as f:
            qst_dict=np.load(f,allow_pickle=True).item()

        qst=cls(qst_dict["POVM_list"],qst_dict["list_of_true_states"],qst_dict["n_QST_shots_each"],qst_dict["n_qubits"],qst_dict["bool_exp_measurements"],{},qst_dict["n_cores"],qst_dict["noise_corrected_POVM_list"])
        qst.outcome_index=qst_dict["outcome_index"]
        print(f'Loaded QST settings from {base_path}')
        return qst
    
    @classmethod
    def quick_setup(cls,n_qubits=1):
        """
        Quick setup of QST class, defaults to 1 qubit MLE with Pauli-6
        and selects 5 Haar-random true states, which should be measured 6*10**4 times.
        """
        #n_qubits=2
        n_averages=5
        n_QST_shots=10**4
        bool_exp_measurements=False
        exp_dictionary={}
        n_cores=4
        list_of_true_states = np.array([sf.generate_random_pure_state(n_qubits) for _ in range(n_averages)])
        POVM_list=POVM.generate_Pauli_POVM(n_qubits)
        return QST(POVM_list,list_of_true_states,n_QST_shots,n_qubits,bool_exp_measurements,exp_dictionary,n_cores=n_cores)
    
    
    @classmethod    
    def setup_experimental_QST(cls, n_qubits, n_QST_shots, exp_dictionary, measurement_POVM_list = None):
        """
        Sets up an experimental QST object that calls costum measurement functions.
        The default POVM can be overridden during MLE. 
        
        return a QST object set up for experimental runs. 
        """
        # Default measurement is Pauli-6
        if measurement_POVM_list == None:
            measurement_POVM_list = POVM.generate_Pauli_POVM(n_qubits)
            
        return QST(measurement_POVM_list,None,n_QST_shots,n_qubits,True,exp_dictionary)
    
    def print_status(self):
        print(f'Use experimental measurements: {self.bool_exp_measurement}')
        print(f'Number of qubits: {self.n_qubits}')
        print(f'Number of averages: {self.n_averages}')
        print(f'Number of shots used for QST: {self.n_shots_total}')


    def get_infidelity(self):
        return np.copy(self.infidelity)
    
    def get_rho_estm(self):
        return np.copy(self.rho_estimate)
    
    def get_rho_true(self):
        return np.copy(self.true_state_list)
    
    def get_outcomes(self):
        return np.copy(self.outcome_index)
    
    def set_outcomes(self,outcome_index):
        self.outcome_index = np.copy(outcome_index)
        
    def get_uncertainty(self):
        return np.copy(self.uncertainty)

    def generate_data(self, override_POVM_list = None, custom_measurement_function = None):
        """
        Simulates sampling from the states defined. Default POVM is the selected measurement.
        Measurement can be overridden by e.g. the corrected POVM from detector. 
        """

        # Overrides POVM if promted
        if override_POVM_list is None:
            measured_POVM_list=self.POVM_list
        else:
            measured_POVM_list=override_POVM_list
        

        n_POVMs=len(self.POVM_list)
        n_shots_each_POVM=self.n_shots_each_POVM

        for i in range(self.n_averages): # We run the estimator over all averages required.
            
            # Generate data
            temp_outcomes=np.zeros((n_POVMs,n_shots_each_POVM))
            index_iterator=0

            for j in range(n_POVMs):
                temp_outcomes[j]=mf.measurement(n_shots_each_POVM, measured_POVM_list[j],self.true_state_list[i], self.bool_exp_measurement, self.exp_dictionary,state_angle_representation=self.true_state_angles_list[i], custom_measurement_function = custom_measurement_function) + index_iterator
                index_iterator+=len(self.POVM_list[j].get_POVM())
            
            # Reshape lists
            temp_outcomes=np.reshape(temp_outcomes,-1)
            self.outcome_index[i]=np.copy(temp_outcomes)

    
    def perform_MLE(self, override_POVM_list=None):
        """
        Runs core loop of MLE.
        """
        # Select POVM to use for state reconstruction 
        if override_POVM_list is None:
            full_operator_list=self.full_operator_list  
        else:
            full_operator_list=np.array([a.get_POVM() for a in override_POVM_list])
            full_operator_list=np.reshape(full_operator_list,(-1,2**self.n_qubits,2**self.n_qubits))
            
        outcome_index=self.outcome_index.astype(int)
        for i in range(self.n_averages):
            rho_estm=QST.iterativeMLE(full_operator_list,outcome_index[i])
            self.rho_estimate[i]=rho_estm.copy() # Is copy nessecary here?
            self.infidelity[i,-1]= 1 - np.real(np.einsum('ij,ji->',rho_estm,self.true_state_list[i]))
        
        
        
        
    def iterativeMLE(full_operator_list: np.array, outcome_index: np.array ):
        '''
        Estimates state according to iterative MLE.
        :param full_operator_list: full list of POVM elemnts
        :outcome_index: list of all outcome_indices such that
             full_operator_list[index] = POVM element
        :return: dxd array of iterative MLE estimator
        '''


        unique_index, index_counts=np.unique(outcome_index,return_counts=True)
        OP_list=full_operator_list[unique_index]
        return QST.iterative_MLE_index(index_counts, OP_list)

    def iterative_MLE_index(index_counts, OP_list):
        dim = OP_list.shape[-1]

        
        dist     = float(1)

        rho_1 = np.eye(dim)/dim
        rho_2 = np.eye(dim)/dim
        j = 0
        iter_max = 500
        while j<iter_max and dist>1e-14:
            p      = np.einsum('ik,nki->n', rho_1, OP_list)
            R      = np.einsum('n,n,nij->ij', index_counts, 1/p, OP_list)
            update = R@rho_1@R
            rho_1  = update/np.trace(update)

            if j>=40 and j%20==0:
                dist  = sf.qubit_infidelity(rho_1, rho_2)
            rho_2 = rho_1

            j += 1

        return rho_1
        

    def perform_BME(self,override_POVM_list=None, compute_infidelity_uncertainty=False):
        """
        Runs the core loop of BME.
        """
        
        # Checks if BME is performed for more than 2 qubits
        if self.n_qubits>2:
            print(f'BME does not support more than 2 qubits, current is {self.n_qubits}.')
            print(f'Returning the thermal state.')
            return 1/(2**self.n_qubits)*np.eye(2**self.n_qubits)
        
        # Select POVM to use for state reconstruction 
        if override_POVM_list == None:
            full_operator_list=self.full_operator_list
        else:
            full_operator_list=np.array([a.get_POVM() for a in override_POVM_list])
            full_operator_list=np.reshape(full_operator_list,(-1,2**self.n_qubits,2**self.n_qubits))
       
            


        outcome_index=self.outcome_index.astype(int)
        for j in range(self.n_averages):
            #print(f'Started QST run {j}/{self.n_averages}')
            # Initalize bank and weights
            rho_bank=generate_bank_particles(self.n_bank,self.n_qubits)
            weights=np.full(self.n_bank, 1/self.n_bank)
            S_treshold=0.1*self.n_bank  

            

            # Shuffle outcomes such that BME converges as expected
            rng = np.random.default_rng()
            rng.shuffle(outcome_index[j])

            # Start BME loop
            for k in range(len(outcome_index[j])):
                
                weights=QST.weight_update(weights,rho_bank,full_operator_list[outcome_index[j,k]])
                S_effective=1/np.dot(weights,weights)

                #If effective sample size of posterior distribution is too low we resample
                if (S_effective<S_treshold):
                   rho_bank, weights=QST.resampling(self.n_qubits,rho_bank,weights,outcome_index[j,:k],full_operator_list,self.n_cores,self.__MH_steps)
                self.infidelity[j,k]=1-np.real(np.einsum('ij,kji,k->',self.true_state_list[j],rho_bank,weights))
                
                
                
                # Compute averag bures distance of distribution
                #if k%1000==0:
                #    self.uncertainty[j,k]=average_Bures(rho_bank,weights,self.n_qubits)
            # Compute the final uncertainty
            if compute_infidelity_uncertainty:
                self.uncertainty[j,-1] = QST.infidelity_uncertainty(rho_bank,weights)
            self.rho_estimate[j]=np.array(np.einsum('ijk,i->jk',rho_bank,weights))
            print(f'Completed run {j+1}/{self.n_averages}. Final infidelity: {self.infidelity[j,-1]}.')
    
    
    def infidelity_uncertainty(rho_bank,weights):
        """
        Computes the Bayesian uncertatiny of the likelihood function in terms for the average infidelity between the Bayesian mean state and the weighted bank particles.
        """
        rho_mean = np.einsum('ijk,i->jk',rho_bank,weights)
        bank_infidelity = np.array([sf.qubit_infidelity(rho_mean,particle_rho) for particle_rho in rho_bank])
        infidelity_uncertainty = np.einsum('i,i->',bank_infidelity,weights)
        return infidelity_uncertainty
    
    def weight_update(weights,rho_bank,measurement_operator):
        conditional_probability=np.einsum('kj,ijk->i',measurement_operator,rho_bank,optimize=False) # Optimizing this einsum is not worth it!
        return np.real(conditional_probability*weights/np.dot(conditional_probability,weights))
    

    def resampling(n_qubits,rho_bank,weights,outcome_index,full_operator_list,n_cores,MH_steps):
        # Start by sampling bank particles by their relative weight
        # Calculate the cumulative probabilites for easy sampling
        cumulative_sum=np.cumsum(weights)
        # Calculate the kick strenght based on the bures variance of the distribution
        likelihood_variance=np.sqrt(np.real(average_Bures(rho_bank,weights,n_qubits,n_cores)))
        if n_qubits==2:
            likelihood_variance*=0.1
        elif n_qubits==1:
            likelihood_variance*=0.4
        #print(likelihood_variance)
        index_values,index_counts=np.unique(outcome_index,return_counts=True)
        random_seed=np.random.randint(1e8,size=(int(len(rho_bank))))
        new_rho_bank,n_accepted_iterations=zip(*Parallel(n_jobs=n_cores)(delayed(QST.resampling_bank)(n_qubits,rho_bank,cumulative_sum,full_operator_list,index_counts,index_values,likelihood_variance,MH_steps,rng) for rng in random_seed ))
        new_rho_bank=np.asarray(new_rho_bank)
        n_accepted_iterations=np.sum(n_accepted_iterations)
        
        #print(f'    Acceptance rate:{n_accepted_iterations/(len(rho_bank)*MH_steps)}')
        new_weights=np.full(len(weights),1/len(weights))
        return new_rho_bank, new_weights
        
    
    
    def resampling_bank(n_qubits,rho_bank,cumulative_sum,full_operator_list,index_counts,index_values,likelihood_variance,MH_steps,rng_seed):
        """
        The resampling scheme is following what is outlined in appendix C of https://link.aps.org/doi/10.1103/PhysRevA.93.012103
        """
        np.random.seed(rng_seed)


        # Randomly pick a bank particle that new one is based off
        r=np.random.random()
        base_rho=rho_bank[np.argmax(cumulative_sum>r)]
        base_likelihood=logLikelihood(base_rho,full_operator_list,index_counts,index_values)
        # Purification
        chol=np.linalg.cholesky(base_rho)
        purified_state=chol.flatten()
        n_accepted_iterations=0
        #print(f'MH:{MH_steps}')
        for _ in range(MH_steps):
            
            # Draw random pertubation size
            d=np.random.normal(0,likelihood_variance)
            #print(1-d**2/2)
            a=1-d**2/2
            b=np.sqrt(1-a**2)
            # Compute pertubation matrix
            g=np.random.normal(0,1,(len(purified_state))) + np.random.normal(0,1,(len(purified_state)))*1j
            perturbed_state=a*purified_state + b*(g-purified_state*(purified_state.conj()@g))/np.linalg.norm(g-purified_state*(purified_state.conj()@g))

            # Take partial trace and compute likelihood
            #purified_density=np.outer(perturbed_state,perturbed_state.conj())
            # Reshaping groups qubits on the form rho_{a_1,b_1,a_2,b_2} where a is the physical qubit and b is ancilla
            # This generalizes to n qubit case by rho_{a_1,a_2,b_1,b_2,â_1,â_2,b^_1,b^_2}
            #                                    =rho_{a',b',â',b^'} where primed now counts to 4 insted of binary.
            #purified_density=np.reshape(purified_density,(2*n_qubits,2*n_qubits,2*n_qubits,2*n_qubits)) 
            #perturbed_rho=np.trace(purified_density,axis1=1,axis2=3)
            

            reshaped_state = np.reshape(perturbed_state, (2*n_qubits, 2*n_qubits))
            perturbed_rho = reshaped_state @ reshaped_state.conj().T
            #print(np.allclose(perturbed_rho2,perturbed_rho))
            temp_likelihood=logLikelihood(perturbed_rho,full_operator_list,index_counts,index_values)
            # Check if the MH step should be accepted
            ratio=np.exp(temp_likelihood - base_likelihood)
            r=np.random.random()
            if (r<ratio): # If step is accepted replace the purified state with the perturbed state. Overwrite the comparative likelihood function
                purified_state=perturbed_state
                base_likelihood=temp_likelihood
                n_accepted_iterations+=1

        # Add the last accepted state to set of new bank particles. 
        purified_density=np.outer(perturbed_state,perturbed_state.conj())
        purified_density=np.reshape(purified_density,(2*n_qubits,2*n_qubits,2*n_qubits,2*n_qubits))
        perturbed_rho=np.trace(purified_density,axis1=1,axis2=3)        
        if n_accepted_iterations/MH_steps<0.2:
            print(f'Low acceptance! Accepted ratio: {n_accepted_iterations/MH_steps}.')
        return perturbed_rho, n_accepted_iterations
    




def average_Bures(rho_bank,weights,n_qubits,n_cores): 
    """
    Computes the average Bures distance of the current bank. 
    Current 2+ qubit impementation uses the Qutip fidelity function. 
    """
    mean_state=np.array(np.einsum('ijk,i->jk',rho_bank,weights))
    b=0
    # Checks wether we are one or two qubits
    if n_qubits==1:
        for i in range(len(rho_bank)):
            infid=sf.qubit_infidelity(rho_bank[i],mean_state)
            b+=2*(infid)*weights[i]
    else: # 2 Qubit case is much slower, NEW: parallelized fidelity computation. 
        fid=Parallel(n_jobs=n_cores)(delayed(parallel_Bures)(rho,mean_state) for rho in rho_bank)
        b=np.einsum('i,i->',2*(1-np.real(fid)),weights)
    return b

def parallel_Bures(rho,mean_state):
    fid=qt.fidelity(qt.Qobj(rho),qt.Qobj(mean_state))
    return fid


def logLikelihood(rho,full_operator_list,index_counts,index_values): 
    """
    Returns the loglikelihood of rho given the statevector outcomes. 
    This could be made easier to compute if there were only measurment modes uses. 
    """
    return np.dot(np.log(np.real(np.einsum('ij,kji->k',rho,full_operator_list)))[index_values],index_counts)


def generate_bank_particles(nBankParticles,nQubits,boolBuresPrior=False):
    """
    Returns a set of bank particles of given bank size and number of qubits. 
    Can generate mixed states from either HS random or Bures random states.
    """
    rhoBank=np.zeros([nBankParticles,2**nQubits,2**nQubits],dtype=complex)    
    if boolBuresPrior:
         for i in range(nBankParticles):
            rhoBank[i]=sf.generate_random_Bures_mixed_state(nQubits)
    else:
        for i in range(nBankParticles):
            rhoBank[i]=sf.generate_random_Hilbert_Schmidt_mixed_state(nQubits)
    return rhoBank