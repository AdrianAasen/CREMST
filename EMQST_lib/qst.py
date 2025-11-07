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
import copy
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
            self.n_bank=200
            self._MH_steps=50
        elif n_qubits==2:
            self.n_bank=2000
            self._MH_steps=200
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
        "MH_steps": self._MH_steps,
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
       

    def perform_adaptive_BME(self,override_POVM_list = None, compute_uncertainty = False, depolarizing_strength = 0, adaptive_burnin_steps = None):
        """
        Special purpose adaptive BME. Iteratively updates the measured POVM based on an optimization problem.
        As opposed to normal BME, measurements must be performed live, and if measurements were performed already they will be deleted.
        For adaptive noise, only depolarizing strength is supported, and it is applied to the POVM elements.
        If depolarizing_strength is set to 0, no depolarization is applied.""" 
        # These imports are only needed for adaptive BME, and requires to be run in a linux environment. 
        import jax.numpy as jnp
        from jax import grad, jit, vmap
        from jax import random
        import optax
        import jax
        from EMQST_lib import adaptive_functions as ad
        if self.n_qubits==1:
            total_grad_steps = 100
        else:
            total_grad_steps = 1000 
        if adaptive_burnin_steps is None:
            adaptive_burnin_steps = 0

       
       
        # Start by defining optimizer
        if self.n_qubits == 1: # Select two different schedules for 1 and 2 qubits
            #opt_Schedule = optax.cosine_decay_schedule(0.02, decay_steps=total_grad_steps, alpha=0.95)
            opt_Schedule = optax.linear_onecycle_schedule(peak_value=0.01,transition_steps=total_grad_steps,pct_start=0.25,pct_final=0.7,div_factor=20,final_div_factor=10)
        elif self.n_qubits == 2:
            opt_Schedule = optax.linear_onecycle_schedule(peak_value=0.02,transition_steps=total_grad_steps,pct_start=0.25,pct_final=0.7,div_factor=20,final_div_factor=10)
            #opt_Schedule = optax.linear_schedule(init_value=0.05, end_value=0.001, transition_steps=total_grad_steps)
        else:
            print('Adaptive BME not supported for more than 2 qubits.')
            return None
        
        def optim(learning_rate):
            return optax.adam(learning_rate=learning_rate)# optax.sgd(learning_rate=learning_rate, momentum=0.9)  # Simpler than ADAM

        optimizer = optim(opt_Schedule)
   
        @jax.jit
        def jax_step(angles, opt_state, rho_bank, weights, best_guess):
            loss_value, grads = jax.value_and_grad(ad.adaptive_cost_function)(angles, rho_bank, weights, best_guess, self.n_qubits)
            updates, opt_state = optimizer.update(grads, opt_state, angles)
            angles = optax.apply_updates(angles, updates)
            return angles, opt_state, loss_value

        
        # Continue with normal BME setup
        # Checks if BME is performed for more than 2 qubits
        if self.n_qubits>2:
            print(f'BME does not support more than 2 qubits, current is {self.n_qubits}.')
            print(f'Returning the thermal state.')
            return 1/(2**self.n_qubits)*np.eye(2**self.n_qubits)
        




        # Define noisy POVM to be used for sampling
        # Create single (potentially depolarized) Pauli-6 POVM
        temp_POVM = POVM.generate_Pauli_POVM(self.n_qubits) # Starts out with Pauli-6 burnin
        decompiled_array = np.array([temp_POVM[i].get_POVM() for i in range(len(temp_POVM))])
        pauli_6_array = 1/3**(self.n_qubits)*np.array(decompiled_array.reshape(-1,decompiled_array.shape[-2],decompiled_array.shape[-1]))
        
        depolarized_pauli_6_array = np.array([sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in pauli_6_array])
        
        # Generate the proper pauli-6 POVM
        #pauli_6 = POVM(pauli_6_array)
        depolarized_pauli_6 = POVM(depolarized_pauli_6_array)
        full_operator_list = [] # The full operator list will also contain adaptive POVMs and no direct way to separate them. The outcomes are collected and assigned correct outcome corresponding to the measureemnt operator.
        outcome_index = self.outcome_index.astype(int)
        


        for j in range(self.n_averages):
            full_operator_list.append(depolarized_pauli_6_array)
            # Apply depolarizing channel to intial POVM elements.
            outcome_offset = 0
            adaptive_threshold = adaptive_burnin_steps
            adaptive_it = 0
            current_POVM = copy.deepcopy(depolarized_pauli_6)

            # Initalize bank and weights
            rho_bank=generate_bank_particles(self.n_bank,self.n_qubits)
            weights=np.full(self.n_bank, 1/self.n_bank)
            if self.n_qubits==1:
                S_treshold=0.1*self.n_bank
            elif self.n_qubits==2:
                S_treshold=0.05*self.n_bank  
            
            # Adaptive measurements strategy is based on finding the overall best unitary rotation that has the maximal information gain.
            for shot_index in range(self.n_shots_total):
                
                # Update POVM if adaptive criteria is met.
                if shot_index > adaptive_threshold:
                    adaptive_it += 1
                    prev_loss = 0
                    # Usest the current mean state as the state to adapt around.
                    current_mean_state = np.array(np.einsum('i...,i', rho_bank, weights))
                    inital_angles = ad.rho_to_angles(current_mean_state, self.n_qubits)
                    # Initialize optimizer with random angles, create jax-friendly objects.
                    opt_state = optimizer.init(inital_angles)
                    best_angles = inital_angles
                    angles = inital_angles
                    successful_gradient_step_it = 0

                    jbank=jnp.array(rho_bank)
                    jweight=jnp.array(weights)
                    jmean_state=jnp.array(current_mean_state)
                    for _ in range(total_grad_steps):
                        angles, opt_state, current_loss = jax_step(angles, opt_state, jbank, jweight, jmean_state)
                        if current_loss<prev_loss:
                            #print('Updated')
                            best_angles = angles
                            prev_loss = current_loss
                            successful_gradient_step_it += 1
                    #print(f'Relative adaptive success: {successful_gradient_step_it/total_grad_steps    }')
                    angles = best_angles
                    projective_vector = sf.angles_to_state_vector(angles, self.n_qubits)
                    out = np.einsum('ij,ik->ijk',projective_vector,projective_vector.conj())
                    #print(f'Size: {out.shape}, norm {np.sum(out,axis=0)}')
                    # Apply depolarizing channel to new POVM elements
                    depolarized_out = np.array([sf.depolarizing_channel(element, depolarizing_strength) for element in out])
                    
                    #current_POVM = POVM(out)
                    current_POVM = POVM(depolarized_out)    
                    # Offset needs to be computed before new measurement setting is used. 
                    outcome_offset = len(full_operator_list[j]) 
                    if len(depolarized_out)!=2**self.n_qubits:
                        print('Warning! Adaptive POVM does not have correct number of outcomes!')
                        print(f'Number of outcomes: {len(depolarized_out)}, expected {2**self.n_qubits}.')
                        print(depolarized_out)
                    full_operator_list[j] = np.append(full_operator_list[j], depolarized_out , axis=0)
                    adaptive_threshold += np.maximum(1,int(shot_index/100))

                # outcome_offset is used to be able to mix multiple different POVMs with the same outcome_index.
                individual_outcome = mf.simulated_measurement(1, current_POVM, self.true_state_list[j])
                outcome_index[j,shot_index] = individual_outcome + outcome_offset
                if outcome_index[j,shot_index]>len(full_operator_list[j])-1:
                    print('Warning! outcome index is larger than possible outcome!')
                    print(f'Outcome index: {outcome_index[j,shot_index]}, max possible {len(full_operator_list[j])}.')
                    print(f'Current POVM has {len(current_POVM.get_POVM())} elements.')
                    print(current_POVM.get_POVM())
                    print(f'Last computed offset: {outcome_offset}.')
                    print(f'Individual outcome: {individual_outcome}.')
                    print(f'Total POVM size: {len(full_operator_list[j])}.')
                    print(f'Last two povm elements are: {full_operator_list[j][-2:]}.')
                weights = QST.weight_update(weights,rho_bank,full_operator_list[j][outcome_index[j,shot_index]])
                S_effective=1/np.dot(weights,weights)
                # If effective sample size of posterior distribution is too low we resample
                if (S_effective<S_treshold):
                    #print("Resampling")
                    rho_bank, weights=QST.resampling(self.n_qubits,rho_bank,weights,outcome_index[j,:shot_index],full_operator_list[j],self.n_cores,self._MH_steps)
                self.infidelity[j,shot_index]=1-np.real(np.einsum('ij,kji,k->',self.true_state_list[j],rho_bank,weights))
                # Compute average bures distance of distribution
                if compute_uncertainty and shot_index%100==0:
                    self.uncertainty[j,shot_index] = 2*self.infidelity[j,shot_index]/average_Bures(rho_bank,weights,self.n_qubits,self.n_cores)
            self.rho_estimate[j]=np.array(np.einsum('ijk,i->jk',rho_bank,weights))
            print(f'Completed run {j+1}/{self.n_averages}. Final infidelity: {self.infidelity[j,-1]}.')
                    




                
    def perform_BME(self, override_POVM_list = None, compute_uncertainty = False):
        """
        Runs the core loop of BME.
        compute_uncertainy: np.array that contains the sample numbers for which the uncertainty should be computed.
        """
        
        # Checks if BME is performed for more than 2 qubits
        if self.n_qubits>2:
            print(f'BME does not support more than 2 qubits, current is {self.n_qubits}.')
            print(f'Returning the thermal state.')
            return 1/(2**self.n_qubits)*np.eye(2**self.n_qubits)
        

        # Select POVM to use for state reconstruction 
        if override_POVM_list is None:
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
            if self.n_qubits==1:
                S_treshold=0.1*self.n_bank
            elif self.n_qubits==2:
                S_treshold=0.05*self.n_bank
            
            # Shuffle outcomes such that BME converges as expected
            rng = np.random.default_rng()
            rng.shuffle(outcome_index[j])

            # Start BME loop
            for k in range(len(outcome_index[j])):
                
                weights=QST.weight_update(weights,rho_bank,full_operator_list[outcome_index[j,k]])
                S_effective=1/np.dot(weights,weights)

                #If effective sample size of posterior distribution is too low we resample
                if (S_effective<S_treshold):
                   rho_bank, weights=QST.resampling(self.n_qubits,rho_bank,weights,outcome_index[j,:k],full_operator_list,self.n_cores,self._MH_steps)
                self.infidelity[j,k]=1-np.real(np.einsum('ij,kji,k->',self.true_state_list[j],rho_bank,weights))

                # Compute average bures distance of distribution
                if compute_uncertainty and k%100==0:
                    self.uncertainty[j,k] = 2*self.infidelity[j,k]/average_Bures(rho_bank,weights,self.n_qubits,self.n_cores)

            self.rho_estimate[j]=np.array(np.einsum('ijk,i->jk',rho_bank,weights))
            print(f'Completed run {j+1}/{self.n_averages}. Final infidelity: {self.infidelity[j,-1]}.')
    
    
    def infidelity_uncertainty(rho_bank,weights):
        """
        Computes the Bayesian uncertatiny of the likelihood function in terms for the average infidelity between the Bayesian mean state and the weighted bank particles.
        """
        rho_mean = np.einsum('ijk,i->jk',rho_bank,weights)
        bank_infidelity = np.array([sf.qubit_infidelity(rho_mean,particle_rho) for particle_rho in rho_bank])**2 # The square is to make it equivalent to the variance. 
        infidelity_uncertainty = np.einsum('i,i->',bank_infidelity,weights)
        return infidelity_uncertainty
    
    
    def infidelity_uncertainty(rho_bank,weights):
        """
        Computes the Bayesian uncertatiny of the likelihood function in terms for the average infidelity between the Bayesian mean state and the weighted bank particles.
        """
        rho_mean = np.einsum('ijk,i->jk',rho_bank,weights)
        bank_infidelity = np.array([sf.qubit_infidelity(rho_mean,particle_rho) for particle_rho in rho_bank])**2 # The square is to make it equivalent to the variance. 
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
            likelihood_variance*=0.2 
        elif n_qubits==1:
            likelihood_variance*=0.4 # This is better than 1.0, tested empirically.
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
    
    def perform_grid_adaptive_BME(
        self,
        candidate_angles_grid=None,
        compute_uncertainty=None,
        depolarizing_strength=0,
        adaptive_burnin_steps=None,
    ):
        """
        Adaptive BME with a PRE-MADE GRID (no gradient descent).
        At each adaptive update, we evaluate the information-gain objective on a
        uniform grid over the (one or two) Bloch spheres and pick the best point.

        Parameters
        ----------
        candidate_angles_grid : np.ndarray or None
            - 1 qubit: shape (K, 2) with rows (theta, phi)
            - 2 qubits: shape (K, 4) with rows (theta1, phi1, theta2, phi2)
            If None, a small default grid is constructed (OK for testing).

        compute_uncertainty : bool or None
            If truthy, computes uncertainty every 50 shots (same as before).

        depolarizing_strength : float
            Depolarizing noise applied to POVM elements (0 = no depolarization).

        adaptive_burnin_steps : int or None
            Number of initial shots before the first adaptive update.
            Defaults to 0 if None.
        """
        # --- Imports (kept minimal; we still use JAX to JIT/vmap the cost eval) ---
        import numpy as np
        import jax
        import jax.numpy as jnp
        from EMQST_lib import adaptive_functions as ad

        # --- Guards and defaults ---
        if self.n_qubits > 2:
            print(f'BME does not support more than 2 qubits, current is {self.n_qubits}.')
            print(f'Returning the thermal state.')
            return 1/(2**self.n_qubits)*np.eye(2**self.n_qubits)

        if adaptive_burnin_steps is None:
            adaptive_burnin_steps = 0

        # # --- Default grid builder (fallback only; you should pass your uniform grid) --
        # if candidate_angles_grid is None:
        #     candidate_angles_grid = default_uniform_grid(self.n_qubits, per_sphere=256)

        # # Sanity check on grid shape
        # expected_cols = 2 if self.n_qubits == 1 else 4
        # if candidate_angles_grid.ndim != 2 or candidate_angles_grid.shape[1] != expected_cols:
        #     raise ValueError(
        #         f"candidate_angles_grid must have shape (K,{expected_cols}) for n_qubits={self.n_qubits}."
        #     )

        # --- Build initial (possibly depolarized) Pauli-6 POVM ---
        temp_POVM = POVM.generate_Pauli_POVM(self.n_qubits)  # starts with Pauli-6 burn-in
        decompiled_array = np.array([temp_POVM[i].get_POVM() for i in range(len(temp_POVM))])
        pauli_6_array = (1 / 3**(self.n_qubits)) * np.array(
            decompiled_array.reshape(-1, decompiled_array.shape[-2], decompiled_array.shape[-1])
        )
        depolarized_pauli_6_array = np.array(
            [sf.depolarizing_channel(np.copy(element), depolarizing_strength) for element in pauli_6_array]
        )
        depolarized_pauli_6 = POVM(depolarized_pauli_6_array)

        # --- Storage for mixed POVM list (base + appended adaptives) and outcome indices ---
        full_operator_list = []
        outcome_index = self.outcome_index.astype(int)

        # --- JAX function: vectorized evaluation of the adaptive cost over the grid ---
        @jax.jit
        def score_grid(grid_angles, rho_bank, weights, best_guess):
            """
            Returns scores for each grid angle vector using adaptive_cost_function.
            Lower is better (assuming cost is negative information gain etc.).
            """
            # vmapped over grid rows
            def _score_one(angles_array):
                return ad.adaptive_cost_function_array(angles_array, rho_bank, weights, best_guess, self.n_qubits)
            return jax.vmap(_score_one)(grid_angles)

        for j in range(self.n_averages):
            full_operator_list.append(depolarized_pauli_6_array)
            outcome_offset = 0
            adaptive_threshold = adaptive_burnin_steps
            adaptive_it = 0
            current_POVM = copy.deepcopy(depolarized_pauli_6)

            # Initialize particle bank and weights
            rho_bank = generate_bank_particles(self.n_bank, self.n_qubits)
            weights = np.full(self.n_bank, 1.0 / self.n_bank)
            if self.n_qubits == 1:
                S_treshold = 0.1 * self.n_bank
            else:
                S_treshold = 0.05 * self.n_bank

            # --- Main loop over shots ---
            for shot_index in range(self.n_shots_total):

                # Adaptive update: choose best grid point (no gradient descent)
                if shot_index > adaptive_threshold:
                    adaptive_it += 1
                    if self.n_qubits == 1:
                        candidate_angles_grid = default_uniform_grid(self.n_qubits, per_sphere=4*256*int(np.log10(shot_index+1)))
                    else: # two-qubit
                        candidate_angles_grid = default_uniform_grid(self.n_qubits, per_sphere=32*int(np.log10(shot_index+1)))
                    # Current posterior mean (best_guess)
                    current_mean_state = np.array(np.einsum('i...,i', rho_bank, weights))

                    # JAX arrays
                    jbank = jnp.array(rho_bank)
                    jweight = jnp.array(weights)
                    jmean_state = jnp.array(current_mean_state)
                    jgrid = jnp.array(candidate_angles_grid)

                    # Evaluate scores over grid and choose argmin
                    scores = score_grid(jgrid, jbank, jweight, jmean_state)
                    best_idx = int(jnp.argmin(scores))
                    best_angles = np.array(candidate_angles_grid[best_idx])

                    # Convert numpy array angles to dictionary format expected by angles_to_state_vector
                    if self.n_qubits == 1:
                        angles_dict = {
                            "theta": best_angles[0],
                            "phi": best_angles[1]
                        }
                    elif self.n_qubits == 2:
                        angles_dict = {
                            "theta_A": best_angles[0],
                            "phi_A": best_angles[1],
                            "theta_B": best_angles[2], 
                            "phi_B": best_angles[3]
                        }
                    else:
                        raise ValueError(f"n_qubits={self.n_qubits} not supported")
                    
                    # Turn best angles into a complete projective POVM (same pattern as original adaptive BME)
                    projective_vector = sf.angles_to_state_vector(angles_dict, self.n_qubits)  # shape (2^n_qubits, 2^n_qubits)
                    out = np.einsum('ij,ik->ijk', projective_vector, projective_vector.conj())  # Create projectors from state vectors

                    # Apply depolarizing channel to new POVM elements
                    depolarized_out = np.array([sf.depolarizing_channel(element, depolarizing_strength) for element in out])

                    # Update current POVM and bookkeeping
                    current_POVM = POVM(depolarized_out)
                    outcome_offset = len(full_operator_list[j])
                    if len(depolarized_out) != 2**self.n_qubits:
                        print('Warning! Adaptive POVM has an unexpected number of outcomes!')
                        print(f'Number of outcomes: {len(depolarized_out)}, expected {2**self.n_qubits}.')

                    full_operator_list[j] = np.append(full_operator_list[j], depolarized_out, axis=0)

                    # Grow the gap between adaptive updates, like your original schedule
                    adaptive_threshold += max(1, int(shot_index / 100))

                # --- Simulate measurement and update weights ---
                individual_outcome = mf.simulated_measurement(1, current_POVM, self.true_state_list[j])
                outcome_index[j, shot_index] = individual_outcome + outcome_offset

                if outcome_index[j, shot_index] > len(full_operator_list[j]) - 1:
                    print('Warning! outcome index is larger than possible outcome!')
                    print(f'Outcome index: {outcome_index[j,shot_index]}, max possible {len(full_operator_list[j])}.')
                    print(f'Current POVM has {len(current_POVM.get_POVM())} elements.')
                    print(current_POVM.get_POVM())
                    print(f'Last computed offset: {outcome_offset}.')
                    print(f'Individual outcome: {individual_outcome}.')
                    print(f'Total POVM size: {len(full_operator_list[j])}.')
                    print(f'Last two povm elements are: {full_operator_list[j][-2:]}.')

                weights = QST.weight_update(weights, rho_bank, full_operator_list[j][outcome_index[j, shot_index]])
                S_effective = 1.0 / np.dot(weights, weights)

                # Resample if posterior collapses
                if S_effective < S_treshold:
                    rho_bank, weights = QST.resampling(
                        self.n_qubits,
                        rho_bank,
                        weights,
                        outcome_index[j, :shot_index],
                        full_operator_list[j],
                        self.n_cores,
                        self._MH_steps,
                    )

                # Track infidelity
                self.infidelity[j, shot_index] = 1 - np.real(np.einsum('ij,kji,k->', self.true_state_list[j], rho_bank, weights))

                # Optional uncertainty
                if compute_uncertainty and (shot_index % 50 == 0):
                    self.uncertainty[j, shot_index] = 2 * self.infidelity[j, shot_index] / average_Bures(
                        rho_bank, weights, self.n_qubits, self.n_cores
                    )

            # Posterior mean at the end
            self.rho_estimate[j] = np.array(np.einsum('ijk,i->jk', rho_bank, weights))
            print(f'Completed run {j+1}/{self.n_averages}. Final infidelity: {self.infidelity[j,-1]}.')
   


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



def spherical_fibonacci_angles(M: int) -> np.ndarray:
    """
    Near-uniform points on S^2 using the spherical Fibonacci node set.
    Returns angles array of shape (M, 2): (theta, phi), with
      theta in [0, pi], phi in [0, 2pi).
    """
    i = np.arange(M, dtype=np.float64)
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    # z in (-1, 1), evenly spaced by area
    z = 1.0 - 2.0 * (i + 0.5) / M
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    phi = (i * ga) % (2.0 * np.pi)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    return np.stack([theta, phi], axis=1)  # (M, 2)

def default_uniform_grid(n_qubits: int, per_sphere: int = 256) -> np.ndarray:
    """
    Builds a near-uniform grid:
      - 1 qubit: (per_sphere, 2) array (theta, phi)
      - 2 qubits: (per_sphere**2, 4) array (theta1, phi1, theta2, phi2)
    """
    one = spherical_fibonacci_angles(per_sphere)  # (M,2)
    if n_qubits == 1:
        return one
    if n_qubits == 2:
        # Cartesian product of two spheres
        A, B = np.meshgrid(np.arange(per_sphere), np.arange(per_sphere), indexing='ij')
        left  = one[A.ravel()]   # (M^2, 2)
        right = one[B.ravel()]   # (M^2, 2)
        return np.concatenate([left, right], axis=1)  # (M^2, 4)
    raise ValueError("Only n_qubits in {1,2} supported.")
