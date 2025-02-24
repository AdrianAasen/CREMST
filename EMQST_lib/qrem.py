import numpy as np
import os
from joblib import Parallel, delayed
from functools import reduce
import scipy as sp


from EMQST_lib import overlapping_tomography as ot
from EMQST_lib import measurement_functions as mf
from EMQST_lib import support_functions as sf
from EMQST_lib import clustering as cl
from EMQST_lib.povm import POVM, load_random_exp_povm


class QREM:
    def __init__(self, simulation_dictionary,**kwargs):
        """
        Simulation class for the Scalable Quantum Random Error Model (QREM) simulator.
        Args:
        simulation_dictionary (dict): Dictionary containing the parameters for the simulation.
        """
        self._n_qubits = simulation_dictionary['n_qubits']
        self._n_QDT_shots = simulation_dictionary['n_QDT_shots']
        self._n_QST_shots = simulation_dictionary['n_QST_shots']
        self._n_QDT_hash_symbols = simulation_dictionary['n_QDT_hash_symbols']
        self._n_QST_hash_symbols = simulation_dictionary['n_QST_hash_symbols']
        self._n_cores = simulation_dictionary['n_cores']
        self._data_path = simulation_dictionary['data_path']
        self._max_cluster_size = simulation_dictionary['max_cluster_size']
        

        # Optional parameters
        self._initial_cluster_size = kwargs.get('initial_cluster_size', None)
        self._path_to_exp_POVMs = kwargs.get('path_to_exp_PVOMS', "Exp_povms/Extracted_modified")
        self._two_point_corr_labels = kwargs.get('two_point_corr_labels',None)
        if self._two_point_corr_labels is not None:
            self._n_two_point_correlators = len(self._two_point_corr_labels)
        self._chunk_size = kwargs.get('chunk_size', 4) # Chunk size is to simplify state measurement simulation


        # Automatic parameters 
        self._path_to_hashes =  f"EMQST_lib/hash_family/"
        self._povm_array = None
        self._exp_povms_used = []
        self._noise_cluster_labels = None

        # Define an IC calibration basis, here we use the SIC states (by defining the angles (as will be used for state preparation))
        self._one_qubit_calibration_angles = np.array([[[0,0]],[[2*np.arccos(1/np.sqrt(3)),0]],
                                                        [[2*np.arccos(1/np.sqrt(3)),2*np.pi/3]],
                                                        [[2*np.arccos(1/np.sqrt(3)),4*np.pi/3]]])
        self._one_qubit_calibration_states=np.array([sf.get_density_matrix_from_angles(angle) for angle in self._one_qubit_calibration_angles])

        self._possible_QST_instructions = np.array(["X", "Y", "Z"]) # For QST we need to meaure each qubit in the 3 Pauli basis.
        # Experiment equivalent = [[pi/2, 0], [pi/2, pi/2], [0,0]]
        self._possible_QDT_instructions = np.array([0, 1, 2, 3]) # For QDT we need to measure each of the 4 calibration states.  
        self._QDT_hash_family = None
        self._QST_hash_family = None

        # Load QDT hash family
        if self._n_QDT_hash_symbols>2:
            for files in os.listdir(self._path_to_hashes):
                if files.endswith(f"{self._n_qubits},{self._n_QDT_hash_symbols}).npy"):
                    print(f'Using hash from {files}.')
                    with open(f'{self._path_to_hashes}{files}' ,'rb') as f:
                        self._QDT_hash_family = np.load(f)
                    break # This break is to make sure it does not load a worse hash if it is stored.
            if self._QDT_hash_family is None: 
                raise ValueError("Did not find hash for this combination, please change settings or create relevant perfect hash family.")
        else: # For k=2 we can use the 2-RDM hash family
            self._QDT_hash_family = ot.create_2RDM_hash(self._n_qubits)
        self._n_QDT_hashes = len(self._QDT_hash_family)


        # Load QST hash family
        if self._n_QST_hash_symbols>2:
            for files in os.listdir(self._path_to_hashes):
                if files.endswith(f"{self._n_qubits},{self._n_QST_hash_symbols}).npy"):
                    print(f'Using hash from {files}.')
                    with open(f'{self._path_to_hashes}{files}' ,'rb') as f:
                        self._QST_hash_family = np.load(f)
                    break # This break is to make sure it does not load a worse hash if it is stored.
            if self._QST_hash_family is None: 
                raise ValueError("Did not find hash for this combination, please change settings or create relevant perfect hash family.")
        else: # For k=2 we can use the 2-RDM hash family
            self._QST_hash_family = ot.create_2RDM_hash(self._n_qubits)
        self._n_QST_hashes = len(self._QST_hash_family)

     # Experiment equivalent =  one_qubit_calibration_angles
        self._hashed_QST_instructions = ot.create_hashed_instructions(self._QST_hash_family, self._possible_QST_instructions, self._n_QST_hash_symbols)
        # Create QDT instructions based on hashing (covering arrays) 
        self._hashed_QDT_instructions = ot.create_hashed_instructions(self._QDT_hash_family, self._possible_QDT_instructions, self._n_QDT_hash_symbols)
        # Create hashed calibration states
        self._hashed_calib_states = np.array([ot.calibration_states_from_instruction(instruction, self._one_qubit_calibration_states) for instruction in self._hashed_QDT_instructions])


    def set_random_cluster_size(self, max_cluster_size = 3):
        """
        Sets the cluster size randomly that adhers to the max cluster size.
        """
        self._max_cluster_size = max_cluster_size
        n_chunks = int(self._n_qubits//self._chunk_size) # Assumes chunk_size i a factor of n_qubits. 
        # Set max chunk size and cluster size equal
        self._initial_cluster_size = ot.generate_chunk_sizes(self._chunk_size, n_chunks, self._max_cluster_size)


    def set_initial_cluster_size(self, intial_cluster_array):
        self._initial_cluster_size = intial_cluster_array.copy()

    def print_current_state(self):
        print("The shot budget of the currents settings are:")
        print(f'QDT shots for computational basis reconstruction: {(self._n_QDT_hashes*(4**self._n_QDT_hash_symbols -4) +4):,} x {self._n_QDT_shots:,}.')
        print(f'QST shots for arbitrary {self._n_QST_hash_symbols}-RDM reconstruction: {(self._n_QST_hashes*(3**self._n_QST_hash_symbols -3) +3):,} x {self._n_QST_shots:,}.')
        return 1
    
    
    # get and set functions
    @property
    def n_qubits(self):
        return self._n_qubits
    
    @property
    def inital_cluster_size(self):
        return self._initial_cluster_size
    
    @property
    def cluster_cutoff(self):
        return self._cluster_cutoff
    
    @property
    def Z(self):
        return self._Z
    
    @property
    def QDT_outcomes(self):
        return self._QDT_outcomes

    @property
    def n_QDT_shots(self):
        return self._n_QDT_shots
    
    @property
    def n_QST_shots(self):
        return self._n_QST_shots
    
    @property
    def QDT_hash_family(self):
        return self._QDT_hash_family
    
    @property
    def QST_hash_family(self):
        return self._QST_hash_family
    
    @property
    def n_QDT_hash_symbols(self):
        return self._n_QDT_hash_symbols
    
    @property
    def n_QST_hash_symbols(self):
        return self._n_QST_hash_symbols
    
    @property
    def one_qubit_calibration_states(self):
        return self._one_qubit_calibration_states
    
    @property
    def n_cores(self):
        return self._n_cores
    
    @property
    def data_path(self):
        return self._data_path
    
    @property
    def rho_true_array(self):
        return self._rho_true_array
    @property
    def Z(self):
        return self._Z

    @rho_true_array.setter
    def rho_true_array(self, value):
        self._rho_true_array = value

    def save_initialization(self, save_path = None):
        QDOT_run_dictionary = {
        "n_QST_shots": self._n_QST_shots,
        "n_QDT_shots": self._n_QDT_shots,
        "n_QST_hash_symbols": self._n_QST_hash_symbols,
        "n_QDT_hash_symbols": self._n_QDT_hash_symbols,
        "n_qubits": self._n_qubits,
        "n_cores": self._n_cores,
        "QST_hash_family": self._QST_hash_family,
        "n_QST_hashes": self._n_QST_hashes,
        "QDT_hash_family": self._QDT_hash_family,
        "n_QDT_hashes": self._n_QDT_hashes,
        "noise_mode": self._noise_mode,
        "povm_array": [povm.get_POVM() for povm in self._povm_array], # Can be of inhomogenious shape.
        "initial_cluster_size": self._initial_cluster_size,
        "n_clusters": self._n_clusters,
        "possible_QDT_instructions": self._possible_QDT_instructions,
        "possible_QST_instructions": self._possible_QST_instructions,
        "exp_POVMs_loaded" : self._exp_povms_used # List of paths to to POVMs used in the noise sumulations. 
        }
        
        if save_path is not None:
            self._data_path = save_path


        with open(f'{self._data_path}/run_settings.npy','wb') as f:
            np.save(f,QDOT_run_dictionary)


    def set_POVM_array_manually(self, povm_array, inital_cluster_size):
        """
        Set the POVMs to be used for the QDT measurements manually.
        """
        self._initial_cluster_size = inital_cluster_size.copy()
        self._povm_array = povm_array.copy()
        self.true_cluster_labels = cl.get_true_cluster_labels(self._initial_cluster_size)
        self._n_clusters = len(self._initial_cluster_size)

    def set_exp_POVM_array(self, noise_mode='strong'):
        """
        Sets the POVMs from experimental list at random. The noise mode is used.
        Args:
        Noise_mode (int): The noise mode to use for the POVMs.
                        'strong' : Random noise from strong set of noise.
                        'weak' : Random noise from weak set of noise.
        """
        if self._initial_cluster_size is None:
            raise ValueError("Please set the cluster size before setting the POVM array.")

        self._n_clusters = len(self._initial_cluster_size)
        self._povm_array = []
        for size in self._initial_cluster_size:
            if noise_mode == 'strong':
                self._noise_mode = 'strong' + ' exp'
                noisy_POVM_list, load_path = load_random_exp_povm(self._path_to_exp_POVMs, size)
                self._exp_povms_used.append(load_path)
            elif noise_mode == 'weak':
                self._noise_mode = 'weak' + ' exp'
                noisy_POVM_list, load_path = load_random_exp_povm(self._path_to_exp_POVMs, size, use_Z_basis_only=True)
                self._exp_povms_used.append(load_path)
            self._povm_array.append(noisy_POVM_list)
        self.true_cluster_labels = cl.get_true_cluster_labels(self._initial_cluster_size)
        print(f'Loaded {len(self._exp_povms_used)} POVMs from {self._path_to_exp_POVMs}.')
        

    def copy_POVM_array(self, qrem):
        self._povm_array = np.array(qrem._povm_array)
        self._initial_cluster_size = np.array(qrem._initial_cluster_size)
        self.true_cluster_labels = qrem.true_cluster_labels
        self._noise_mode = qrem._noise_mode
        self._n_clusters = qrem._n_clusters
        
        

    def set_coherent_POVM_array(self, angle=np.pi/10):
        """
        Sets noise POVM where XX-crosstalk is applied to all adjacent qubits.
        If a cluster only has a single qubit, a costant rotation with the same angle is applied.
        The model rotates an angle around the collective x axis of the qubits.
        """
        if self._initial_cluster_size is None:
            raise ValueError("Please set the cluster size before setting the POVM array.")
        X = np.array([[0,1],[1,0]])
        Id = np.eye(2)
        self._n_clusters = len(self._initial_cluster_size)
        self._povm_array = []
        self._noise_mode = 'coherent' + 'angle=' + str(angle)
        for size in self._initial_cluster_size:
            if size == 1 or size == 2:
                rotation_matrix = sf.rot_about_collective_X(angle, size)
            else: # If size is larger than 2, iterative next neighbor rotations
                XX = np.kron(X,X)
                full_H = np.zeros((2**size,2**size),dtype=complex)

                for i in range(size-1): 
                    H_temp = [Id]*(size-1)
                    H_temp[i] = XX
                    full_H += reduce(np.kron,H_temp)
                rotation_matrix = sp.linalg.expm(-1/2j * angle * full_H)
                
            # print("Check unitary:")
            # print(rotation_matrix@rotation_matrix.conj().T)
            comp_povm_array = POVM.generate_computational_POVM(size)[0].get_POVM()
            self._povm_array.append(POVM(np.einsum('jk,ikl,lm->ijm',rotation_matrix.conj().T,comp_povm_array,rotation_matrix)))
                
        self.true_cluster_labels = cl.get_true_cluster_labels(self._initial_cluster_size)




    def set_correlated_POVM_array(self, k_mean=0.5, noise_mode='iSWAP'):
        """
        Sets the POVM array to be a depolarized iSWAP/CNOT POVM.
        This noise mode overrides the cluster size and sets it to be all neigbhoring clusters.
        The noise is appled inversly to the POVM to be equivalent to the same implementation on the state.
        """
        if noise_mode == 'iSWAP':
            iSWAP=np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]],dtype=complex)
            noise_matrix = iSWAP
            self._noise_mode = 'iSWAP' 
        elif noise_mode == 'CNOT':
            CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
            noise_matrix = CNOT
            self._noise_mode = 'CNOT'
        else:
            raise ValueError("Noise mode not supported, please use 'iSWAP' or 'CNOT'.")
        self._povm_array = []
        self._initial_cluster_size = np.ones(int(self._n_qubits/2),dtype=int)*2
        self._n_clusters = len(self._initial_cluster_size)
        comp_povm_array = POVM.generate_computational_POVM(2)[0].get_POVM()
        self._random_mixing_strenght =  (np.random.random(int(self._n_qubits/2))*0.2 - 0.1) + k_mean
        self._povm_array = [POVM((1-noise_strenght)*comp_povm_array + noise_strenght*np.einsum('jk,ikl,lm->ijm',noise_matrix.conj().T,comp_povm_array,noise_matrix)) for noise_strenght in self._random_mixing_strenght]
        self.true_cluster_labels = cl.get_true_cluster_labels(self._initial_cluster_size)



        
        
    def perform_QDT_measurements(self):
        if self._povm_array is None:
            raise ValueError("No POVM array set, please set the POVM array before performing measurements.")
               

        # Simulate all instruction measurements
        print(f'Simulating QDT measurements for {self._n_qubits} qubits.')
        QDT_outcomes_parallel = Parallel(n_jobs = self._n_cores, verbose = 5)(delayed(mf.measure_clusters)(self._n_QDT_shots, self._povm_array, rho_array, self._initial_cluster_size) for rho_array in self._hashed_calib_states)
        self._QDT_outcomes = np.asarray(QDT_outcomes_parallel)

    def delete_QDT_outcomes(self):
        del self._QDT_outcomes
        
    def delete_QST_outcomes(self):
        del self._QST_outcomes


    def perform_clustering(self, cutoff = None, max_cluster_size = None, method = None, wc_distance_ord = np.inf):
        """
        Performs hierarchical clustering based on the QDT outcomes.
        wc_distance_ord (int): The order of the vector distance used for the operator norm. Default is infinity norm, as described in the paper.
        """
        if max_cluster_size is not None:
            self._max_cluster_size = max_cluster_size
        elif self._max_cluster_size is None:
            raise ValueError("Please set the max cluster size before performing clustering.")

        if cutoff is None:
            self._cluster_cutoff = 1 - 1/np.sqrt(self._n_QDT_shots)
        else:
            self._cluster_cutoff = cutoff
        self._two_point_POVM, self._two_point_POVM_labels = ot.reconstruct_all_two_qubit_POVMs(self._QDT_outcomes, self._n_qubits, self._QDT_hash_family, self._n_QDT_hash_symbols, self._one_qubit_calibration_states, self._n_cores)
        self._summed_quantum_corr_array, self._unique_corr_labels = ot.compute_quantum_correlation_coefficients(self._two_point_POVM, self._two_point_POVM_labels, mode="WC", wc_distance_ord = wc_distance_ord)
        
        
        # Create distance matrix 
        self._dist_matrix = cl.create_distance_matrix_from_corr(self._summed_quantum_corr_array, self._unique_corr_labels, self._n_qubits)

        self._fcluster_labels, self._Z = cl.hierarchical_clustering(self._dist_matrix, self._cluster_cutoff, method )
        self._noise_cluster_labels = cl.fcluster_to_labels(self._fcluster_labels)

        # If clustering is too large:
        self._noise_cluster_size = [len(item) for item in cl.fcluster_to_labels(self._fcluster_labels)]
        while max(self._noise_cluster_size) > self._max_cluster_size:
            print("Initial Cluster Assignments:", cl.fcluster_to_labels(self._fcluster_labels))
            self._fcluster_labels = cl.split_oversized_clusters(self._Z , self._fcluster_labels, self._max_cluster_size )
            self._noise_cluster_labels = cl.fcluster_to_labels(self._fcluster_labels)
            self._noise_cluster_size = [len(item) for item in cl.fcluster_to_labels(self._fcluster_labels)]
            print("Final Cluster Assignments:", self._noise_cluster_labels)

        
    def update_cluster_cutoff(self, cutoff, method = None):
        """
        Updates the cluster cutoff and recomputes clustering.
        """
        self._cluster_cutoff = cutoff

        self._fcluster_labels, self._Z = cl.hierarchical_clustering(self._dist_matrix,  self._cluster_cutoff, method = method)
        self._noise_cluster_labels = cl.fcluster_to_labels(self._fcluster_labels)

        # If clustering is too large:
        self._noise_cluster_size = [len(item) for item in cl.fcluster_to_labels(self._fcluster_labels)]
        while max(self._noise_cluster_size) > self._max_cluster_size:
            print("Initial Cluster Assignments:", cl.fcluster_to_labels(self._fcluster_labels))
            self._fcluster_labels = cl.split_oversized_clusters(self._Z , self._fcluster_labels, self._max_cluster_size )

            self._noise_cluster_labels = cl.fcluster_to_labels(self._fcluster_labels)
            print(self._noise_cluster_labels)
            self._noise_cluster_size = [len(item) for item in cl.fcluster_to_labels(self._fcluster_labels)]
            print("Final Cluster Assignments:", self._noise_cluster_labels)


    def reconstruct_cluster_POVMs(self):
        """
        Reconstructs the POVMs for each cluster.
        """
        self._clustered_QDOT = ot.reconstruct_POVMs_from_noise_labels(self._QDT_outcomes, self._noise_cluster_labels, self._n_qubits, self._QDT_hash_family, self._n_QDT_hash_symbols,
                                                        self._one_qubit_calibration_states, self._n_cores)

    def reconstruct_cluster_with_perfect_clustering(self):
        """
        Reconstructs the POVMs for each cluster with perfect clustering.
        """
        self._perfect_clustered_QDOT = ot.reconstruct_POVMs_from_noise_labels(self._QDT_outcomes, self.true_cluster_labels, self._n_qubits, self._QDT_hash_family, self._n_QDT_hash_symbols,
                                                        self._one_qubit_calibration_states, self._n_cores)
        
    def reconstruct_all_one_qubit_POVMs(self):
        """
        Reconstructs the POVMs for each qubit.
        """
        self._one_qubit_POVMs = ot.reconstruct_all_one_qubit_POVMs(self._QDT_outcomes, self._n_qubits, self._QDT_hash_family, self._n_QDT_hash_symbols, self._one_qubit_calibration_states, self._n_cores)



    def set_chunked_true_states(self, n_averages = 1, mode = 'random', chunk_size = None):
        """
        Sets the true states for QST.
        Lists of modes:
        'random' : Haar random states of chunk size.
        'GHZ' : GHZ states of chunk size.

        rho_true_array (list): List of true states for QST, shape [n_averages, n_chunks, 2**chunk_size, 2**chunk_size]
        """
        if chunk_size is not None:
            self._chunk_size = chunk_size
        self._n_averages = n_averages
        self._true_state_mode = mode
        self._state_size_array = [self._chunk_size]*int(self._n_qubits/self._chunk_size)
        self._rho_true_labels = cl.get_true_cluster_labels(self._state_size_array)
        if mode == 'random':
            self._rho_true_array = [[sf.generate_random_pure_state(size) for size in self._state_size_array] for _ in range(n_averages)]
        elif mode == 'GHZ':
            self._rho_true_array = [[sf.generate_GHZ(size) for size in self._state_size_array] for _ in range(n_averages)]

        #self._rho_true_list, self._rho_labels_in_state = ot.tensor_chunk_states(self._rho_true_array, self._rho_true_labels, self._noise_cluster_labels, self._two_point_corr_labels)

    def copy_chunked_true_states(self,qrem):
        """
        Copies the chunked true states from self to another QREM object.
        
        """
        self._rho_true_array = np.array(qrem._rho_true_array)
        self._rho_true_labels = np.array(qrem._rho_true_labels)
        self._n_averages = qrem._n_averages
        self._true_state_mode = qrem._true_state_mode
        self._state_size_array = np.array(qrem._state_size_array)
        
        
    def perform_averaged_QST_measurements(self, n_QST_shots = None):
        """
        Performs QST measurements on the reconstructed POVMs.
        """
        if n_QST_shots is not None:
            self._n_QST_shots = n_QST_shots
        if self._clustered_QDOT is None:
            raise ValueError("Please reconstruct the POVMs before performing QST measurements.")

        self._QST_outcomes = [mf.measure_hashed_chunk_QST(self._n_QST_shots, self._chunk_size, self._povm_array, self._initial_cluster_size, self._rho_true_array[i], self._state_size_array, self._hashed_QST_instructions) for i in range(self._n_averages) ]

    def compute_correlator_true_states(self):
        """
            Initializes all internal parameters for the requested two-point correlators.
            This method prepares the true states for the relevant correlators and reconstructs the two-qubit 
            POVMs (Positive Operator-Valued Measures) for that set of correlators.
            Parameters:
            -----------
            two_point_corr_labels : list of lists
                A list of lists where each list contains two qubit indices representing the correlators. 
                If not provided, random pairs of qubits will be generated.
            n_two_point_correlators : int, optional
                The number of two-point correlators to generate if `two_point_corr_labels` is not provided. 
                Default is 1.
            Notes:
            ------
            - If `two_point_corr_labels` is provided, it sets the internal labels and the number of correlators 
              based on the length of this list.
            - If `two_point_corr_labels` is not provided, it generates random pairs of qubits for the specified 
              number of correlators.
            - It prints the set two-point correlators.
            - It creates true states for the correlators to compare to and stores them in `traced_down_correlator_rho_true_array` based on the set of 'rho_ture_array'.
            - It reconstructs the two-qubit POVMs for the set of correlators and stores them in `two_point_POVM_array`.
        """
        if  self._two_point_corr_labels is None:
            ValueError("Please set the two-point correlators before performing measurements.")
    
        self._n_two_point_correlators = len(self._two_point_corr_labels)


        print(f'Setting two-point correlators to {self._two_point_corr_labels}.')
        # Create true states for the correlators to compare to. 
        self.traced_down_correlator_rho_true_array = []

        for av in range(self._n_averages):
            rho_true_list, rho_labels_in_state = ot.tensor_chunk_states(self._rho_true_array[av], self._rho_true_labels, 
                                                                                    self._noise_cluster_labels, self._two_point_corr_labels)

            traced_down_rho_true = [ot.trace_down_qubit_state(rho_true_list[i], rho_labels_in_state[i], 
                                    np.setdiff1d(rho_labels_in_state[i], self._two_point_corr_labels[i])) for i in range(len(rho_true_list))]
            self.traced_down_correlator_rho_true_array.append(traced_down_rho_true)
        # Reconstruct all two-qubit POVMs used for the correlators (only used for the REMST method (two-point QREM))
        self._two_point_POVM_array = ot.reconstruct_spesific_two_qubit_POVMs(self._QDT_outcomes, self._two_point_corr_labels , self._n_qubits, self._QDT_hash_family, 
                                                         self._n_QDT_hash_symbols, self._one_qubit_calibration_states, self._n_cores)

        
    
    
    def perform_two_point_correlator_QST(self, reconstruction_methods, assume_perfect_clustering = False):
        """
        This is the core of the reconstruction analasis.
        Args:
        reconstruction_methods (list of int): integer list indicating what reconstruction methods to use.
        The methods are labeld as follows:
        0: no QREM
        1: factorized QREM
        2: two RDM QREM
        3: Cluster-concious two-point QREM
        4: Classical correlated QREM
        5: Entanglement safe QREM
        There are in addition two more methods that are not included which can be accessed with the following labels:
        6: Classical state reductio
        7: State reduction QREM

        Returns a dictionary with all the reconstructed results. 
        """
        if assume_perfect_clustering:
            used_povm = self._povm_array
            applied_clustering = self.true_cluster_labels
        else:
            used_povm = self._clustered_QDOT
            applied_clustering = self._noise_cluster_labels
        

        result_dict  = ot.perform_full_comparative_QST(applied_clustering, self._QST_outcomes,  self._two_point_corr_labels, 
                                                  used_povm, self._one_qubit_POVMs, self._two_point_POVM_array, self._n_averages, 
                                                  self._QST_hash_family, self._n_QST_hash_symbols, self._n_qubits, self._n_cores, method = reconstruction_methods)
        
        # Include additional info in result dict
        result_dict["rho_true_array"] = self._rho_true_array
        result_dict["traced_down_rho_true_array"] = self.traced_down_correlator_rho_true_array 
        result_dict["initial_cluster_size"] = self._initial_cluster_size
        result_dict["noise_mode"] = self._noise_mode
        result_dict["n_QDT_shots"] = self._n_QDT_shots  
        result_dict["n_QST_shots"] = self._n_QST_shots
        
        return result_dict

    def perform_correlated_QREM_comparison(self,comparison_methods = None):
        """
        Function performs comparativ QST measurements where each method recieves the same measurements outcomes as correlated QREM. 
        
        The protocol is as follows:
        - Performs the nessecary QST measurements to reconstruct the connected noise clusters.
        - Computes the readout error mitigated states for the connected cluster, the two-point correlators and the one-qubit POVMs.
        - Assumes it takes in a single two-point correlator label, to make function parallelizable.
        Args
        comparison_methods: list of integers that selects which methods to compare to correlated QREM.
        0: no QREM
        1: factorized QREM
        2: two RDM QREM
        3: Classical correlated QREM
        """
        if comparison_methods is None:
            comparison_methods = [0]
            print(f'No method selected. Comparsion defaults to no QREM.')
        # Perform QST measurements for each of the connected clusters separatly
        QST_result = Parallel(n_jobs = self._n_cores, verbose = 5)(delayed(mf.correlator_spesific_QST_measurements)(
            self._noise_cluster_labels, two_point_label, self._n_averages, self._n_qubits, comparison_methods, 
            self._n_QST_shots, self._chunk_size, self._povm_array, self._initial_cluster_size, self._rho_true_array,
            self._state_size_array) for two_point_label in self._two_point_corr_labels)
        self._QST_outcomes, target_qubits, QST_instructions = zip(*QST_result)

        print(f'QST state sizes: {[len(qubit_set) for qubit_set in target_qubits]}.')
        
        # For each of these QST results, compute the following based on methods:
        state_results = Parallel(n_jobs=self._n_cores, verbose = 5)(delayed(ot.perform_comparative_QST)(
            self._noise_cluster_labels, self._two_point_corr_labels[i], self._QST_outcomes[i],
            self._clustered_QDOT,self._one_qubit_POVMs, self._two_point_POVM_array[i], self._n_averages,
            self._n_qubits, comparison_methods, target_qubits[i], QST_instructions[i]
            ) for i in range(len(self._two_point_corr_labels)))
        #result_array = np.array(state_results)
        return state_results

        #print(self.traced_down_correlator_rho_true_array[0][i])
        # print('Checking infidelities')
        # for i in range(len(state_results)):
        #     print(sf.qubit_infidelity(state_results[i][0], self.traced_down_correlator_rho_true_array[0][i]))


        