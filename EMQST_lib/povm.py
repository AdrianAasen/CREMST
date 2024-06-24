import numpy as np
import scipy as sp
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
from functools import reduce
from itertools import repeat, chain, product
from scipy.optimize import minimize

from EMQST_lib import support_functions as sf



class POVM():
    """
    This class contains all information needed from the POVM.
    Lists of POVMs and individual POVMs.
    The POVM generation functions support individual spin measurements on each qubit,
    to allgin with the experimental implementation on the superconducting qubits.
    The class itself does support arbitrary POVMs. 
    """
    def __init__(self,POVM_list,angle_representation=np.array([])):
        self.POVM_list=POVM_list
        self.angle_representation=angle_representation

    
    def __eq__(self,other):
        if not isinstance(other, POVM):
            return False
        if self.POVM_list.shape != other.get_POVM().shape:
            return False
        return np.all(self.POVM_list==other.get_POVM())
    
    @classmethod
    def POVM_from_angles(cls, angles):
        """
        Creates a POVM class based on a set of angles defining spin measurements. 

        Args:
            angles (ndarray): n x 2 array of angles [theta, phi] (n qubits, 2 angles)
                angles defines what is considered the 'up' outcome. 

        Returns:
            POVM: An instance of the POVM class. 
        """
        
        angle_representation = np.zeros((2**len(angles), len(angles), 2))

        opposite_angles = sf.get_opposing_angles(angles)
        angle_Matrix = np.array([angles, opposite_angles])
        projector_up = np.zeros((len(angles), 2, 2), dtype=complex)
        projector_down = np.zeros((len(angles), 2, 2), dtype=complex)

        for i in range(len(angles)):
            projector_up[i] = sf.get_projector_from_angles(np.array([angles[i]]))
            projector_down[i] = sf.get_projector_from_angles(np.array([opposite_angles[i]]))
            
            # Creates matrix with [up/down index, qubit positition, 2x2 matrix]
        projector_matrix = np.array([projector_up, projector_down])
        POVM_list = np.zeros((2**len(angles), 2**len(angles), 2**len(angles)), dtype=complex)
        M_temp = 1

        for i in range(2**len(angles)): # iterates over all possible bitstrings
            bit_string = bin(i)[2:].zfill(len(angles)) # Creates binary string of all possible permutations
            M_temp = 1

            for j in range(len(angles)):
                M_temp = np.kron(M_temp, projector_matrix[int(bit_string[j]), j])
                angle_representation[i, j] = angle_Matrix[int(bit_string[j]), j]

            POVM_list[i] = M_temp

        return cls(POVM_list, angle_representation)
    
    # @classmethod
    # def POVM_from_angles(cls,angles):
    #     """
    #     Creates a POVM class based on a set of angles defining spin measurements. 
    #     angles: ndarray n x 2 (n qubits, 2 angles [theta,phi])
    #             angles defines what is considere the 'up' outcome. 
    #     return POVM class. 
    #     """
        
    #     angle_representation=np.zeros((2**len(angles),len(angles),2))

    #     opposite_angles=sf.get_opposing_angles(angles)
    #     angle_Matrix=np.array([angles,opposite_angles])
    #     projector_up=np.zeros((len(angles),2,2),dtype=complex)
    #     projector_down=np.zeros((len(angles),2,2),dtype=complex)
    #     #print(opposite_angles)
    #     for i in range(len(angles)):
    #         projector_up[i]=sf.get_projector_from_angles(np.array([angles[i]]))
    #         projector_down[i]=sf.get_projector_from_angles(np.array([opposite_angles[i]]))
    #     projector_matrix=np.array([projector_up,projector_down]) # creates matrix with [up/down index, qubit positition, 2x2 matrix]
    #     #print(projector_matrix)
    #     POVM_list=np.zeros((2**len(angles),2**len(angles),2**len(angles)),dtype=complex)
    #     M_temp=1
    #     for i in range (2**len(angles)): # iterates over all possible bitstrings 
    #         bit_string=bin(i)[2:].zfill(len(angles)) # Creates binary string of all possible permutations
    #         M_temp=1
    #         for j in range (len(angles)):
    #             M_temp=np.kron(M_temp,projector_matrix[int(bit_string[j]),j])
    #             angle_representation[i,j]=angle_Matrix[int(bit_string[j]),j]
    #         POVM_list[i]=M_temp

    #     return cls(POVM_list,angle_representation)
    
    @classmethod
    def generate_Pauli_POVM(cls, n_qubits):
        """
        Recursively create higher qubit POVMs.
        
        Input:
            n_qubits: The number of qubits.
        
        Returns:
            A list of 3 spin POVMs along the x, y, and z axis.
        """
        POVM_set_X = 1/2 * np.array([[[1,1],[1,1]],[[1,-1],[-1,1]]], dtype=complex)
        POVM_set_Y = 1/2 * np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]], dtype=complex)
        POVM_set_Z = np.array([[[1,0],[0,0]],[[0,0],[0,1]]], dtype=complex)
        
        POVM_X = cls(POVM_set_X, np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]]]))
        POVM_Y = cls(POVM_set_Y, np.array([[[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]]]))
        POVM_Z = cls(POVM_set_Z, np.array([[[0,0]],[[np.pi,0]]]))
        
        POVM_single = np.array([POVM_X, POVM_Y, POVM_Z])
        POVM_list = np.copy(POVM_single)
        
        for _ in range(n_qubits - 1):
            POVM_list = POVM.tensor_POVM(POVM_list, POVM_single)
            
        return POVM_list
    
    @classmethod
    def tensor_POVM(cls, POVM_1, POVM_2):
        """
        Tensor product of two POVMs. This method also tensor together the angle representaiton of the POVMs.

        Args:
            POVM_1 (list): List of POVM objects.
            POVM_2 (list): List of POVM objects.

        Returns:
            np.ndarray: Array of tensor product POVMs.

        """
        if isinstance(POVM_1, POVM): # If they send in POVM elements, we need to make them into a list. 
            POVM_1 = np.array([POVM_1])
        if isinstance(POVM_2, POVM):
            POVM_2 = np.array([POVM_2])
            
        POVM_list = np.array([cls(np.array([np.kron(a, b) for a in POVM_a.get_POVM() for b in POVM_b.get_POVM()]),
                                np.array([np.concatenate((angle_a, angle_b)) for angle_a in POVM_a.get_angles() for angle_b in POVM_b.get_angles()]))
                                for POVM_a in POVM_1 for POVM_b in POVM_2])
        return POVM_list
    # @classmethod
    # def tensor_POVM(cls,POVM_1,POVM_2):
    #     POVM_list=np.array([cls(np.array([np.kron(a,b) for a in POVM_a.get_POVM() for b in POVM_b.get_POVM()]),
    #                             np.array([np.concatenate((angle_a,angle_b)) for angle_a in POVM_a.get_angles() for angle_b in POVM_b.get_angles()] ))
    #                          for POVM_a in POVM_1 for POVM_b in POVM_2])
    #     return POVM_list
        
    
    @classmethod
    def empty_POVM(cls):
        """
        Returns an empty class.

        This method creates and returns an empty instance of the POVM class.

        Returns:
        - An empty instance of the POVM class.

        """
        return cls(np.array([]), np.array([]))

    @classmethod
    def generate_computational_POVM(cls, n_qubits=1):
        """
        Returns the z-basis POVM for the specified number of qubits.

        Parameters:
        - n_qubits (optional): The number of qubits. Default is 1.

        Returns:
        - return_POVM: The z-basis POVM.

        """
        # Set up single qubit
        single_qubit = np.array([cls(np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex),np.array([[[0,0]],[[np.pi,0]]]))])
        return_POVM  = single_qubit
        for _ in range(n_qubits-1):
            return_POVM = POVM.tensor_POVM(return_POVM,single_qubit)
        
        return return_POVM

    
    def get_histogram(self, rho):
        """
        Takes in state of same dimension as the POVM.
        Returns histogram for all probabilities of outcomes defined by POVM.
        State and POVM dimension must be compatible. 

        Parameters:
        - rho: numpy.ndarray
            The state of arbitrary dimension.

        Returns:
        - numpy.ndarray
            The histogram of probabilities for all outcomes defined by POVM.
        """
        return np.real(np.einsum('ijk,kj->i', self.POVM_list, rho))
    
    def get_POVM(self):
            """
            Returns a copy of the POVM list.

            Returns:
                numpy.ndarray: A copy of the POVM list.
            """
            return np.copy(self.POVM_list)
    
    def get_angles(self):
            """
            Returns a copy of the angle representation of the POVM.

            Returns:
                numpy.ndarray: A copy of the angle representation of the POVM.
            """
            return np.copy(self.angle_representation)

    @classmethod
    def depolarized_POVM(cls, base_POVM, strength=0.1):
        """
        Creates a depolarized POVM by combining a base POVM with a depolarization strength.

        Args:
            base_POVM (POVM): The base POVM to be depolarized.
            strength (float): The depolarization strength. Default is 0.1.

        Returns:
            POVM: The depolarized POVM.

        """
        base_POVM_list = base_POVM.get_POVM()
        dim = int(len(base_POVM_list[0]))
        new_list = strength/dim * np.eye(dim) + (1-strength) * base_POVM_list
        return cls(new_list)

    @classmethod
    def generate_noisy_POVM(cls, base_POVM ,noise_mode):
        """
        Takes in an POVM and applied the inverse Krauss channel.
        Returns noisy POVM class object. 
        noise_mode (single qubit)
        1: Constant depolarizing noise
        2: Stronger depolarizing noise
        3: Amplitude damping noise
        4: Constant over-rotation
        
        noise mode (2 qubits) (applies transformation with a given probability)
        0: No noise
        1: CNOT noise
        2: ISWAP noise
        3: Constant random rotation
        4: Constant factorized rotation
        5: Small facotrized overrotation
        6: Small x rotation only on first qubit
        7: Strong depolarizing
        """
        base_POVM_list=base_POVM.get_POVM()
        X=np.array([[0,1],[1,0]])
        Y=np.array([[0,-1j],[1j,0]])
        Z=np.array([[1,0],[0,-1]])
        n_qubits=int(np.log2(len(base_POVM_list[0])))
        sigma = np.array([X,Y,Z],dtype=complex)
        if noise_mode==0:
            print("No noise mode selected. Returning base POVM")
            return cls(base_POVM_list)
        
        if n_qubits == 1:
        
            if noise_mode == 1: # Constant depolarizing noise
                p=0.05   
                Krauss_op=np.array([np.sqrt(1-(3*p)/4)*np.eye(2),np.sqrt(p)*X/2,np.sqrt(p)*Y/2,np.sqrt(p)*Z/2],dtype=complex)
            elif noise_mode == 2: # Stronger depolarizing noise
                p=0.2
                Krauss_op=np.array([np.sqrt(1-(3*p)/4)*np.eye(2),np.sqrt(p)/2*X,np.sqrt(p)*Y/2,np.sqrt(p)*Z/2],dtype=complex)
            elif noise_mode == 3: # Amplitude damping noise
                gamma=0.2
                K0=np.array([[1,0],[0,np.sqrt(1-gamma)]],dtype=complex)
                K1=np.array([[0,np.sqrt(gamma)],[0,0]],dtype=complex)
                Krauss_op=np.array([K0.conj().T,K1.conj().T])
            elif noise_mode == 4: # Constant over-rotation
                rotAngle=np.pi/5
                U = np.cos(rotAngle/2)*np.eye(2) - 1j* np.sin(rotAngle/2)*np.array([[0,1],[1,0]])
                Krauss_op=np.array([U],dtype=complex)

            noisy_POVM_list=np.einsum('nij,qjk,nlk->qil',Krauss_op,base_POVM_list,Krauss_op.conj())
            return cls(noisy_POVM_list)
        
        elif n_qubits == 2: # Two qubit noise
            CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
            ISWAP=np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]],dtype=complex)
            if noise_mode == 1: 
                noise_transformation=CNOT
                k = 0.2 # Mixing strenght (probability)
                noisy_POVM_list=k*base_POVM_list + (1-k)*np.einsum('jk,ikl,lm->ijm',noise_transformation.conj().T,base_POVM_list,noise_transformation)
                return cls(noisy_POVM_list) 
            elif noise_mode == 2:
                noise_transformation=ISWAP
                k = 0.2 # Mixing strenght (probability)
                noisy_POVM_list=k*base_POVM_list + (1-k)*np.einsum('jk,ikl,lm->ijm',noise_transformation.conj().T,base_POVM_list,noise_transformation)
                return cls(noisy_POVM_list) 
                
            elif noise_mode == 3: # Constant 2 qubit random rotation
                U=unitary_group.rvs(2**n_qubits)
                Krauss_op=np.array([U],dtype=complex)
                noisy_POVM_list=np.einsum('nij,qjk,nlk->qil',Krauss_op,base_POVM_list,Krauss_op.conj())
                return cls(noisy_POVM_list)
            
            elif noise_mode == 4: # Factorized random rotation 
                U_1 = unitary_group.rvs(2)
                U_2 = unitary_group.rvs(2)
                U = np.kron(U_1,U_2)
                Krauss_op=np.array([U],dtype=complex)
                noisy_POVM_list=np.einsum('nij,qjk,nlk->qil',Krauss_op,base_POVM_list,Krauss_op.conj())
                return cls(noisy_POVM_list)
            
            elif noise_mode == 5: # Facotrized small overrotation
                axis_1 = np.array([1,0,0])
                axis_2 = np.array([1/np.sqrt(2),1/np.sqrt(2),0])
                angle_1 = np.pi /5
                angle_2 = np.pi /7
                U_1 = sp.linalg.expm(-1/2j * angle_1 * np.einsum('j,jkl->kl',axis_1,sigma))
                U_2 = sp.linalg.expm(-1/2j * angle_2 * np.einsum('j,jkl->kl',axis_2,sigma))
                U = np.kron(U_1,U_2)
                Krauss_op=np.array([U],dtype=complex)
                noisy_POVM_list=np.einsum('nij,qjk,nlk->qil',Krauss_op,base_POVM_list,Krauss_op.conj())
                return cls(noisy_POVM_list)
            
            elif noise_mode == 6: # Small rotation on first qubit only (same as 5 but with 0 on second angle)
                axis_1 = np.array([1,0,0])
                axis_2 = np.array([1/np.sqrt(2),1/np.sqrt(2),0])
                angle_1 = np.pi /5
                angle_2 = 0
                U_1 = sp.linalg.expm(-1/2j * angle_1 * np.einsum('j,jkl->kl',axis_1,sigma))
                U_2 = sp.linalg.expm(-1/2j * angle_2 * np.einsum('j,jkl->kl',axis_2,sigma))
                U = np.kron(U_1,U_2)
                Krauss_op=np.array([U],dtype=complex)
                noisy_POVM_list=np.einsum('nij,qjk,nlk->qil',Krauss_op,base_POVM_list,Krauss_op.conj())
                return cls(noisy_POVM_list)
            
            elif noise_mode == 7: # Strong 2 qubit depolarizing
                p=0.2
                new_list=p/2**n_qubits*np.eye(2**n_qubits) + (1-p)*base_POVM_list
                return cls(new_list)
                
            else:
                print(f'Invalid 2 qubit noise mode {noise_mode}, returning None.')
                return None
            
            
    
    @classmethod
    def generate_random_POVM(cls,dim,n_outcomes):
        """
        Generates random POVMs by generating a random positive semi-definite matrix.
        To make sure the final element is also positive we normalize the matrix eigenvalues
        such that their diagonals don't sum up to be greater than 1/n_outcomes. 
        In this way we are guaranteed that the the POVM normalisation is satisified and the final
        POVM element is also positive. 
        
        """
        POVM_list=np.zeros((n_outcomes,dim,dim),dtype=complex)
        matrixSize = dim
        for i in range (n_outcomes-1):
            A = np.random.normal(size=(matrixSize, matrixSize)) + 1j*np.random.normal(size=(matrixSize, matrixSize))
            #A*=1/np.real(np.trace(A))
            B=np.dot(A, A.conj().T)
            B*=1/(np.real(np.trace(B))*n_outcomes)
            
            POVM_list[i] = B
        POVM_list[-1]=np.eye(dim)-np.sum(POVM_list,axis=0)
        #random_POVM=POVM(POVM_list)
        #print(POVM_list)
        #print(np.trace(POVM_list[0]))
        #for i in range(n_outcomes):
        #    eigV,U=np.linalg.eig(POVM_list[i])
        #    print(eigV)
        return cls(POVM_list)
    
    @classmethod 
    def generate_Pauli_from_comp(cls, comp_POVM):
        """
        This function takes in a computational basis (could be reconstructed) and turns it a Pauli-6 basis
        by applying all possible rotations. This function scales exponentially. 

        Input:
            - comp_POVM: single computation-basis POVM object.

        Returns:
            - ndarray of rotated computational POVMs in the order XX, XY, XZ, YX ...
        """
        comp_list = comp_POVM.get_POVM()
        # Finds # qubits from dimension
        n_qubits = int(np.log2(len(comp_list[0]))) 
        sigma_x = np.array([[0,1], [1,0]])
        sigma_y = np.array([[0,-1j], [1j,0]])
        rot_to_x = sp.linalg.expm(-1j * np.pi/4 * sigma_y)
        rot_to_y = sp.linalg.expm(-1j * (-np.pi/4) * sigma_x)
        
        # Create list of single qubit rotations from comp to Pauli
        rot_list = np.array([rot_to_x, rot_to_y, np.eye(2)]) 
        
        # Creates all possible combinations of the single qubit list
        comb_list = np.array(list(product(rot_list, repeat=n_qubits))) 
        
        # Tensors together all elements in the list
        tensored_rot = np.array([reduce(np.kron, comb) for comb in comb_list]) 
        
        # Applies the rotations to the comp basis.
        new_mesh = np.einsum('nij, mjk, nkl->nmil', tensored_rot, comp_list, np.transpose(tensored_rot, axes=[0,2,1]).conj()) 
        return np.array([POVM(povm) for povm in new_mesh ])
        
    
    @classmethod
    def generate_Pauli_from_hash(cls, hash, n_symbols):
        """
        Takes in a single hashing function and generates the corresponding Pauli-6 measurement sequece for the hash. 
        E.g Input hash [1,0,0,1] would yield a output POVM array of size 9 (with two types the hash has 9 unique measurement (as for 2 qubits)),
        where each POVM element is a 2^4 x 2^4 matrix. 
        
        Input:
            hash = single numpy array of the size of the qubit system
            n_sybols = number of unique symbols in the hash
            
        Return:
            numpy array of POVM elements corresponding to the hashed function
        """
        n_qubit_subsystem = n_symbols # Number of unique symbols in the hash
        single_qubit_pauli = np.array([povm.get_POVM() for povm in POVM.generate_Pauli_POVM(1)])
        
        # Create all combinations of single qubit measurements to be masked by the hash
        # The final element is to reverse the order such that the left most entire is the left most qubit (in binary counting)
        comb_list = np.array(list(product(single_qubit_pauli, repeat=n_qubit_subsystem)))[:,::-1]
        hashed_list = comb_list[:,hash]
        # Tensor together the measuremers from the masked hashed list
        toal_POVM_list = np.array([reduce(np.kron,input) for input in hashed_list])

        return np.array([cls(povm) for povm in toal_POVM_list])
        
    def get_quantum_correlation_coefficient(self):
        """ 
        For two qubit POVMs one can compute how correlated the POVMs are. and extract a correlation coefficient. 
        This procedure follows eq. (7) and (5) from http://arxiv.org/abs/2311.10661
        
        The procedure will return both variants of the correlation coefficient, tracing down first the first qubit and then the second qubit.
        """
        
        # Check if the POVM is a two qubit POVM
        if len(self.POVM_list[0]) != 4:
            print("The POVM is not a two qubit POVM")
            return None



        def func(x, *args):
            M = args[0]
            qubit = args[1]
            vec1 = x[:3]
            vec2 = x[3:]
            sigma_vec = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]])
            Delta = np.einsum('ijk,i->jk',sigma_vec,vec1-vec2)
            if qubit==0:
                op = (M@np.kron(np.eye(2),Delta)).reshape(2,2,2,2)
                op = np.einsum('jklk->jl',op)
            else:
                op = (M@np.kron(Delta,np.eye(2))).reshape(2,2,2,2)
                op = np.einsum('kjkl->jl',op)
            return -1/2*np.linalg.norm(op, ord = 2) # Why do we devide by 2?

        def cons_1(x):
            return np.linalg.norm(x[:3])-1
        def cons_2(x):
            return np.linalg.norm(x[3:])-1
        bound = sp.optimize.Bounds(-1.000,1.000)
        cons = [{'type':'eq','fun':cons_1},{'type':'eq','fun':cons_2}]

        # Define M for tracing out qubit 0
        M0 = self.POVM_list[0] + self.POVM_list[1]
        # Define M for tracing out qubit 1
        M1 = self.POVM_list[0] + self.POVM_list[2]
        tolerance = {"ftol": 1e-8} 
        tol = 10**-8
        x0 = np.array([0,0,1,0,0,0])
        sol0 = minimize(func, x0, args=(M0,0), method='SLSQP', bounds=bound, constraints=cons, tol=tol, options=tolerance)
        x0 = np.array([0,0,-1,0,0,0])
        sol1 = minimize(func, x0, args=(M1,1), method='SLSQP', bounds=bound, constraints=cons, tol = tol, options=tolerance)
        return -np.array([sol0['fun'],sol1['fun']])
        
    
    
    
    def get_classical_correlation_coefficient(self):
        """ 
        For two qubit POVMs one can compute how correlated the POVMs are, optimized over only classical states (e.g. pure states in the computational basis. ) and extract a correlation coefficient. 
        This procedure follows eq. (7) and (5) from http://arxiv.org/abs/2311.10661.
        
        The procedure will return both variants of the correlation coefficient, tracing down first the first qubit and then the second qubit.
        """
        # Check if the POVM is a two qubit POVM
        if len(self.POVM_list[0]) != 4:
            print("The POVM is not a two qubit POVM")
            return None
        states = np.array([[[1,0], [0,0]],[[0,0], [0,1]] ])
        
        POVM_A = self.partial_trace(states[0],0)    
        POVM_B = self.partial_trace(states[1],0)
        c_0 = np.linalg.norm(POVM_A.get_POVM()[0] - POVM_B.get_POVM()[0], ord = 2)

        POVM_A = self.partial_trace(states[0],1)    
        POVM_B = self.partial_trace(states[1],1)
        c_1 = np.linalg.norm(POVM_A.get_POVM()[0] - POVM_B.get_POVM()[0], ord = 2)
  
        return np.array([c_0,c_1])
        
             
        
    
    def partial_trace(self, rho, qubit=0):
        """
        Traces down a two qubit POVM to a single qubit POVM. By default it traces out the 0 qubit. Following the procedute explained in  http://arxiv.org/abs/2311.10661.
        Rho single qubit state of the environment.
        """
        if len(self.POVM_list[0]) != 4:
            print("The POVM is not a two qubit POVM")
            return None
        
        povm_array = self.POVM_list
        

        if qubit == 0:
            op = np.kron(np.eye(2),rho)
            combined_op = np.einsum('ijk,kl->ijl',povm_array,op).reshape((-1,2,2,2,2))
            traced_down_povm = np.einsum('ijklk->ijl',combined_op)
            summed_povm = np.array([traced_down_povm[0] + traced_down_povm[1], traced_down_povm[2] + traced_down_povm[3]])
        else: 
            op = np.kron(rho, np.eye(2))
            combined_op = np.einsum('ijk,kl->ijl',povm_array,op).reshape((-1,2,2,2,2))
            traced_down_povm = np.einsum('ikjkl->ijl',combined_op)
            summed_povm = np.array([traced_down_povm[0] + traced_down_povm[2],traced_down_povm[1] + traced_down_povm[3]])
        
        return POVM(summed_povm)