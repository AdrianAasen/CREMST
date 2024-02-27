import numpy as np
import scipy as sp
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm


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
        #self.POVM_dimension=len(POVM_list[-1])
        #self.n_elemetns=len(POVM_list)

    
    
    @classmethod
    def POVM_from_angles(cls,angles):
        """
        Creates a POVM class based on a set of angles defining spin measurements. 
        angles: ndarray n x 2 (n qubits, 2 angles [theta,phi])
                angles defines what is considere the 'up' outcome. 
        return POVM class. 
        """
        
        angle_representation=np.zeros((2**len(angles),len(angles),2))

        opposite_angles=sf.get_opposing_angles(angles)
        angle_Matrix=np.array([angles,opposite_angles])
        projector_up=np.zeros((len(angles),2,2),dtype=complex)
        projector_down=np.zeros((len(angles),2,2),dtype=complex)
        #print(opposite_angles)
        for i in range(len(angles)):
            projector_up[i]=sf.get_projector_from_angles(np.array([angles[i]]))
            projector_down[i]=sf.get_projector_from_angles(np.array([opposite_angles[i]]))
        projector_matrix=np.array([projector_up,projector_down]) # creates matrix with [up/down index, qubit positition, 2x2 matrix]
        #print(projector_matrix)
        POVM_list=np.zeros((2**len(angles),2**len(angles),2**len(angles)),dtype=complex)
        M_temp=1
        for i in range (2**len(angles)): # iterates over all possible bitstrings 
            bit_string=bin(i)[2:].zfill(len(angles)) # Creates binary string of all possible permutations
            M_temp=1
            for j in range (len(angles)):
                M_temp=np.kron(M_temp,projector_matrix[int(bit_string[j]),j])
                angle_representation[i,j]=angle_Matrix[int(bit_string[j]),j]
            POVM_list[i]=M_temp
            
            #print(M_temp)
        return cls(POVM_list,angle_representation)
    
    @classmethod
    def generate_Pauli_POVM(cls,n_qubits):
        """
        Recursivly create higher qubit POVMs
        Returns a list of 3 spin POVMs along x, y and z axis. 
        """
        POVM_set_X=1/2*np.array([[[1,1],[1,1]],[[1,-1],[-1,1]]],dtype=complex)
        POVM_set_Y=1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)
        POVM_set_Z=np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex)
        #POVM_matrix_lsit=np.array([POVM_set_X,POVM_set_Y,POVM_set_Z])
        POVM_X=cls(POVM_set_X,np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]]]))
        POVM_Y=cls(POVM_set_Y,np.array([[[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]]]))
        POVM_Z=cls(POVM_set_Z,np.array([[[0,0]],[[np.pi,0]]]))
        POVM_single=np.array([POVM_X,POVM_Y,POVM_Z])
        POVM_list=np.copy(POVM_single)
        recursion=n_qubits
        while recursion>1:
            POVM_list=POVM.tensor_POVM(POVM_list,POVM_single)
            recursion-=1
            
        return POVM_list
    
    @classmethod
    def tensor_POVM(cls,POVM_1,POVM_2):
        POVM_list=np.array([cls(np.array([np.kron(a,b) for a in POVM_a.get_POVM() for b in POVM_b.get_POVM()]),
                                np.array([np.concatenate((angle_a,angle_b)) for angle_a in POVM_a.get_angles() for angle_b in POVM_b.get_angles()] ))
                             for POVM_a in POVM_1 for POVM_b in POVM_2])
        return POVM_list
        
    
    @classmethod
    def empty_POVM(cls):
        """
        Returns an empty class
        """
        return cls(np.array([]),np.array([]))
    
    @classmethod
    def z_axis_POVM(cls,n_qubits):
        """
        Returns z-basis POVM.
        Currently only supports single qubits
        """
        return cls(np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex),np.array([[[0,0],[np.pi,0]]]))

    
    def get_histogram(self,rho):
        """
        Takes in state of arbitrary dimension.
        Returns histogram for all probabilities of outcomes defined by POVM.
        State and POVM dimension must be compatible. 
        """
        return np.real(np.einsum('ijk,kj->i',self.POVM_list,rho))
    
    def get_POVM(self):
        return np.copy(self.POVM_list)
    
    def get_angles(self):
        return np.copy(self.angle_representation)

    @classmethod
    def depolarized_POVM(cls,base_POVM,strenght=0.1):
        base_POVM_list=base_POVM.get_POVM()
        dim=int(len(base_POVM_list[0]))
        new_list=strenght/dim*np.eye(dim) + (1-strenght)*base_POVM_list
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
    
        
