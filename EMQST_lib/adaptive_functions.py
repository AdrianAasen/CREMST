import numpy as np
#plt.rcParams.update({'font.size': 22})

from joblib import Parallel, delayed
from datetime import datetime
from scipy.stats import unitary_group
import qutip as qt
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
import jax
from EMQST_lib import measurement_functions as mf
from EMQST_lib import visualization as vis
from EMQST_lib import support_functions as sf
from EMQST_lib.povm import POVM

def adaptive_cost_function(angles,rho_bank,weights,best_guess,n_qubits):
    """
    Computes the expected entropy reduction of the posterior (likelihood) distribution.
    The angles are taken in as a dictionary and indicate what mesaurement is to be perfomred.
    Noise correction is currently removed.  
    """
    # Crates projector from angles
    projective_vector = angles_to_state_vector(angles, n_qubits)


    out = jnp.einsum('ij,ik->ijk',projective_vector,projective_vector.conj())
    # Computes the entropy of prior and posterior distributions. See 10.1103/PhysRevA.85.052120 for more details.
    K = Shannon(jnp.einsum('ijk,kj->i',out,best_guess))
    J = Shannon(jnp.einsum('ijk,lkj->il',out,rho_bank))
    # Returns the negative values such that it becomes a minimization problem rather than maximization problem.
    return -jnp.real(K-jnp.dot(J,weights))


def adaptive_cost_function_array(angles_array, rho_bank, weights, best_guess, n_qubits):
    """
    JAX-compatible version of adaptive_cost_function that takes angles as an array.
    
    Parameters:
    -----------
    angles_array : jnp.array
        - For 1 qubit: [theta, phi]
        - For 2 qubits: [theta_A, phi_A, theta_B, phi_B]
    """
    # Create projective vector directly from array
    projective_vector = angles_array_to_state_vector(angles_array, n_qubits)
    
    out = jnp.einsum('ij,ik->ijk', projective_vector, projective_vector.conj())
    # Computes the entropy of prior and posterior distributions. See 10.1103/PhysRevA.85.052120 for more details.
    K = Shannon(jnp.einsum('ijk,kj->i', out, best_guess))
    J = Shannon(jnp.einsum('ijk,lkj->il', out, rho_bank))
    # Returns the negative values such that it becomes a minimization problem rather than maximization problem.
    return -jnp.real(K - jnp.dot(J, weights))


def angles_array_to_state_vector(angles_array, n_qubits):
    """
    JAX-compatible version that takes angles as an array instead of dictionary.
    
    Parameters:
    -----------
    angles_array : jnp.array
        - For 1 qubit: [theta, phi]  
        - For 2 qubits: [theta_A, phi_A, theta_B, phi_B]
    """
    if n_qubits == 1:
        theta, phi = angles_array[0], angles_array[1]
        tempMesh = jnp.array([jnp.cos(theta/2), jnp.exp(1j*phi)*jnp.sin(theta/2)])
        meshState = jnp.array([tempMesh, get_opposing_state(tempMesh)])
    elif n_qubits == 2:
        theta_A, phi_A, theta_B, phi_B = angles_array[0], angles_array[1], angles_array[2], angles_array[3]
        tempMeshA = jnp.array([jnp.cos(theta_A/2), jnp.exp(1j*phi_A)*jnp.sin(theta_A/2)])
        tempMeshB = jnp.array([jnp.cos(theta_B/2), jnp.exp(1j*phi_B)*jnp.sin(theta_B/2)])
        meshA = jnp.array([tempMeshA, get_opposing_state(tempMeshA)])
        meshB = jnp.array([tempMeshB, get_opposing_state(tempMeshB)])
        meshState = jnp.array([jnp.kron(meshA[0],meshB[0]), jnp.kron(meshA[0],meshB[1]), 
                              jnp.kron(meshA[1],meshB[0]), jnp.kron(meshA[1],meshB[1])])
    else:
        raise ValueError(f"n_qubits={n_qubits} not supported")
    return meshState


def Shannon(prob):
    """
    Returns the Shannon entropy of the probability histogram. 
    """
    return jnp.real(jnp.sum(-(prob*jnp.log2(prob)),axis=0))



def angles_to_state_vector(angles,n_qubits):
    """
    Jax version of AnglesToStateVector.
    """
    if n_qubits==1:
        tempMesh=jnp.array([jnp.cos(angles["theta"]/2),jnp.exp(1j*angles["phi"])*jnp.sin(angles["theta"]/2)])
        meshState=jnp.array([tempMesh,get_opposing_state(tempMesh)])
    else:
        tempMeshA=jnp.array([jnp.cos(angles["theta_A"]/2),jnp.exp(1j*angles["phi_A"])*jnp.sin(angles["theta_A"]/2)])
        tempMeshB=jnp.array([jnp.cos(angles["theta_B"]/2),jnp.exp(1j*angles["phi_B"])*jnp.sin(angles["theta_B"]/2)])
        meshA=jnp.array([tempMeshA,get_opposing_state(tempMeshA)])
        meshB=jnp.array([tempMeshB,get_opposing_state(tempMeshB)])
        meshState=jnp.array([jnp.kron(meshA[0],meshB[0]),jnp.kron(meshA[0],meshB[1]),jnp.kron(meshA[1],meshB[0]),jnp.kron(meshA[1],meshB[1])])
    return meshState

def get_opposing_state(meshState):
    """
    Returns orthogonal state for an arbitrary single qubit state.
    Jax version of getOpposingState. It has a small bug if meshState[0]==1 and should return np.array([0, 1],dtype=complex).
    """
    # if meshState[1]==0:
    #    return np.array([0, 1],dtype=complex)

    a=1
    b=-jnp.conjugate(meshState[0])/jnp.conjugate(meshState[1])
    norm=jnp.sqrt(a*jnp.conjugate(a) + b*jnp.conjugate(b))
    oppositeMeshState=jnp.array([a/norm, b/norm])
    return oppositeMeshState

def generate_random_angles(n_qubits):
    """
    Generate random angles for each qubit uniformly distributed on the sphere.
    Returns angles in the same format as rho_to_angles for consistency.
    
    For single qubit: theta in [0, pi], phi in [0, 2*pi]
    For two qubits: separate random angles for each qubit
    """
    import numpy as np
    
    if n_qubits == 1:
        # Uniform distribution on sphere: cos(theta) uniform in [-1,1], phi uniform in [0,2*pi]
        cos_theta = np.random.uniform(-1, 1)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0, 2*np.pi)
        
        return {
            "theta": jnp.array(theta),
            "phi": jnp.array(phi)
        }
    
    elif n_qubits == 2:
        # Generate random angles for each qubit independently
        cos_theta_A = np.random.uniform(-1, 1)
        theta_A = np.arccos(cos_theta_A)
        phi_A = np.random.uniform(0, 2*np.pi)
        
        cos_theta_B = np.random.uniform(-1, 1)
        theta_B = np.arccos(cos_theta_B)
        phi_B = np.random.uniform(0, 2*np.pi)
        
        return {
            "theta_A": jnp.array(theta_A),
            "phi_A": jnp.array(phi_A),
            "theta_B": jnp.array(theta_B),
            "phi_B": jnp.array(phi_B)
        }
    
    else:
        raise ValueError("n_qubits must be 1 or 2")
    
    
    
def rho_to_angles(rho,nQubits):
    positive_Pauli_eigenstates=np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),1j*1/np.sqrt(2)],[1,0]])
    #negative_Pauli_eigenstates=np.array([[1/np.sqrt(2),-1/np.sqrt(2)],[1/np.sqrt(2),-1j*1/np.sqrt(2)],[0,1]])
    if nQubits==1:
        xup=qubit_histogram(jnp.array(rho),jnp.array([positive_Pauli_eigenstates[0]]))[0]
        yup=qubit_histogram(jnp.array(rho),jnp.array([positive_Pauli_eigenstates[1]]))[0]
        zup=qubit_histogram(jnp.array(rho),jnp.array([positive_Pauli_eigenstates[2]]))[0]
        phi=jnp.arctan2(2*yup-1,2*xup-1)
        theta=jnp.arccos((2*zup-1)/jnp.sqrt((2*xup-1)**2 + (2*yup-1)**2 + (2*zup-1)**2 ))

        angles={
            "phi": jnp.array(phi),
            "theta": jnp.array(theta)
        }
    else:
        temp_rho=jnp.reshape(rho,(2,2,2,2))
        rhoA=jnp.trace(temp_rho,axis1=1,axis2=3)
        rhoB=jnp.trace(temp_rho,axis1=0,axis2=2)
        anglesA=rho_to_angles(rhoA,1)
        anglesB=rho_to_angles(rhoB,1)
        angles={
            "phi_A": anglesA["phi"],
            "theta_A": anglesA["theta"],
            "phi_B": anglesB["phi"],
            "theta_B": anglesB["theta"]
        }
    return angles
    
def qubit_histogram(rho,projective_vectors):
    return jnp.real(jnp.einsum('ji,ik,jk->j',projective_vectors.conj(),rho,projective_vectors))


# def generate_random_projective_2q_povm():
#     """
#     Generates a random two-qubit projective measurement by taking the tensor product of two random single-qubit states.
#     Each single-qubit state is uniformly distributed on the Bloch sphere.
#     Returns a 4x4 numpy array where each row corresponds to a measurement outcome.
#     """
#     def generate_randomMeshState():
#         phi=2*np.pi*np.random.random()
#         theta=np.arccos(1-2*np.random.random())
#         mesh1=np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
#         meshState=np.array([mesh1, get_opposing_state(mesh1)])
#         return meshState
#     mesh1=generateRandomMeshState()
#     mesh2=generateRandomMeshState()
#     meshState=np.array([np.kron(mesh1[0],mesh2[0]),np.kron(mesh1[0],mesh2[1]),np.kron(mesh1[1],mesh2[0]),np.kron(mesh1[1],mesh2[1])])
#     mesh_operators = np.einsum('ij,ik->ijk',meshState,meshState.conj())

#     return POVM(mesh_operators)
