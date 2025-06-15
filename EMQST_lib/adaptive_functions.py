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

def adaptive_cost_function(angles,rho_bank,weights,best_guess,n_qubits,noiseCorrection=0):
    """
    Computes the expected entropy reduction of the posterior (likelihood) distribution.
    The angles are taken in as a dictionary and indicate what mesaurement is to be perfomred.
    Noise correction is currently removed.  
    """
    # Crates projector from angles
    projective_vector = angles_to_state_vector(angles, n_qubits)
    #out=jnp.einsum('ij,ik->ijk',meshState,meshState.conj())
    #theta=angles["theta"]%(2*jnp.pi)
    
    #r=0#noiseCorrection
    #q=jnp.exp(-r)#jnp.exp(-r*((jnp.pi/2-jnp.abs(theta-jnp.pi/2))*jnp.heaviside(jnp.pi-theta,0) + (jnp.pi/2-jnp.abs(theta-3*jnp.pi/2))*jnp.heaviside(theta-jnp.pi,0)))
    out=jnp.einsum('ij,ik->ijk',projective_vector,projective_vector.conj()) #+ 1/2*(1-q)*jnp.eye(2**n_qubits)
    # Computes the entropy of prior and posterior distributions. See 10.1103/PhysRevA.85.052120 for more details.
    K = Shannon(jnp.einsum('ijk,kj->i',out,best_guess))
    J = Shannon(jnp.einsum('ijk,lkj->il',out,rho_bank))
    # Returns the negative values such that it becomes a minimization problem rather than maximization problem.
    return -jnp.real(K-jnp.dot(J,weights))

def rho_to_angles(rho):
    """
    Converts a density matrix to angles.
    Uses jax-friendly numpy operations.
    """
    # Pauli matrices
    sigma_x = jnp.array([[0, 1], [1, 0]])
    sigma_y = jnp.array([[0, -1j], [1j, 0]])
    sigma_z = jnp.array([[1, 0], [0, -1]])
    
    # Bloch vector components
    x = jnp.real(jnp.trace(rho @ sigma_x))
    y = jnp.real(jnp.trace(rho @ sigma_y))
    z = jnp.real(jnp.trace(rho @ sigma_z))
    
    # Compute Bloch angles
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    return {"theta": jnp.array(theta), "phi": jnp.array(phi)}

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
        tempMeshA=jnp.array([jnp.cos(angles["thetaA"]/2),jnp.exp(1j*angles["phiA"])*jnp.sin(angles["thetaA"]/2)])
        tempMeshB=jnp.array([jnp.cos(angles["thetaB"]/2),jnp.exp(1j*angles["phiB"])*jnp.sin(angles["thetaB"]/2)])
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