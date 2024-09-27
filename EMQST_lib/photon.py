import numpy as np
from functools import reduce
from itertools import product, chain, repeat, combinations
from EMQST_lib import support_functions as sf
from EMQST_lib.povm import POVM
from EMQST_lib import dt


def photon_MLE(OP_list: np.array, index_counts: np.array ):
    '''
    Estimates state according to iterative MLE.
    :param full_operator_list: full list of POVM elemnts
    :outcome_index: list of all outcome_indices such that
            full_operator_list[index] = POVM element
    :return: dxd array of iterative MLE estimator
    '''
    dim = OP_list.shape[-1]

    iter_max = 500
    dist     = float(1)

    rho_1 = np.eye(dim)/dim
    rho_2 = np.eye(dim)/dim
    j = 0

    #unique_index, index_counts=np.unique(outcome_index,return_counts=True)
    #OP_list=full_operator_list[unique_index]
    #print(f" op list{np.sum(OP_list,axis = 0)}")
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


def photon_label_to_operator(name_list):
    # Define the eigenstates
    plus_x = np.array([1, 1]) / np.sqrt(2)
    minus_x = np.array([1, -1]) / np.sqrt(2)

    plus_y = np.array([1, 1j]) / np.sqrt(2)
    minus_y = np.array([1, -1j]) / np.sqrt(2)

    plus_z = np.array([1, 0])
    minus_z = np.array([0, 1])

    # Define density matrices
    xup = np.outer(plus_x, np.conj(plus_x))
    xdown = np.outer(minus_x, np.conj(minus_x))

    yup = np.outer(plus_y, np.conj(plus_y))
    ydown = np.outer(minus_y, np.conj(minus_y))

    zup = np.outer(plus_z, np.conj(plus_z))
    zdown = np.outer(minus_z, np.conj(minus_z))
    # Define translation dictionary
    op_dict = {
        'H': zup,
        'V': zdown,
        'L': yup,
        'R': ydown,
        'D': xup,
        'A': xdown
    }
    if len(name_list[0]) == 1:
        return np.array([op_dict[name] for name in name_list])
    elif len(name_list[0]) == 2:
        return np.array([np.kron(op_dict[name[0]], op_dict[name[1]]) for name in name_list])
    else:  
        raise ValueError("Operator name list is not of the right format")
    
    
    
def coincidence_to_states(coincidence_counts, operator_order, n_qubits = 2):
    """
    Takes in experimental coincidence counts and an operator order for the counts and returns the estimated state.
    
    Steps:
    Need to turn the coincidence counts into proper probabilities and the 
    set of operators into a proper set of POVMs by adding a normalization operator. 
    """
    # Flatten standard outcome
    coincidence_counts = coincidence_counts.flatten()

    # Add additional operators for each projection, such that each operator becomes a POVM.
    # The additional operators are the identity minus the operator, making each two operator identity. 
    additional_op = np.array([np.eye(2**n_qubits) - op for op in operator_order])
    # Compute the total number of coincidences for each POVM on the 4 first projections. 
    n_partial = np.sum(coincidence_counts[:2**n_qubits]) 
    
    additional_coincidenses = n_partial - coincidence_counts

    #print(np.sum(operator_order[:4], axis = 0))
    # Include the additional operators and coincidences in the list of operators and coincidences, making them full POVMs
    new_op = np.append(operator_order,additional_op[2**n_qubits:],axis = 0)
    new_coincidenes = np.append(coincidence_counts, additional_coincidenses[2**n_qubits:])
    rho = photon_MLE(new_op, new_coincidenes)
    
    return rho

def coincidence_to_POVM(QDT_coincidence, calib_states):
    """
    Takes in coincidence counts the calibration state  and returns the POVMs for [H,V] [R,L] and [D,A].
    Currently requires manual setting of order_list if the input order is not the default. 
    The default measurement order is [H,V,R,D], if other than this specify with measurement_order. 
    The QDT coincidences has shape [n_calib_states, n_measurements]
    """
    # if measurement_order is None:
    #    measurement_order = ['H', 'V', 'R', 'D']
    
    # Create inital guess states for QDT MLE
    inital_guess_POVM = POVM.generate_random_POVM(2, 2)
    
    # Find total shot count for each state
    n_shot_total = np.array([QDT_coincidence[i,0] + QDT_coincidence[i,1] for i in range(len(QDT_coincidence))])
    

    # Create coincidences for the other measurements. 
    additional_coincidenses = np.array([n_shot_total[0] - QDT_coincidence[i] for i in range(len(QDT_coincidence))],dtype=int)

    new_coincidenes = np.concatenate((QDT_coincidence,additional_coincidenses[:,2:]),axis = 1)

    new_coincidenes[new_coincidenes < 0] = 0
    # Create 3 sets of two outcome Paulies.
    # Order array set such that new_coincidences gives this order: [[H,V],[R,L],[D,A]] 
    order_list = np.array([[0,1], [4, 2], [3,5]])
    povm_recon = [dt.POVM_MLE(1, new_coincidenes[:,order]+1, calib_states[:], inital_guess_POVM) for order in order_list]
    return povm_recon


def simulate_photon_coincidence_counts(N_shots, states, projectors):
    prob = np.real(np.einsum('mij,nji->mn',states, projectors))
    return N_shots*prob